# -*-coding: utf-8 -*-

import sys
import os
import pickle
import time
import json
from collections import deque
import copy
import random

from .agent import Agent
class RunningSteward(object):
    """
    The steward of running the dialogue system.
    """
    def __init__(self, parameter, wandb):
        self.epoch_size = parameter.get("simulation_size",100)
        slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
        disease_set = pickle.load(file=open(parameter["disease_symptom"], "rb"))
        self.agent = Agent(slot_set=slot_set, disease_set=disease_set, goal_set = goal_set, parameter=parameter)

        self.best_result = {"success_rate":0.0, "average_reward": 0.0, "average_turn": 0,"average_wrong_disease":10}

    def simulate(self, epoch_number, train_mode=False):
        """
        Simulating the dialogue session between agent and user simulator.
        :param agent: the agent used to simulate, an instance of class Agent.
        :param epoch_number: the epoch number of simulation.
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: nothing to return.
        """
        # initializing the count matrix for AgentWithGoal
        # print('Initializing the count matrix for AgentWithGoal')
        # self.simulation_epoch(epoch_size=500, train_mode=train_mode)
        save_model = self.parameter.get("save_model")
        save_performance = self.parameter.get("save_performance")
        # self.dialogue_manager.state_tracker.user.set_max_turn(max_turn=self.parameter.get('max_turn'))
        for index in range(0, epoch_number,1):
            # Training AgentDQN with experience replay
            if train_mode is True:
                self.dialogue_manager.train()
                # Simulating and filling experience replay pool.
                self.simulation_epoch(epoch_size=self.epoch_size, index=index)
            else:
                result = self.evaluate_model(dataset="test", index=index)
                if result["success_rate"] > self.best_result["success_rate"] and \
                        result["success_rate"] > dialogue_configuration.SUCCESS_RATE_THRESHOLD and train_mode==True :
                        #result["average_wrong_disease"] <= self.best_result["average_wrong_disease"] and \
                    self.dialogue_manager.state_tracker.agent.flush_pool()
                    self.simulation_epoch(epoch_size=self.epoch_size, index=index)
                    if save_model is True:
                        self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                        if self.parameter.get("agent_id").lower() in ["agenthrljoint", "agenthrljoint2",'agentdqn']:
                            self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index,
                                                                checkpoint_path=self.checkpoint_path)
                        print("###########################The model was saved.###################################")
                    else:
                        pass
                    self.best_result = copy.deepcopy(result)
        # The training is over and save the model of the last training epoch.
        if save_model is True and train_mode is True and epoch_number > 0:
            self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
            if self.parameter.get("agent_id").lower() in ["agenthrljoint","agenthrljoint2"]:
                self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index, checkpoint_path=self.checkpoint_path)
        if save_performance is True and train_mode is True and epoch_number > 0:
            self.__dump_performance__(epoch_index=index)

    def simulation_epoch(self, epoch_size, index):
        """
        Simulating one epoch when training model.
        :param epoch_size: the size of each epoch, i.e., the number of dialogue sessions of each epoch.
        :return: a dict of simulation results including success rate, average reward, average number of wrong diseases.
        """
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        inform_wrong_disease_count = 0
        num_of_true_slots = 0
        num_of_implicit_slots = 0
        real_implicit_slots = 0
        avg_f1_total = 0
        avg_recall_total = 0
        self.dialogue_manager.state_tracker.agent.train_mode() # 
        inform_wrong_disease_count = 0
        for epoch_index in range(0,epoch_size, 1):
            self.dialogue_manager.initialize(dataset="train")
            episode_over = False
            while episode_over is False:
                reward, episode_over, dialogue_status,slots_proportion_list= self.dialogue_manager.next(greedy_strategy=True, save_record=True, index=index)
                total_reward += reward
            total_turns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.status_fail
            assert len(slots_proportion_list)>0
            num_of_true_slots+=slots_proportion_list[0]
            num_of_implicit_slots+=slots_proportion_list[1]
            real_implicit_slots += slots_proportion_list[2]
            avg_f1_total += slots_proportion_list[0] * 2 / (slots_proportion_list[1] + slots_proportion_list[2])
            avg_recall_total += slots_proportion_list[0] / (slots_proportion_list[2]+1e-10)
                
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
            '''
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
            '''
        evaluate_session_number = epoch_size
        success_rate = float("%.3f" % (float(success_count) / evaluate_session_number))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / evaluate_session_number))
        average_reward = float("%.3f" % (float(total_reward) / evaluate_session_number))
        average_turn = float("%.3f" % (float(total_turns) / evaluate_session_number))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / evaluate_session_number))
        all_recall = float("%.3f" % (float(num_of_true_slots) / float(real_implicit_slots)))
        avg_recall = float("%.3f" % (float(avg_recall_total) / evaluate_session_number))
        avg_f1 = float("%.3f" % (float(avg_f1_total) / evaluate_session_number))
        all_f1 = float("%.3f" % (float(num_of_true_slots * 2) / (num_of_implicit_slots+real_implicit_slots)))
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn,
               "average_wrong_disease":average_wrong_disease,"ab_success_rate":absolute_success_rate}
        # print("%3d simulation success rate %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        # self.dialogue_manager.state_tracker.agent.eval_mode() # for training
        return res

    def evaluate_model(self, dataset, index):
        """
        Evaluating model during training.
        :param index: int, the simulation index.
        :return: a dict of evaluation results including success rate, average reward, average number of wrong diseases.
        """
        if self.parameter.get("use_all_labels"):
            self.dialogue_manager.repeated_action_count = 0
            self.dialogue_manager.group_id_match = 0
        if self.parameter.get("initial_symptom"):
            self.dialogue_manager.group_id_match = 0
        self.dialogue_manager.repeated_action_count = 0
        save_performance = self.parameter.get("save_performance")
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        #evaluate_session_number = len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        dataset_len=len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        evaluate_session_number=self.parameter.get("evaluate_session_number")
        evaluate_session_index = random.sample(range(dataset_len), dataset_len)
        inform_wrong_disease_count = 0
        num_of_true_slots = 0
        num_of_implicit_slots = 0
        real_implicit_slots = 0
        avg_f1_total = 0
        avg_recall_total = 0
        #for goal_index in range(0,evaluate_session_number, 1):
        for goal_index in evaluate_session_index:
            self.dialogue_manager.initialize(dataset=dataset, goal_index=goal_index)
            episode_over = False
            while episode_over == False:
                reward, episode_over, dialogue_status,slots_proportion_list = self.dialogue_manager.next(
                    save_record=False,greedy_strategy=False, index=index)
                total_reward += reward
            assert len(slots_proportion_list)>0
            num_of_true_slots+=slots_proportion_list[0]
            num_of_implicit_slots+=slots_proportion_list[1]
            real_implicit_slots += slots_proportion_list[2]
            avg_f1_total += slots_proportion_list[0] * 2 / (slots_proportion_list[1] + slots_proportion_list[2] + 1e-10)
            avg_recall_total += slots_proportion_list[0] / (slots_proportion_list[2] + 1e-10)
            #(slots_proportion_list)
            total_turns += self.dialogue_manager.state_tracker.turn
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
        success_rate = float("%.3f" % (float(success_count) / evaluate_session_number))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / evaluate_session_number))
        average_reward = float("%.3f" % (float(total_reward) / evaluate_session_number))
        average_turn = float("%.3f" % (float(total_turns) / evaluate_session_number))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / evaluate_session_number))
        all_recall = float("%.3f" % (float(num_of_true_slots) / float(real_implicit_slots)))
        avg_recall = float("%.3f" % (float(avg_recall_total) / evaluate_session_number))
        avg_f1 = float("%.3f" % (float(avg_f1_total) / evaluate_session_number))
        all_f1 = float("%.3f" % (float(num_of_true_slots * 2) / (num_of_implicit_slots+real_implicit_slots)))
        if num_of_implicit_slots>0:
            #match rate表示agent所问道的症状当中是病人确实有的概率为多大。match rate2表示病人有多少比例的隐形症状被agent问出。
            match_rate=float("%.3f" %(float(num_of_true_slots)/float(num_of_implicit_slots)))
        else:
            match_rate=0.0
        average_repeated_action = float("%.4f" % (float(self.dialogue_manager.repeated_action_count) / evaluate_session_number))

        self.dialogue_manager.state_tracker.agent.train_mode() # for training.
        res = {
            "success_rate":success_rate,
            "average_reward": average_reward,
            "average_turn": average_turn,
            "average_repeated_action": average_repeated_action,
            "all_recall": all_recall,
            "all_f1": all_f1,
            "avg_recall":avg_recall,
            "avg_f1": avg_f1
        }
        self.learning_curve.setdefault(index, dict())
        self.learning_curve[index]["success_rate"]=success_rate
        self.learning_curve[index]["average_reward"]=average_reward
        self.learning_curve[index]["average_turn"] = average_turn
        #self.learning_curve[index]["average_wrong_disease"]=average_wrong_disease
        self.learning_curve[index]["average_match_rate"]=match_rate
        self.learning_curve[index]["all_recall"] = all_recall
        self.learning_curve[index]["average_repeated_action"] = average_repeated_action

        if save_performance:
            self.wandb.log({'success_rate': success_rate, 'average_turn': average_turn, 'average_reward': average_reward, 'all_recall': all_recall, 'all_f1': all_f1, 'average_repeated_action': average_repeated_action})



        if index % 10 ==9:
            print('[INFO]', self.parameter["run_info"])
        if self.parameter.get("classifier_type")=="deep_learning" and self.parameter.get("disease_as_action") == False and self.parameter.get("train_mode"):
            self.dialogue_manager.train_deep_learning_classifier(index, epochs=20)
        '''
        if index % 1000 == 999 and save_performance == True:
            self.__dump_performance__(epoch_index=index)
        '''
        print("%3d simulation SR [%s], ave reward %s, ave turns %s, all_f1 %s, all_recall %s, ave repeated %s, avg recall %s, avg f1 %s" % (index,res['success_rate'],res['average_reward'], res['average_turn'], res["all_f1"],res["all_recall"],res["average_repeated_action"],res['avg_recall'], res['avg_f1']))

        return res

    def warm_start(self):
        """
        Warm-starting the dialogue, using the sample from rule-based agent to fill the experience replay pool for DQN.
        :param agent: the agent used to warm start dialogue system.
        :param epoch_number: the number of epoch when warm starting, and the number of dialogue sessions of each epoch
                             equals to the simulation epoch.
        :return: nothing to return.
        """
        self.agent.warm_start()
