# -*-coding: utf-8 -*-

from inspect import Parameter
import sys
import os
import pickle
import time
import json
from collections import deque
import copy
import json
sys.path.append(os.getcwd().replace("HRL/dialogue_system/run",""))

from OpenMedicalChatBox.HRL.agent import AgentRule
from OpenMedicalChatBox.HRL.agent import AgentDQN
from OpenMedicalChatBox.HRL.user_simulator import UserRule as User
from OpenMedicalChatBox.HRL.dialogue_manager import DialogueManager
from OpenMedicalChatBox.HRL.dialogue_manager import DialogueManager_HRL
from OpenMedicalChatBox.HRL import dialogue_configuration
import random

class RunningSteward(object):
    """
    The steward of running the dialogue system.
    """
    def __init__(self, parameter, checkpoint_path):
        self.epoch_size = parameter.get("simulation_size",100)
        self.parameter = parameter
        self.checkpoint_path = checkpoint_path
        self.learning_curve = {}


        slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        #action_set = pickle.load(file=open(parameter["action_set"], "rb"))
        goal_set = pickle.load(file=open(parameter["goal_set"], "rb"))
        disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))

        exam_set = pickle.load(file=open(parameter["test_set"], "rb"))
        disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
        #disease_exam = pickle.load(file=open(parameter["disease_test"], "rb"))
        user = User(goal_set=goal_set, disease_syptom=disease_symptom, exam_set = exam_set, parameter=parameter)
        agent = AgentRule(slot_set=slot_set, disease_symptom=disease_symptom, parameter=parameter)
        if parameter.get("use_all_labels"):
            self.dialogue_manager = DialogueManager_HRL(user=user, agent=agent, parameter=parameter)
        else:
            self.dialogue_manager = DialogueManager(user=user, agent=agent, parameter=parameter)

        if self.parameter.get("disease_as_action") == False:
            if self.parameter.get("classifier_type") == "machine_learning":
                self.dialogue_manager.train_ml_classifier()
                print("############   the machine learning model is training  ###########")
            elif self.parameter.get("classifier_type") == "deep_learning":
                self.dialogue_manager.build_deep_learning_classifier()
            else:
                raise ValueError("the classifier type is not among machine_learning and deep_learning")

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
        
        self.dialogue_manager.state_tracker.agent.eval_mode()

        # self.dialogue_manager.state_tracker.user.set_max_turn(max_turn=self.parameter.get('max_turn'))
        for index in range(0, epoch_number,1):
            # Training AgentDQN with experience replay
            if train_mode is True:
                self.dialogue_manager.train()
                # Simulating and filling experience replay pool.
                self.simulation_epoch(epoch_size=self.epoch_size, index=index)

            # Evaluating the model.
            #print(index)
            #  and result1["success_rate"] > self.best_result["success_rate"]
            #if self.parameter['data_type'] == 'real':
            #result = self.evaluate_model(dataset="validate", index=index)
            result = self.evaluate_model(dataset="test", index=index)
            if (result["success_rate"] > self.best_result["success_rate"] and \
                    result["success_rate"] > dialogue_configuration.SUCCESS_RATE_THRESHOLD) and train_mode==True:
                    #result["average_wrong_disease"] <= self.best_result["average_wrong_disease"] and \
                self.dialogue_manager.state_tracker.agent.flush_pool()
                self.simulation_epoch(epoch_size=self.epoch_size, index=index)
                if save_model is True:
                    self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                    if self.parameter.get("agent_id").lower() in ["agenthrljoint", "agenthrljoint2",'agentdqn']:
                        if self.parameter.get("disease_as_action") == False:
                            self.dialogue_manager.save_dl_model(model_performance=result, episodes_index=index,
                                                                checkpoint_path=self.checkpoint_path)
                    print("###########################The model was saved.###################################")
                else:
                    pass
                self.best_result = copy.deepcopy(result)
            '''
            else:
            result = self.evaluate_model(dataset="test", index=index)
            if (result["success_rate"] > self.best_result["success_rate"] and \
                    result["success_rate"] > dialogue_configuration.SUCCESS_RATE_THRESHOLD) and (index > 250) and train_mode==True :
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
            '''
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
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        inform_wrong_disease_count = 0
        if self.parameter['sequential_sampling']:
            epoch_size = len(self.dialogue_manager.state_tracker.user.goal_set['train'])
        for epoch_index in range(0,epoch_size, 1):
            if self.parameter['sequential_sampling']:
                self.dialogue_manager.initialize(dataset="train", goal_index= epoch_index)
            else:
                self.dialogue_manager.initialize(dataset="train")
            episode_over = False
            while episode_over is False:
                reward, episode_over, dialogue_status,slots_proportion_list= self.dialogue_manager.next(greedy_strategy=True, save_record=True, index=index)
                total_reward += reward
            total_turns += self.dialogue_manager.state_tracker.turn/2
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1
        success_rate = float("%.3f" % (float(success_count) / epoch_size))
        absolute_success_rate = float("%.3f" % (float(absolute_success_count) / epoch_size))
        average_reward = float("%.3f" % (float(total_reward) / epoch_size))
        average_turn = float("%.3f" % (float(total_turns) / epoch_size))
        average_wrong_disease = float("%.3f" % (float(inform_wrong_disease_count) / epoch_size))
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn,
               "average_wrong_disease":average_wrong_disease,"ab_success_rate":absolute_success_rate}
        # print("%3d simulation success rate %s, ave reward %s, ave turns %s, ave wrong disease %s" % (index,res['success_rate'], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
        self.dialogue_manager.state_tracker.agent.train_mode() # for training
        return res

    def evaluate_model(self, dataset, index, mode = 'eval'):
        """
        Evaluating model during training.
        :param index: int, the simulation index.
        :return: a dict of evaluation results including success rate, average reward, average number of wrong diseases.
        """
        if self.parameter.get("use_all_labels"):
            self.dialogue_manager.repeated_action_count = 0
            self.dialogue_manager.group_id_match = 0
        self.dialogue_manager.repeated_action_count = 0
        self.dialogue_manager.reach_max_turn = 0
        save_performance = self.parameter.get("save_performance")
        self.dialogue_manager.state_tracker.agent.eval_mode() # for testing
        success_count = 0
        absolute_success_count = 0
        total_reward = 0
        total_turns = 0
        #evaluate_session_number = len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        dataset_len=len(self.dialogue_manager.state_tracker.user.goal_set[dataset])
        evaluate_session_number=dataset_len
        evaluate_session_index = random.sample(range(dataset_len), evaluate_session_number)
        inform_wrong_disease_count = 0
        num_of_true_slots = 0
        num_of_implicit_slots = 0
        real_implicit_slots = 0
        avg_f1_total = 0
        avg_recall_total = 0
        record = {}
        
        new_dataset = dict()
        new_dataset['test'] = list()
        new_dataset['train'] = list()
        #for goal_index in range(0,evaluate_session_number, 1):
        for goal_index in evaluate_session_index:
            self.dialogue_manager.initialize(dataset=dataset, goal_index=goal_index)
            episode_over = False
            while episode_over == False:
                reward, episode_over, dialogue_status,slots_proportion_list = self.dialogue_manager.next(
                    save_record=False,greedy_strategy=False, index=index, mode = 'eval')
                total_reward += reward
            if self.parameter['save_record'] and not(self.parameter['train_mode']):
                goal = self.dialogue_manager.state_tracker.user.goal
                state_record = self.dialogue_manager.state_tracker.get_state()
                record[goal['consult_id']] = dict()
                record[goal['consult_id']]['disease_tag'] = goal['disease_tag']
                try:
                    record[goal['consult_id']]['disease_pre'] = state_record['agent_action']['inform_slots']['disease']
                except:
                    record[goal['consult_id']]['disease_pre'] = None    
                record[goal['consult_id']]['goal'] = goal['goal']
                record[goal['consult_id']]['history'] = state_record['current_slots']
                record[goal['consult_id']]['process'] = dict()
                for turn in state_record['history']:
                    if turn['inform_slots']:
                        record[goal['consult_id']]['process'].update(turn['inform_slots'])
                        
            assert len(slots_proportion_list)>0
            num_of_true_slots+=slots_proportion_list[0]
            num_of_implicit_slots+=slots_proportion_list[1]
            real_implicit_slots += slots_proportion_list[2]
            avg_f1_total += slots_proportion_list[0] * 2 / (slots_proportion_list[1] + slots_proportion_list[2] + 1e-10)
            avg_recall_total += slots_proportion_list[0] / (slots_proportion_list[2] + 1e-10)
            #(slots_proportion_list)
            total_turns += self.dialogue_manager.state_tracker.turn/2
            inform_wrong_disease_count += self.dialogue_manager.inform_wrong_disease_count
            if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_SUCCESS:
                success_count += 1
                if self.dialogue_manager.inform_wrong_disease_count == 0:
                    absolute_success_count += 1

        '''
        name = self.parameter['saved_model'].split('/')[-2].split('_')[0] + '_record.p'
        pickle.dump(file=open(os.path.join('HRL/dialogue_system/records/', name), "wb"), obj=self.dialogue_manager.record)
        ''' 
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
        reach_max_turn = float("%.4f" % (float(self.dialogue_manager.reach_max_turn) / evaluate_session_number))
        self.dialogue_manager.state_tracker.agent.train_mode() # for training.
        res = {
            "success_rate":success_rate,
            "average_reward": average_reward,
            "average_turn": average_turn,
            "average_repeated_action": average_repeated_action,
            "all_recall": all_recall,
            "all_f1": all_f1,
            "avg_recall":avg_recall,
            "avg_f1": avg_f1,
            'reach_max_turn': reach_max_turn
        }
        record['0-res'] = res
        if self.parameter['save_record'] and not(self.parameter['train_mode']):
            name = self.parameter['saved_model'].split('/')[-2].split('_')[0]
            with open('HRL/dialogue_system/records/' + name + '.json', 'w', encoding = 'utf-8') as f:
                json.dump(record, f, sort_keys=True, indent=2, ensure_ascii=False)
            self.parameter['save_record'] = False
        self.learning_curve.setdefault(index, dict())
        self.learning_curve[index]["success_rate"]=success_rate
        self.learning_curve[index]["average_reward"]=average_reward
        self.learning_curve[index]["average_turn"] = average_turn
        #self.learning_curve[index]["average_wrong_disease"]=average_wrong_disease
        self.learning_curve[index]["average_match_rate"]=match_rate
        self.learning_curve[index]["all_recall"] = all_recall
        self.learning_curve[index]["average_repeated_action"] = average_repeated_action
        '''
        if save_performance:
            self.wandb.log({'success_rate': success_rate, 'average_turn': average_turn, 'average_reward': average_reward, 'all_recall': all_recall, 'all_f1': all_f1, 'average_repeated_action': average_repeated_action})
        '''
        if index % 10 ==9:
            print('[INFO]', self.parameter["run_info"])
        if self.parameter.get("classifier_type")=="deep_learning" and self.parameter.get("disease_as_action") == False and self.parameter.get("train_mode"):
            self.dialogue_manager.train_deep_learning_classifier(index, epochs=20)

        if index % 1000 == 999 and save_performance == True:
            self.__dump_performance__(epoch_index=index)
        print("%3d simulation SR [%s], ave reward %s, ave turns %s, all_f1 %s, all_recall %s, ave repeated %s, avg recall %s, avg f1 %s, reach maxturn %s" % (index,res['success_rate'],res['average_reward'], res['average_turn'], res["all_f1"],res["all_recall"],res["average_repeated_action"],res['avg_recall'], res['avg_f1'], res['reach_max_turn']))

        if self.parameter.get("use_all_labels") == True and self.parameter.get("disease_as_action") == False:
            #self.dialogue_manager.train_deep_learning_classifier(epochs=100)

            if self.parameter.get("agent_id").lower() == "agenthrljoint":
                temp_by_group = {}
                for key,value in self.dialogue_manager.acc_by_group.items():
                    temp_by_group[key] = [0.0, 0.0]
                    if value[1] > 0:
                        temp_by_group[key][0] = float("%.3f" % (value[0]/value[1]))
                        temp_by_group[key][1] = float("%.3f" % (value[1]/value[2]))
                if index % 10 == 9:
                    #self.dialogue_manager.train_deep_learning_classifier(epochs=20)
                    print(self.dialogue_manager.acc_by_group)
                    print(temp_by_group)
                self.dialogue_manager.acc_by_group = {x: [0, 0, 0] for x in self.dialogue_manager.state_tracker.agent.master_action_space}

        return res

    def warm_start(self, epoch_number):
        """
        Warm-starting the dialogue, using the sample from rule-based agent to fill the experience replay pool for DQN.
        :param agent: the agent used to warm start dialogue system.
        :param epoch_number: the number of epoch when warm starting, and the number of dialogue sessions of each epoch
                             equals to the simulation epoch.
        :return: nothing to return.
        """
        for index in range(0,epoch_number,1):
            res = self.simulation_epoch(epoch_size=self.epoch_size, index=index)
            print("%3d simulation SR %s, ABSR %s,ave reward %s, ave turns %s, ave wrong disease %s" % (
            index, res['success_rate'], res["ab_success_rate"], res['average_reward'], res['average_turn'], res["average_wrong_disease"]))
            # if len(self.dialogue_manager.experience_replay_pool)==self.parameter.get("experience_replay_pool_size"):
            #     break

    def __dump_performance__(self, epoch_index):
        """
        Saving the performance of model.

        Args:
            epoch_index: int, indicating the current epoch.
        """
        file_name = self.parameter["run_info"] + "_" + str(epoch_index) + ".p"
        performance_save_path = self.parameter["performance_save_path"]
        if os.path.isdir(performance_save_path) is False:
            os.mkdir(performance_save_path)
        dirs = os.listdir(performance_save_path)
        for dir in dirs:
            if self.parameter["run_info"] in dir:
                os.remove(os.path.join(performance_save_path, dir))
        pickle.dump(file=open(os.path.join(performance_save_path,file_name), "wb"), obj=self.learning_curve)
