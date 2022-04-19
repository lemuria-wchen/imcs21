# -*- coding:utf-8 -*-

import copy
import random
from collections import deque
import sys, os
sys.path.append(os.getcwd().replace("Flat_DQN/dialogue_system/dialogue_manager",""))

from OpenMedicalChatBox.Flat_DQN.state_tracker import StateTracker as StateTracker
from OpenMedicalChatBox.Flat_DQN import dialogue_configuration
import numpy as np
from sklearn import svm
import pickle
from OpenMedicalChatBox.Flat_DQN.disease_classifier import dl_classifier

class DialogueManager(object):
    """
    Dialogue manager of this dialogue system.
    """
    def __init__(self, user, agent, parameter):
        self.state_tracker = StateTracker(user=user, agent=agent, parameter=parameter)
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        self.inform_wrong_disease_count = 0
        self.save_dialogue = parameter.get("save_dialogue")
        self.action_history = []
        self.repeated_action_count = 0
        self.disease_replay = deque(maxlen=10000)
        self.disease_symptom = pickle.load(open(self.parameter.get("disease_symptom"), 'rb'))
        self.slot_set = pickle.load(open(self.parameter.get("slot_set"), 'rb'))
        #self.action_set = pickle.load(file=open(parameter["action_set"], "rb"))
        self.exam_set = pickle.load(file=open(parameter["test_set"], "rb"))
        #self.disease_test = pickle.load(file=open(parameter["disease_test"], "rb"))
        #self.slot_set.pop('disease')
        self.id2disease = {}
        self.disease2id = {}
        for disease, v in self.disease_symptom.items():
            self.id2disease[v['index']] = disease
            self.disease2id[disease] = v['index']
        #print('++++++++++++++++',self.parameter.get("max_turn")-4)

    def next(self, greedy_strategy, save_record, index, mode = 'train'):
        """
        The next two turn of this dialogue session. The agent will take action first and then followed by user simulator.
        :param save_record: bool, save record?
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: immediate reward for taking this agent action.
        """
        # Agent takes action.
        state = self.state_tracker.get_state()
        #print(state['turn'])
        agent_action, action_index = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn,
                                                                   greedy_strategy=greedy_strategy,
                                                                   disease_tag=self.state_tracker.user.goal["disease_tag"])
        if self.parameter.get("disease_as_action")==False:
            if len(agent_action['inform_slots'])>0 or state["turn"] == (self.parameter.get("max_turn")-6) or action_index in self.action_history:
                state_rep = self.current_state_representation(state)
                Ys, pre_disease = self.model.predict([state_rep])
                disease = self.state_tracker.user.goal["disease_tag"]
                self.disease_replay.append((state_rep, self.disease2id[disease]))
                #print(state_rep)
                #print(pre_disease)
                agent_action = {'action': 'inform', 'inform_slots': {"disease":self.id2disease[pre_disease[0]]}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}}

        # agent_action, action_index = self.state_tracker.agent.next(state=state,turn=self.state_tracker.turn,greedy_strategy=greedy_strategy, train_mode=train_mode)
        self.state_tracker.state_updater(agent_action=agent_action)
        # print("turn:%2d, state for agent:\n" % (state["turn"]) , json.dumps(state))

        # User takes action.
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        # print("turn:%2d, update after user :\n" % (state["turn"]), json.dumps(state))

        if action_index in self.action_history:
            reward += self.parameter.get("reward_for_repeated_action")
            self.repeated_action_count += 1
            #episode_over = True
        else:
            self.action_history.append(action_index)

        # if self.state_tracker.turn == self.state_tracker.max_turn:
        #     episode_over = True

        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            self.inform_wrong_disease_count += 1

        if save_record is True and mode == 'train':
            self.record_training_sample(
                state=state,
                agent_action=action_index,
                next_state=self.state_tracker.get_state(),
                reward=reward,
                episode_over=episode_over
                )

        # Output the dialogue.

        slots_proportion_list = []
        if episode_over == True:
            self.action_history = []
            current_slots = copy.deepcopy(state["current_slots"]["implicit_inform_slots"])
            num_of_true_slots = 0
            real_implicit_slots = len(self.state_tracker.user.goal['goal']['implicit_inform_slots'])
            for values in current_slots.values():
                if values != "I don't know." :
                    num_of_true_slots += 1
            num_of_all_slots = len(current_slots)
            slots_proportion_list.append(num_of_true_slots)
            slots_proportion_list.append(num_of_all_slots)
            slots_proportion_list.append(real_implicit_slots)
            if self.save_dialogue == True :
                state = self.state_tracker.get_state()
                goal = self.state_tracker.user.get_goal()
                self.__output_dialogue(state=state, goal=goal)

        #print('a:', action_index)
        #print("r:", reward)
        #print('###',self.repeated_action_count)
        #print(self.action_history)

        return reward, episode_over, dialogue_status, slots_proportion_list

    def initialize(self, dataset, goal_index=None):
        self.state_tracker.initialize()
        self.inform_wrong_disease_count = 0
        user_action = self.state_tracker.user.initialize(dataset=dataset, goal_index=goal_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        # print("#"*30 + "\n" + "user goal:\n", json.dumps(self.state_tracker.user.goal))
        # state = self.state_tracker.get_state()
        # print("turn:%2d, initialized state:\n" % (state["turn"]), json.dumps(state))

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over)


    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    def train(self):
        self.state_tracker.agent.train_dqn()
        self.state_tracker.agent.update_target_network()

    def __output_dialogue(self,state, goal):
        history = state["history"]
        file = open(file=self.dialogue_output_file,mode="a+",encoding="utf-8")
        file.write("User goal: " + str(goal)+"\n")
        for turn in history:
            #print(turn)
            try:
                speaker = turn["speaker"]
            except:
                speaker = 'agent'
            action = turn["action"]
            inform_slots = turn["inform_slots"]
            request_slots = turn["request_slots"]
            file.write(speaker + ": " + action + "; inform_slots:" + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
        file.write("\n\n")
        file.close()

    def current_state_representation(self, state):
        state_rep = [0]*len(self.slot_set)
        current_slots = copy.deepcopy(state['current_slots'])
        for slot, value in current_slots['inform_slots'].items():
            if value == True:
                state_rep[self.slot_set[slot]] = 1
            #elif value == "I don't know.":
            #    state_rep[self.slot_set[slot]] = -1
            #else:
            #    print(value)
            #    raise ValueError("the slot value of inform slot is not among True and I don't know")
        return state_rep

    def train_ml_classifier(self):
        goal_set = pickle.load(open(self.parameter.get("goal_set"),'rb'))
        self.slot_set = pickle.load(open(self.parameter.get("slot_set"),'rb'))
        #disease_symptom = pickle.load(open(self.parameter.get("disease_symptom"),'rb'))
        self.slot_set.pop('disease')
        disease_y = []
        total_set = random.sample(goal_set['train'], 5000)

        slots_exp = np.zeros((len(total_set), len(self.slot_set)))
        for i, dialogue in enumerate(total_set):
            tag = dialogue['disease_tag']
            # tag_group=disease_symptom1[tag]['symptom']
            disease_y.append(tag)
            goal = dialogue['goal']
            explicit = goal['explicit_inform_slots']
            for exp_slot, value in explicit.items():
                try:
                    slot_id = self.slot_set[exp_slot]
                    if value == True:
                        slots_exp[i, slot_id] = '1'
                except:
                    pass

        self.model = svm.SVC(kernel='linear', C=1)
        self.model.fit(slots_exp, disease_y)

    def build_deep_learning_classifier(self):
        self.model = dl_classifier(input_size=len(self.slot_set), hidden_size=256,
                                   output_size=len(self.disease_symptom),
                                   parameter=self.parameter)
        if self.parameter.get("train_mode") == False:
            temp_path = self.parameter.get("saved_model")
            path_list = temp_path.split('/')
            path_list.insert(-1, 'classifier')
            saved_model = '/'.join(path_list)
            self.model.restore_model(saved_model)
            self.model.eval_mode()

    def train_deep_learning_classifier(self, epochs):
        #self.model.train_dl_classifier(epochs=5000)
        #print("############   the deep learning model is training over  ###########")
        for iter in range(epochs):
            batch = random.sample(self.disease_replay, min(self.parameter.get("batch_size"),len(self.disease_replay)))
            loss = self.model.train(batch=batch)

        test_batch = random.sample(self.disease_replay, min(1000,len(self.disease_replay)))
        test_acc = self.model.test(test_batch=test_batch)
        print('disease_replay:{},loss:{:.4f}, test_acc:{:.4f}'.format(len(self.disease_replay), loss["loss"], test_acc))
        #self.model.test_dl_classifier()

    def save_dl_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        temp_checkpoint_path = os.path.join(checkpoint_path, 'classifier/')
        self.model.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def exp_transform(self, x):
        exp_sum = 0
        for i in x:
            exp_sum += np.exp(i)
        return [np.exp(i)/exp_sum for i in x]