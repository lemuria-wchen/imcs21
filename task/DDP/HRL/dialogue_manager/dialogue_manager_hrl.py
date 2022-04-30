# -*- coding:utf-8 -*-
"""
dialogue manager for hierarchical reinforcement learning policy
"""

import copy
import random
from collections import deque
import sys, os

sys.path.append(os.getcwd().replace("HRL/dialogue_manager",""))

from OpenMedicalChatBox.HRL.state_tracker import StateTracker as StateTracker
from OpenMedicalChatBox.HRL import dialogue_configuration
from OpenMedicalChatBox.HRL.disease_classifier import dl_classifier
import numpy as np
from sklearn import svm
import pickle

class DialogueManager_HRL(object):
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
        self.master_action_history = []
        self.lower_action_history = []
        self.group_id_match = 0
        self.repeated_action_count = 0
        self.reach_max_turn = 0
        self.record = dict()
        self.slot_set = pickle.load(open(self.parameter.get("slot_set"), 'rb'))
        self.disease_symptom = pickle.load(open(self.parameter.get("disease_symptom"),'rb'))
        try:
            self.slot_set.pop('disease')
        except:
            pass
        self.id2disease = {}
        self.disease2id = {}
        for disease, v in self.disease_symptom.items():
            self.id2disease[v['index']] = disease
            self.disease2id[disease] = v['index']
        self.disease_replay = {'train': deque(maxlen=10000), 'test': deque(maxlen=1000)}

        self.worker_right_inform_num = 0
        self.acc_by_group = {x:[0,0,0] for x in ['1', '4','5','6','7','12','13','14','19']}
        # 这里的三维向量分别表示inform的症状正确的个数、group匹配正确的个数、隶属于某个group的个数


        if self.parameter.get("train_mode")==False:
            self.test_by_group = {x:[0,0,0] for x in ['1', '2']}
            #这里的三维向量分别表示成功次数、group匹配正确的个数、隶属于某个group的个数
            self.disease_record = []
            self.lower_reward_by_group = {x: [] for x in ['1', '2']}
            # self.master_index_by_group = {x:[] for x in ['12', '13', '14', '19', '1', '4', '5', '6', '7']}
            self.master_index_by_group = []
            self.symptom_by_group = {x: [0,0] for x in  ['1', '2']}

    def next(self, greedy_strategy, save_record, index, mode = 'train'):
        """
        The next two turn of this dialogue session. The agent will take action first and then followed by user simulator.
        :param save_record: bool, save record?
        :param train_mode: bool, True: the purpose of simulation is to train the model, False: just for simulation and the
                           parameters of the model will not be updated.
        :return: immediate reward for taking this agent action.
        """
        # Agent takes action.
        lower_reward = 0
        state = self.state_tracker.get_state()
        group_id = self.state_tracker.user.goal["group_id"]
        if self.parameter.get("agent_id")=='agentdqn':
            self.master_action_space = self.state_tracker.agent.action_space
        else:
            self.master_action_space = self.state_tracker.agent.master_action_space
        #if self.state_tracker.agent.subtask_terminal:
        #    self.master_state = copy.deepcopy(state)
        if self.parameter.get("agent_id") == 'agentdqn':
            master_action_index = len(self.master_action_space)
            agent_action,  lower_action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                                            greedy_strategy=greedy_strategy,
                                                                                            index=index)
        else:
            agent_action, master_action_index, lower_action_index = self.state_tracker.agent.next(state=state, turn=self.state_tracker.turn,
                                                                                              greedy_strategy=greedy_strategy, index=index)
        if len(agent_action["request_slots"]) > 0:
            lower_action = list(agent_action["request_slots"])
            assert len(list(agent_action["request_slots"].keys())) == 1
            action_type = "symptom"
        elif len(agent_action["inform_slots"]) > 0:
            lower_action = list(agent_action["inform_slots"])
            assert len(list(agent_action["inform_slots"].keys()))==1
            action_type = "disease"
            #print("#########")
        else:
            lower_action = "return to master"
            assert agent_action["action"] == 'return'
            action_type = 'return'
            #print("***************************************************************************************")
        #print(state['turn'])

        if self.parameter.get("disease_as_action")==False:
            if self.parameter.get("train_mode") == True:
                condition = False
            else:
                condition = state['turn']==self.parameter.get("max_turn")+16
            if action_type == "disease" or condition:# or lower_action in self.lower_action_history:
                #once the action is repeated or the dialogue reach the max turn, then the classifier will output the predicted disease
                state_rep = self.current_state_representation(state)
                disease = self.state_tracker.user.goal["disease_tag"]
                Ys, pre_disease = self.model.predict([state_rep])
                '''
                if index>20:
                    print(Ys)
                    temp = self.exp_transform(Ys.detach().cpu().numpy()[0])
                    print(max(temp), temp)
                '''
                #print(self.id2disease[pre_disease[0]])
                if mode == 'train':
                    self.disease_replay['train'].append((state_rep, self.disease2id[disease]))
                else:
                    self.disease_replay['test'].append((state_rep, self.disease2id[disease]))
                lower_action_index = -1
                master_action_index = len(self.master_action_space)
                agent_action = {'action': 'inform', 'inform_slots': {"disease":self.id2disease[pre_disease[0]]}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}}
                #print(agent_action)
                
                self.record[self.state_tracker.user.goal['consult_id']] = list(state['current_slots']['implicit_inform_slots'])
                
                
                if self.parameter.get("disease_remove"):
                    agent_action = {'action': 'inform', 'inform_slots': {"disease":'UNK'}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}}
                if self.parameter.get("train_mode") == False:
                    self.disease_record.append([disease,self.id2disease[pre_disease[0]]])
        #if master_action_index==9:
        #    print(master_action_index)  这里也存在9
        self.state_tracker.state_updater(agent_action=agent_action)
        # print("turn:%2d, state for agent:\n" % (state["turn"]) , json.dumps(state))

        # User takes action.
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.next(agent_action=agent_action,turn=self.state_tracker.turn)
        self.state_tracker.state_updater(user_action=user_action)
        # print("turn:%2d, update after user :\n" % (state["turn"]), json.dumps(state))
        #print('status', dialogue_status)
        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
            self.reach_max_turn += 1

        # if self.state_tracker.turn == self.state_tracker.max_turn:
        #     episode_over = True

        if master_action_index < len(self.master_action_space):
            self.master_action_history.append(self.master_action_space[master_action_index])
        #print(self.state_tracker.get_state()["turn"])
        #print(master_action_index)

        if self.parameter.get('agent_id').lower()=='agenthrljoint2' and lower_action in self.lower_action_history:
            lower_reward = self.parameter.get("reward_for_repeated_action")

        if self.parameter.get("agent_id") == "agenthrljoint2" and agent_action["action"]!="inform":
            #print(master_action_index)
            reward, lower_reward = self.next_by_hrl_joint2(dialogue_status, lower_action, state, master_action_index, group_id, episode_over, reward)
            if dialogue_status ==  dialogue_configuration.CONSTRAINT_CHECK_FAILURE:
                reward = self.parameter.get("reward_for_repeated_action")
                #print("************************")
                self.repeated_action_count += 1
                #episode_over = True
            else:
                self.action_history.append(lower_action)



        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_INFORM_WRONG_DISEASE:
            self.inform_wrong_disease_count += 1

        if save_record is True and mode == 'train':
            #print(self.action_history)
            if self.state_tracker.get_state()["turn"]==2:
                self.record_training_sample(
                    state=state,
                    agent_action=lower_action_index,
                    next_state=self.state_tracker.get_state(),
                    reward=reward,
                    episode_over=episode_over,
                    lower_reward = lower_reward,
                    master_action_index = master_action_index
                    )

        # Output the dialogue.
        slots_proportion_list = []
        if episode_over == True:
            self.action_history = []
            self.lower_action_history = []
            
            current_slots = copy.deepcopy(state["current_slots"]["implicit_inform_slots"])
            #current_slots.update(state["current_slots"]["explicit_inform_slots"])
            
            num_of_true_slots = 0
            #real_implicit_slots = len(self.state_tracker.user.goal['goal']['implicit_inform_slots']) + len(self.state_tracker.user.goal['goal']['explicit_inform_slots'])
            real_implicit_slots = len(self.state_tracker.user.goal['goal']['implicit_inform_slots'])
            for values in current_slots.values():
                if values != "I don't know." :
                    num_of_true_slots += 1
            num_of_all_slots = len(current_slots)
            slots_proportion_list.append(num_of_true_slots)  #表示current slot问到的隐性slot的个数
            slots_proportion_list.append(num_of_all_slots)   #表示current slot中含有的所有隐性slot的个数，即request的次数
            slots_proportion_list.append(real_implicit_slots)    #表示当前的goal 当中所含有的true slot的个数
            try:
                last_master_action = self.master_action_history[-1]
            except:
                last_master_action = -1
            if self.save_dialogue == True :
                state = self.state_tracker.get_state()
                goal = self.state_tracker.user.get_goal()
                self.__output_dialogue(state=state, goal=goal, master_history=self.master_action_history)
            if last_master_action == group_id:
                self.group_id_match += 1
            self.master_action_history = []
            if self.parameter.get("train_mode") == False and self.parameter.get("agent_id").lower()=="agenthrlnew2":
                #self.test_by_group[group_id][2] += 1
                #if last_master_action == group_id:
                #    self.test_by_group[group_id][1] += 1
                #    if reward == self.parameter.get("reward_for_success"):
                #        self.test_by_group[group_id][0] += 1
                if action_type == "disease":
                    self.test_by_group[last_master_action][2] += 1
                    if last_master_action == group_id:
                        self.test_by_group[last_master_action][1] += 1
                        if reward == self.parameter.get("reward_for_success"):
                            self.test_by_group[last_master_action][0] += 1

        return reward, episode_over, dialogue_status, slots_proportion_list

    def next_by_hrl_joint2(self, dialogue_status, lower_action, state, master_action_index, group_id, episode_over, reward):
        '''
                    self.acc_by_group[group_id][2] += 1
                    if self.master_action_space[master_action_index] == group_id:
                        self.acc_by_group[group_id][1] += 1
                        if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state()) > 0:
                            self.acc_by_group[group_id][0] += 1
                    '''
        alpha = self.parameter.get("weight_for_reward_shaping")
        '''
        if dialogue_status == dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
            self.repeated_action_count += 1
        '''
        # if action_type == "return":
        #    lower_reward = alpha/2 * self.worker_right_inform_num
        # if action_type == "symptom" and self.state_tracker.agent.subtask_terminal:  #reach max turn
        #    lower_reward = -alpha
        '''
        if self.parameter.get("train_mode") == False:
            if lower_action not in self.lower_action_history:
                # self.lower_reward_by_group[self.master_action_space[master_action_index]].append(lower_reward)
                self.symptom_by_group[self.master_action_space[master_action_index]][1] += 1
                if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state()) > 0:
                    self.symptom_by_group[self.master_action_space[master_action_index]][0] += 1
        '''
        lower_reward = reward + alpha * self.lower_reward_function(state=state, next_state=self.state_tracker.get_state())
        if lower_action[0] in list(state['current_slots']['explicit_inform_slots'].keys()) + list(state['current_slots']['implicit_inform_slots'].keys()):  # repeated action
            lower_reward = -self.parameter.get("reward_for_inform_right_symptom")
            self.state_tracker.agent.subtask_terminal = True
            # episode_over = True
            # self.repeated_action_count += 1
            reward = self.parameter.get("reward_for_repeated_action")
        # elif self.state_tracker.agent.subtask_terminal is False or lower_reward>0:
        else:
            #lower_reward = max(0, lower_reward)
            if lower_reward > 0:
                self.state_tracker.agent.subtask_terminal = True
                self.worker_right_inform_num += 1
            self.lower_action_history.append(lower_action)
            # if self.parameter.get("train_mode")==False:
            #    self.lower_reward_by_group[self.master_action_space[master_action_index]].append(lower_reward)
        # else:
        #    lower_reward = -alpha
        # if self.lower_reward_function(state=state, next_state=self.state_tracker.get_state())>0:
        #   self.worker_right_inform_num += 1
        # if  dialogue_status == dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
        '''
        if self.parameter.get("train_mode") == False:
            self.lower_reward_by_group[self.master_action_space[master_action_index]].append(lower_reward)
        '''
        if episode_over == True:
            self.state_tracker.agent.subtask_terminal = True
            self.state_tracker.agent.subtask_turn = 0
            '''
            #reward = alpha/2 * self.worker_right_inform_num - 2
            if dialogue_status != dialogue_configuration.DIALOGUE_STATUS_REACH_MAX_TURN:
                if self.master_action_space[master_action_index] == group_id:
                    reward = 10
                else:
                    reward = -1
            self.worker_right_inform_num = 0
            #self.lower_action_history = []
            '''
        elif self.state_tracker.agent.subtask_terminal:
            # reward = alpha / 2 * self.worker_right_inform_num - 2
            '''
            if self.master_action_space[master_action_index] == group_id:
                reward = 10
            else:
                reward = -1
            '''
            self.worker_right_inform_num = 0
            if self.parameter.get("train_mode") == False:
                self.master_index_by_group.append([group_id, master_action_index])
            # self.lower_action_history = []
        return reward, lower_reward

    def initialize(self, dataset, goal_index=None):
        self.state_tracker.initialize()
        self.inform_wrong_disease_count = 0
        user_action = self.state_tracker.user.initialize(dataset=dataset, goal_index=goal_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
        # self.group_id_match = 0
        # print("#"*30 + "\n" + "user goal:\n", json.dumps(self.state_tracker.user.goal))
        # state = self.state_tracker.get_state()
        # print("turn:%2d, initialized state:\n" % (state["turn"]), json.dumps(state))

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        if self.parameter.get("agent_id").lower() in ["agenthrljoint",'agenthrljoint2']:
            lower_reward = kwargs.get("lower_reward")
            master_action_index = kwargs.get("master_action_index")
            self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over, lower_reward, master_action_index)
        else:
            self.state_tracker.agent.record_training_sample(state, agent_action, reward, next_state, episode_over)


    def set_agent(self,agent):
        self.state_tracker.set_agent(agent=agent)

    def train(self):
        self.state_tracker.agent.train_dqn()
        self.state_tracker.agent.update_target_network()

    def __output_dialogue(self,state, goal, master_history):
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
            if speaker == "agent":
                try:
                    master_action = master_history.pop(0)
                    file.write(speaker + ": master+ " + str(master_action) +' +'+ action + "; inform_slots:"
                               + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
                except:
                    file.write(speaker + ": master+ " + ' +' + action + "; inform_slots:" + str(inform_slots)
                               + "; request_slots:" + str(request_slots) + "\n")
            else:
                file.write(speaker + ": " + action + "; inform_slots:" + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
        file.write("\n\n")
        assert len(master_history) == 0
        file.close()

    def lower_reward_function(self, state, next_state):
        """
        The reward for lower agent
        :param state:
        :param next_state:
        :return:
        """
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["implicit_inform_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)
        #return max(0, gamma * len(next_slot_dict) - len(slot_dict))

    def current_state_representation(self, state):
        """
        The state representation for the input of disease classifier.
        :param state: the last dialogue state before fed into disease classifier.
        :return: a vector that has equal length with slot set.
        """
        assert 'disease' not in self.slot_set.keys()
        state_rep = [0]*len(self.slot_set)
        current_slots = copy.deepcopy(state['current_slots'])
        if self.parameter.get('data_type') == 'simulated':
            for slot, value in current_slots["implicit_inform_slots"].items():
                state_rep[self.slot_set[slot]] = 1

            for slot, value in current_slots["explicit_inform_slots"].items():
                '''
                if value == True:
                    state_rep[self.slot_set[slot]] = 1
                elif value == "I don't know.":
                    state_rep[self.slot_set[slot]] = -1
                else:
                    print(value)
                    raise ValueError("the slot value of inform slot is not among True and I don't know")
                '''
                # DXY
                state_rep[self.slot_set[slot]] = 1
            #print(state_rep)
        elif self.parameter.get('data_type') == 'real':
            for slot, value in current_slots["implicit_inform_slots"].items():
                if value == '1' or value == True:
                    state_rep[self.slot_set[slot]] = 1
                elif value == '0' or value == False:
                    state_rep[self.slot_set[slot]] = -1
            for slot, value in current_slots["explicit_inform_slots"].items():
                if value == '1' or value == True:
                    state_rep[self.slot_set[slot]] = 1
                elif value == '0'  or value == False:
                    state_rep[self.slot_set[slot]] = -1
        return state_rep

    def current_state_representation_both(self, state):
        assert 'disease' not in self.slot_set.keys()
        state_rep = np.zeros((len(self.slot_set.keys()), 3))
        current_slots = copy.deepcopy(state['current_slots'])
        #for slot, value in current_slots['inform_slots'].items():
        for slot in self.slot_set:
            if slot in current_slots['inform_slots']:
                if current_slots['inform_slots'][slot] == True:
                    state_rep[self.slot_set[slot],:] = [1,0,0]
                elif current_slots['inform_slots'][slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    state_rep[self.slot_set[slot],:] = [0,1,0]
                else:
                    state_rep[self.slot_set[slot],:] = [0,0,1]
            else:
                state_rep[self.slot_set[slot],:] = [0,0,1]
                #raise ValueError("the slot value of inform slot is not among True and I don't know")
        print(state_rep)
        state_rep1 = state_rep.reshape(1, len(self.slot_set.keys()) * 3)[0]
        return state_rep1

    def train_ml_classifier(self):
        goal_set = pickle.load(open(self.parameter.get("goal_set"),'rb'))
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

    def train_deep_learning_classifier(self, index, epochs):
        best_test_acc = 0
        best_model = self.model.model
        for iter in range(epochs):
            try:
                batch = random.sample(self.disease_replay['train'], min(self.parameter.get("batch_size"),len(self.disease_replay['train'])))
                loss = self.model.train(batch=batch)
            except:
                pass
            try:
                test_batch = random.sample(self.disease_replay['test'], min(500,len(self.disease_replay['test'])))
                test_acc = self.model.test(test_batch=test_batch)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_model.load_state_dict(self.model.model.state_dict())
            except:
                pass
        self.model.model.load_state_dict(best_model.state_dict())
        print('disease_replay:{},loss:{:.4f}, test_acc:{:.4f}'.format(len(self.disease_replay['train']), loss["loss"], best_test_acc))

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
