import numpy as np
import copy
import sys, os
import random
import re
import pickle
import math
from collections import deque, Counter

sys.path.append(os.getcwd().replace("HRL/dialogue_system/agent", ""))
from OpenMedicalChatBox.HRL.agent.agent_dqn import AgentDQN as LowerAgent
from OpenMedicalChatBox.HRL.policy_learning.dqn_torch import DQN, DQN2
from OpenMedicalChatBox.HRL.agent.utils import state_to_representation_last, reduced_state_to_representation_last
from OpenMedicalChatBox.HRL import dialogue_configuration


class AgentHRL_joint2(object):
    def __init__(self,slot_set, disease_symptom, parameter):
        self.parameter = parameter
        self.slot_set = slot_set
        try:
            self.slot_set.pop("disease")
        except:
            pass
        self.disease_symptom = disease_symptom
        self.master_experience_replay_size = 10000
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)
        self.input_size_dqn_all = {}
        for i in range(self.parameter.get("groups")):
            self.input_size_dqn_all[i+1] = 81


        self.id2disease = {}
        self.id2lowerAgent = {}
        self.pretrained_lowerAgent = {}
        self.master_action_space = []
        temp_parameter = {}
        for key, value in self.input_size_dqn_all.items():
        #dirs = os.listdir(self.parameter.get("label_all_model_path"))
        #for model in dirs:
            #reg = re.compile(r"(?<=label)\d+")
            #match = reg.search(model)
            #label = match.group(0)
            # print(label)
            label = str(key)
            self.master_action_space.append(label)
            # assert len(label) == 1
            # label = label[0]
            label_all_path = self.parameter.get("file_all")
            label_new_path = os.path.join(label_all_path, 'label' + str(label))
            disease_symptom = pickle.load(open(os.path.join(label_new_path, 'disease_symptom.p'), 'rb'))
            slot_set = pickle.load(open(os.path.join(label_new_path, 'slot_set.p'), 'rb'))
            #action_set = pickle.load(open(os.path.join(label_new_path, 'action_set.p'), 'rb'))

            temp_parameter[label] = copy.deepcopy(parameter)

            path_list = parameter['load_model_path'].split('/')
            path_list.insert(-1, 'lower')
            path_list.insert(-1, str(label))
            temp_parameter[label]['load_model_path'] = '/'.join(path_list)
            temp_parameter[label]['gamma'] = temp_parameter[label]['gamma_worker']  # discount factor for the lower agent.

            temp_parameter[label]["input_size_dqn"] = self.input_size_dqn_all[int(label)]
            #temp_parameter[label]["input_size_dqn"] = (len(slot_set)-1) *3
            self.id2lowerAgent[label] = LowerAgent(slot_set=slot_set,
                                                   disease_symptom=disease_symptom, parameter=temp_parameter[label],
                                                   disease_as_action=False)
            # model_path = os.path.join(self.parameter.get("label_all_model_path"), label)
            #temp_parameter[label]["input_size_dqn"] = self.input_size_dqn_all[int(label)]
            '''
            temp_parameter[label]["input_size_dqn"] = (len(slot_set)) * 3
            #print(slot_set)
            self.pretrained_lowerAgent[label] = LowerAgent(action_set=action_set, slot_set=slot_set,
                                                   disease_symptom=disease_symptom, parameter=temp_parameter[label],
                                                   disease_as_action=True)
            # model_path = os.path.join(self.parameter.get("label_all_model_path"), label)

            self.pretrained_lowerAgent[label].dqn.restore_model(os.path.join(self.parameter.get("label_all_model_path"), model))
            self.pretrained_lowerAgent[label].dqn.current_net.eval()
            self.pretrained_lowerAgent[label].dqn.target_net.eval()
            '''

        # Master policy.
        if parameter.get("state_reduced"):
            input_size = (len(self.slot_set)) * 3    # the dictionary of slot_set contains a key of "disease" which need to be removed first.
        else:
            input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 300)
        self.output_size = len(self.id2lowerAgent)
        if self.parameter.get("disease_as_action") == False:
            self.output_size = len(self.id2lowerAgent) + 1  # the extra one size is the action of activating disease classifier
        #print("input_size",input_size)
        else:
            self.output_size = len(self.id2lowerAgent) + len(self.disease_symptom)
        self.master = DQN2(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=self.output_size,
                           parameter=parameter,
                           named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over'))
        self.parameter = parameter
        # self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.current_lower_agent_id = -1
        self.behave_prob = 1
        print("master:", self.master_action_space)
        self.count = 0
        self.subtask_terminal = True
        self.subtask_turn = 0
        self.subtask_max_turn = 5
        self.past_lower_agent_pool = {key: 0 for key in self.id2lowerAgent.keys()}

        if parameter.get("train_mode") is False:
            print("########## master model is restore now ##########")
            self.master.restore_model(parameter.get("load_model_path"))
            self.master.current_net.eval()
            self.master.target_net.eval()
            for label, agent in self.id2lowerAgent.items():
                #print(temp_parameter[label])
                self.id2lowerAgent[label].dqn.restore_model(temp_parameter[label]['saved_model'])
                self.id2lowerAgent[label].dqn.current_net.eval()
                self.id2lowerAgent[label].dqn.target_net.eval()

        self.agent_action = {
            "turn": 1,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn": None,
            "action": None,
            "request_slots": {},
            "inform_slots": {},
            "explicit_inform_slots": {},
            "implicit_inform_slots": {},
            "speaker": "agent"
        }
        self.subtask_terminal = True
        self.subtask_turn = 0
        self.master_reward = 0

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        # represent the master state into a vector first

        # print(state["turn"])
        if self.parameter.get("state_reduced"):
            try:
               self.slot_set.pop("disease")
            except:
               pass
            state_rep = reduced_state_to_representation_last(state=state,
                                                             slot_set=self.slot_set,
                                                             parameter=self.parameter)  # sequence representation.
        else:
            state_rep = state_to_representation_last(state=state,
                                                     slot_set=self.slot_set,
                                                     disease_symptom=self.disease_symptom,
                                                     max_turn=self.parameter["max_turn"])  # sequence representation.
        # print(len(state_rep))
        # Only when the subtask is terminal, master agent takes an action.
        if self.subtask_terminal == True:
            self.master_state = copy.deepcopy(state)
            #print(len(state_rep))
            self.__master_next(state_rep=state_rep, greedy_strategy=greedy_strategy)
            self.subtask_terminal = False
            self.subtask_turn = 0

        # The selected lower agent takes an agent.
        # symptom_dist = self.disease_to_symptom_dist[self.id2disease[self.current_lower_agent_id]]
        # 在state_to_representation_last的步骤中，可以自动将不属于slot set中的slot去除掉
        
        #if self.parameter.get("disease_as_action"):
        if False:
            self.current_lower_agent_id = self.master_action_space[self.master_action_index]
            agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, self.subtask_turn,
                                                                                                         greedy_strategy=greedy_strategy)
        else:
            if self.master_action_index > (len(self.id2lowerAgent) - 1) or turn >= self.parameter["max_turn"]-3:  # The disease classifier is activated.
                agent_action = {'action': 'inform', 'inform_slots': {"disease": 'UNK'}, 'request_slots': {},
                                "explicit_inform_slots": {}, "implicit_inform_slots": {}}
                agent_action["turn"] = turn
                agent_action["inform_slots"] = {"disease": None}
                agent_action["speaker"] = 'agent'
                agent_action["action_index"] = None
                lower_action_index = -1
                self.subtask_terminal = True
                #print("********************************************************************")
            else:

                #print("**",self.master_action_index)
                self.subtask_turn += 1
                self.current_lower_agent_id = self.master_action_space[self.master_action_index]
                # print(self.current_lower_agent_id)
                agent_action, lower_action_index = self.id2lowerAgent[str(self.current_lower_agent_id)].next(state, self.subtask_turn, greedy_strategy=greedy_strategy)
                #if agent_action['action'] == "return" or self.subtask_turn >= self.subtask_max_turn:
                if self.subtask_turn >= self.subtask_max_turn:
                    self.subtask_terminal = True
                    self.subtask_turn = 0
                    #if agent_action['action'] == 'return':
                    #    print(agent_action['action'])
                else:
                    assert len(list(agent_action["request_slots"].keys())) == 1
            # print(self.current_lower_agent_id, lower_action_index)
            #print(self.subtask_turn, lower_action_index, self.master_action_index)
        return agent_action, self.master_action_index, lower_action_index

    def __master_next(self, state_rep, greedy_strategy):
        # Master agent takes an action.
        epsilon = self.parameter.get("epsilon")
        #print(greedy_strategy)
        if greedy_strategy == True:
            greedy = random.random()
            if greedy < epsilon:
                self.master_action_index = random.randint(0, self.output_size - 1)
                #if self.master_action_index == 9:
                #print("********************************************************************")
                #print(self.master_action_index)
                # master_action_index = random.sample(list(self.id2lowerAgent.keys()),1)[0]
            else:
                self.master_action_index = self.master.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            self.master_action_index = self.master.predict(Xs=[state_rep])[1]

    def next_state_values_DDQN(self, next_state):
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=next_state,
                                                             slot_set=self.slot_set, parameter=self.parameter)  # sequence representation.
        else:
            state_rep = state_to_representation_last(state=next_state,
                                                     action_set=self.action_set,
                                                     slot_set=self.slot_set,
                                                     disease_symptom=self.disease_symptom,
                                                     max_turn=self.parameter["max_turn"])
        action_index = self.master.predict(Xs=[state_rep])[1]
        Ys = self.master.predict_target(Xs=[state_rep])
        next_action_value = Ys.detach().cpu().numpy()[0][action_index]

        return next_action_value

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sample used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.master.singleBatch(batch=batch, params=self.parameter)
        return loss

    def update_target_network(self):
        self.master.update_target_network()
        for key in self.id2lowerAgent.keys():
            self.id2lowerAgent[key].update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.master.save_model(model_performance=model_performance, episodes_index=episodes_index,
                               checkpoint_path=checkpoint_path)
        # Saving lower agent
        for key, lower_agent in self.id2lowerAgent.items():
            temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
            lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index,
                                       checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        index = self.count
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)

        priority_scale = self.parameter.get("priority_scale")
        for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
            batch = random.sample(self.experience_replay_pool, min(batch_size, len(self.experience_replay_pool)))
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (
            float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))
        if self.count % 10 == 9:
            #print(len(self.id2lowerAgent))
            for group_id, lower_agent in self.id2lowerAgent.items():
                # if len(lower_agent.experience_replay_pool) ==10000 or (len(lower_agent.experience_replay_pool)-self.past_lower_agent_pool[group_id])>100:
                if len(lower_agent.experience_replay_pool) > 150:
                    lower_agent.train_dqn(label=group_id)
                    self.past_lower_agent_pool[group_id] = len(lower_agent.experience_replay_pool)

        self.count += 1
        # Training of lower agents.
        # for disease_id, lower_agent in self.id2lowerAgent.items():
        #    lower_agent.train_dqn()

    def reward_shaping(self, state, next_state):
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        #slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict = copy.deepcopy(state["current_slots"]["explicit_inform_slots"])
        slot_dict.update(state["current_slots"]["implicit_inform_slots"])
        #slot_dict.update(state["current_slots"]["proposed_slots"])
        #slot_dict.update(state["current_slots"]["agent_request_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        #next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict = copy.deepcopy(next_state["current_slots"]["explicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, lower_reward, master_action_index):
        # samples of master agent.
        # print(state)
        #print(reward)

        shaping = self.reward_shaping(state, next_state)
        alpha = self.parameter.get("weight_for_reward_shaping")
        '''
        if reward == self.parameter.get("reward_for_repeated_action"):
            lower_reward = reward
            # reward = reward * 2
        else:
            lower_reward = max(0, shaping * alpha)
            # lower_reward = shaping * alpha
        '''
        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping


        # samples of lower agent.
        #print('#', lower_reward)
        if int(agent_action) >= 0:
            #q_value = self.pretrained_lowerAgent[self.master_action_space[self.master_action_index]].get_q_values(state)
            #print(q_value)
            #print('# ', lower_reward)
            self.id2lowerAgent[self.current_lower_agent_id].record_training_sample(state, agent_action, lower_reward,
                                                                                   next_state, episode_over)

        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state,
                                                             slot_set=self.slot_set, parameter=self.parameter)  # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set, parameter=self.parameter)
            master_state_rep = reduced_state_to_representation_last(state=self.master_state, slot_set=self.slot_set, parameter=self.parameter)
        else:
            state_rep = state_to_representation_last(state=state,
                                                     slot_set=self.slot_set,
                                                     disease_symptom=self.disease_symptom,
                                                     max_turn=self.parameter["max_turn"])
            next_state_rep = state_to_representation_last(state=next_state,
                                                          slot_set=self.slot_set,
                                                          disease_symptom=self.disease_symptom,
                                                          max_turn=self.parameter["max_turn"])
            master_state_rep = state_to_representation_last(state=self.master_state,
                                                          slot_set=self.slot_set,
                                                          disease_symptom=self.disease_symptom,
                                                          max_turn=self.parameter["max_turn"])
        # print("state", [idx for idx,x in enumerate(state_rep) if x==1], agent_action)
        # print("nexts", [idx for idx,x in enumerate(next_state_rep) if x==1], reward)
        self.master_reward += reward

        if self.subtask_terminal or int(agent_action) == -1 or episode_over==True:
            #if self.master_reward > -40:
            #    self.master_reward = max(-1, self.master_reward)
            if self.master_reward >-60 and self.master_reward <=0:
                self.master_reward = self.master_reward /4
            #print(self.master_state["current_slots"]["inform_slots"])
            #print(next_state["current_slots"]["inform_slots"])
            #print("***", self.master_reward)

            #print(state['turn'], next_state['turn'])
            if self.master_action_index > (len(self.id2lowerAgent) - 1):
                subtask_turn = 1
            else:
                if self.subtask_turn == 0:
                    subtask_turn = 5
                else:
                    subtask_turn = self.subtask_turn
            #print(subtask_turn)
            self.experience_replay_pool.append((master_state_rep, master_action_index, self.master_reward, next_state_rep, episode_over, subtask_turn))
            self.master_reward = 0

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.master_experience_replay_size)

    def train_mode(self):
        self.master.current_net.train()

    def eval_mode(self):
        self.master.current_net.eval()
        self.master.current_net.eval()
        self.master.target_net.eval()
        for label, agent in self.id2lowerAgent.items():
            self.id2lowerAgent[label].dqn.current_net.eval()
            self.id2lowerAgent[label].dqn.target_net.eval()




