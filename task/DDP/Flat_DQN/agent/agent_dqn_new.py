# -*-coding:utf-8 -*
"""
The agent will maintain two ranked list of candidate disease and symptoms, the two list will be updated every turn based
on the information agent collected. The two ranked list will affect each other according <disease-symptom> pairs.
Agent will choose the first symptom with request as the agent action aiming to ask if the user has the symptom. The rank
model will change if the user's answer is no in continual several times.
"""

import random
import copy
import pickle
import math
import numpy as np
import sys, os
sys.path.append(os.getcwd().replace("Flat_DQN/dialogue_system/agent",""))
from Flat_DQN.dialogue_system.agent.agent import Agent
from Flat_DQN.dialogue_system.policy_learning.dqn_torch import DQN
from Flat_DQN.dialogue_system.agent.utils import state_to_representation_last, reduced_state_to_representation_last
from Flat_DQN.dialogue_system import dialogue_configuration


class AgentDQN(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter, disease_as_action=True):
        super(AgentDQN, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,
                                       parameter=parameter,disease_as_action=disease_as_action)

        # 是否将疾病的症状分布作为额外的输入。
        if self.parameter.get("agent_id") == 'agentdqn':
            disease_as_action = self.parameter.get("disease_as_action")
        self.agent_id = parameter.get("agent_id")
        self.action_space = self._build_action_space(disease_symptom, disease_as_action)
        #print(len(self.action_space))
        #self.slot_set.pop("disease") #这里的slot_set是带有disease这个关键字的



        if parameter.get("state_reduced") and self.parameter.get("agent_id").lower() in ['agenthrljoint','agenthrljoint2']:
            temp_slot_set = copy.deepcopy(slot_set)
            temp_slot_set.pop('disease')
            input_size = len(temp_slot_set) * 3
        elif parameter.get("state_reduced") and parameter.get("use_all_labels")==False:
            temp_slot_set = copy.deepcopy(slot_set)
            temp_slot_set.pop('disease')
            input_size = len(temp_slot_set) * 3
        else:
            #if len(kwargs) == 0:
            input_size = parameter.get("input_size_dqn")
        #print(input_size)
        hidden_size = parameter.get("hidden_size_dqn")
        output_size = len(self.action_space)
        self.dqn = DQN(input_size=input_size, hidden_size=hidden_size,output_size=output_size, parameter=parameter)
        self.action_visitation_count = {}

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        self.agent_action["turn"] = turn
        #state['turn'] = turn
        #print(state['turn'])
        #print(self.slot_set)
        symptom_dist = kwargs.get('symptom_dist')
        if self.parameter.get("state_reduced"):
            if self.parameter.get("agent_id").lower() in ["agenthrljoint","agenthrljoint2"] or self.parameter.get("use_all_labels")==False:
                state_rep = reduced_state_to_representation_last(state=state,
                                                                 slot_set=self.slot_set, parameter=self.parameter)  # sequence representation.
            else:
                state_rep = state_to_representation_last(state=state,
                                                         action_set=self.action_set,
                                                         slot_set=self.slot_set,
                                                         disease_symptom=self.disease_symptom,
                                                         max_turn=self.parameter["max_turn"])
        else:
            state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])


        if greedy_strategy is True:
            greedy = random.random()
            if greedy < self.parameter.get("epsilon"):
                action_index = random.randint(0, len(self.action_space) - 1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]

        agent_action = copy.deepcopy(self.action_space[action_index])
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"
        agent_action["action_index"] = action_index
        return agent_action, action_index

    def next2(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action when the action space is changing and select the action which is not inform disease.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        self.agent_action["turn"] = turn
        symptom_dist = kwargs.get('symptom_dist')
        if self.parameter.get("state_reduced") and self.parameter.get("use_all_labels")==False:
            state_rep = reduced_state_to_representation_last(state=state,
                                                  slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
        else:
            state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])

        # Flat_DQN with goal (not joint training one.)
        if "disease" in self.slot_set.keys():
            slot_num = len(self.slot_set)-1
        else:
            slot_num = len(self.slot_set)

        goal = kwargs.get('goal')
        if self.agent_id.lower() in ['agentwithgoal', 'agentwithgoal2' ]:
            state_rep = np.concatenate((state_rep, goal),axis=0)

        if greedy_strategy is True:
            greedy = random.random()
            if greedy < self.parameter.get("epsilon"):
                action_index = random.randint(0, len(self.action_space) - 1)
            else:
                action_index = self.dqn.predict_slot(Xs=[state_rep], slot_num=slot_num)[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict_slot(Xs=[state_rep], slot_num=slot_num)[1]

        agent_action = self.action_space[action_index] #当前动作空间最后10个的动作是inform disease，前面的都是request slot
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"
        agent_action["action_index"] = action_index
        assert len(list(agent_action["request_slots"].keys())) == 1
        return agent_action, action_index

    def next_state_values_DDQN(self, next_state):
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=next_state,
                                                 slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
        else:
            state_rep = state_to_representation_last(state=next_state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
        action_index = self.dqn.predict(Xs=[state_rep])[1]
        Ys = self.dqn.predict_target(Xs=[state_rep])
        next_action_value = Ys.detach().cpu().numpy()[0][action_index]
        return next_action_value

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sample used to training.
        Return:
             dict with a key `loss` whose value is a float.
        """
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        self.dqn.save_model(model_performance=model_performance, episodes_index = episodes_index, checkpoint_path=checkpoint_path)

    def train_dqn(self, **kwargs):
        """
        Train dqn.
        :return:
        """
        lower_rewards = []
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size", 16)
        priority_scale = self.parameter.get("priority_scale")
        for iter in range(math.ceil(len(self.experience_replay_pool) / batch_size)):
            batch = random.sample(self.experience_replay_pool, min(batch_size, len(self.experience_replay_pool)))
            #print(batch)
            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
            print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))

    def get_q_values(self, state, **kwargs):
        if self.parameter.get("state_reduced"):
            #slot_num = len(self.slot_set)
            #self.slot_set['disease'] = slot_num
            state_rep = reduced_state_to_representation_last(state=state,
                                                 slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
        else:
            slot_num = len(self.slot_set)
            self.slot_set['disease'] = slot_num
            state_rep = state_to_representation_last(state=state, action_set=self.action_set,
                                                     slot_set=self.slot_set,
                                                     disease_symptom=self.disease_symptom,
                                                     max_turn=self.parameter["max_turn"])
        # Lower agent of Flat_DQN with goal (not the one with joint training).
        #goal = kwargs.get('goal')
        #if self.agent_id.lower() == 'agentwithgoal' or self.agent_id.lower=='agentwithgoal2':
        #    state_rep = np.concatenate((state_rep, goal), axis=0)
        #print(len(state_rep))
        Q_values, max_index = self.dqn.predict(Xs=[state_rep])
        return Q_values.cpu().detach().numpy()

    def reward_shaping(self, state, next_state):
        def delete_item_from_dict(item, value):
            new_item = {}
            for k, v in item.items():
                if v != value: new_item[k] = v
            return new_item

        # slot number in state.
        slot_dict = copy.deepcopy(state["current_slots"]["inform_slots"])
        slot_dict.update(state["current_slots"]["explicit_inform_slots"])
        slot_dict.update(state["current_slots"]["implicit_inform_slots"])
        slot_dict.update(state["current_slots"]["proposed_slots"])
        slot_dict.update(state["current_slots"]["agent_request_slots"])
        slot_dict = delete_item_from_dict(slot_dict, dialogue_configuration.I_DO_NOT_KNOW)

        next_slot_dict = copy.deepcopy(next_state["current_slots"]["inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["explicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["implicit_inform_slots"])
        next_slot_dict.update(next_state["current_slots"]["proposed_slots"])
        next_slot_dict.update(next_state["current_slots"]["agent_request_slots"])
        next_slot_dict = delete_item_from_dict(next_slot_dict, dialogue_configuration.I_DO_NOT_KNOW)
        gamma = self.parameter.get("gamma")
        return gamma * len(next_slot_dict) - len(slot_dict)

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        shaping = self.reward_shaping(state, next_state)
        if self.parameter.get("agent_id").lower() in ["agenthrljoint", "agenthrljoint2"]:
            alpha = 0.0
        else:
            alpha = self.parameter.get("weight_for_reward_shaping")
        # if True:
            # print('shaping', shaping)
        # Reward shaping only when non-terminal state.
        if episode_over is True:
            pass
        else:
            reward = reward + alpha * shaping
        if self.parameter.get("state_reduced"):
            state_rep = reduced_state_to_representation_last(state=state, slot_set=self.slot_set, parameter=self.parameter) # sequence representation.
            next_state_rep = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set, parameter=self.parameter)
        else:
            state_rep = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
            next_state_rep = state_to_representation_last(state=next_state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        self.experience_replay_pool.append((state_rep, agent_action, reward, next_state_rep, episode_over))
        self.action_visitation_count.setdefault(agent_action, 0)
        self.action_visitation_count[agent_action] += 1
        #print(reward)

    def train_mode(self):
        self.dqn.current_net.train()

    def eval_mode(self):
        self.dqn.current_net.eval()

