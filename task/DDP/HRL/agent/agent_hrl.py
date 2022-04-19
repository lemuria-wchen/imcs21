# -*- coding: utf-8 -*-
"""
Agent for hierarchical reinforcement learning.
"""

import numpy as np
import copy
import sys, os
import random
from collections import deque
sys.path.append(os.getcwd().replace("HRL/agent",""))
from OpenMedicalChatBox.HRL.agent.agent_dqn import AgentDQN as LowerAgent
from OpenMedicalChatBox.HRL.policy_learning.dqn_torch import DQN
from OpenMedicalChatBox.HRL.agent.utils import state_to_representation_last


class AgentHRL(object):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        self.action_set = action_set
        self.slot_set = slot_set
        self.disease_symptom = disease_symptom

        # symptom distribution by diseases.
        temp_slot_set = copy.deepcopy(slot_set)
        temp_slot_set.pop('disease')
        self.disease_to_symptom_dist = {}
        total_count = np.zeros(len(temp_slot_set))
        for disease, v in self.disease_symptom.items():
            dist = np.zeros(len(temp_slot_set))
            for symptom, count in v['symptom'].items():
                dist[temp_slot_set[symptom]] = count
                total_count[temp_slot_set[symptom]] += count
            self.disease_to_symptom_dist[disease] = dist

        for disease in self.disease_to_symptom_dist.keys():
            self.disease_to_symptom_dist[disease] = self.disease_to_symptom_dist[disease] / total_count

        ##################################
        # Building lower agents. The state representation that the master agent and lower agents are the same, so the
        # slot set are same for these different agents.
        ###########################
        self.id2disease = {}
        self.id2lowerAgent = {}
        for disease, v in disease_symptom.items():
            self.id2disease[v["index"]] = disease
            temp_disease_symptom = {}
            temp_disease_symptom[disease] = {}
            temp_disease_symptom[disease]["index"] = 0
            temp_disease_symptom[disease]["symptom"] = v["symptom"]
            temp_slot_set = {}
            for symptom in v['symptom'].keys():
                temp_slot_set.setdefault(symptom, len(temp_slot_set))
            temp_parameter = copy.deepcopy(parameter)
            temp_parameter["saved_model"] = parameter["saved_model"].split('model_d4_agent')[0] + 'lower/' + str(v["index"]) + '/model_d4_agent' + parameter["saved_model"].split('model_d4_agent')[1]
            self.id2lowerAgent[v["index"]] = LowerAgent(action_set=action_set, slot_set=slot_set, disease_symptom=temp_disease_symptom, parameter=temp_parameter)

        # Master policy.
        input_size = parameter.get("input_size_dqn")
        hidden_size = parameter.get("hidden_size_dqn", 100)
        output_size = len(self.id2lowerAgent)
        self.dqn = DQN(input_size=input_size,
                       hidden_size=hidden_size,
                       output_size=output_size,
                       parameter=parameter,
                       named_tuple=('state', 'agent_action', 'reward', 'next_state', 'episode_over', 'behave_prob'))
        self.parameter = parameter
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.current_lower_agent_id = -1
        self.behave_prob = 1

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

    def next(self, state, turn, greedy_strategy, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        # disease_symptom are not used in state_rep.
        epsilon = self.parameter.get("epsilon")
        state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"]) # sequence representation.

        # Master agent takes an action.
        if greedy_strategy == True:
            greedy = random.random()
            if greedy < epsilon:
                action_index = random.randint(0, len(self.id2lowerAgent) - 1)
            else:
                action_index = self.dqn.predict(Xs=[state_rep])[1]
        # Evaluating mode.
        else:
            action_index = self.dqn.predict(Xs=[state_rep])[1]
        self.behave_prob = 1 - epsilon + epsilon / (len(self.id2lowerAgent) - 1)
        self.current_lower_agent_id = action_index

        # Lower agent takes an agent.
        symptom_dist = self.disease_to_symptom_dist[self.id2disease[self.current_lower_agent_id]]
        agent_action, action_index = self.id2lowerAgent[self.current_lower_agent_id].next(state, turn, greedy_strategy, symptom_dist=symptom_dist)
        return agent_action, action_index

    def train(self, batch):
        """
        Training the agent.
        Args:
            batch: the sample used to training.
        Return:
             dict with a key `loss` whose value it a float.
        """
        loss = self.dqn.singleBatch(batch=batch,params=self.parameter)
        return loss

    def update_target_network(self):
        self.dqn.update_target_network()
        for key in self.id2lowerAgent.keys():
            self.id2lowerAgent[key].update_target_network()

    def save_model(self, model_performance, episodes_index, checkpoint_path=None):
        # Saving master agent
        self.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=checkpoint_path)
        # Saving lower agent
        for key, lower_agent in self.id2lowerAgent.items():
            temp_checkpoint_path = os.path.join(checkpoint_path, 'lower/' + str(key))
            lower_agent.dqn.save_model(model_performance=model_performance, episodes_index=episodes_index, checkpoint_path=temp_checkpoint_path)

    def train_dqn(self):
        """
        Train dqn.
        :return:
        """
        # ('state', 'agent_action', 'reward', 'next_state', 'episode_over')
        # Training of master agent
        cur_bellman_err = 0.0
        batch_size = self.parameter.get("batch_size",16)
        for iter in range(int(len(self.experience_replay_pool) / (batch_size))):
            batch = random.sample(self.experience_replay_pool, batch_size)

            loss = self.train(batch=batch)
            cur_bellman_err += loss["loss"]
        print("[Master agent] cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) / (len(self.experience_replay_pool) + 1e-10), len(self.experience_replay_pool)))

        # Training of lower agents.
        for disease_id, lower_agent in self.id2lowerAgent.items():
            lower_agent.train_dqn()

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        # samples of lower agent
        self.id2lowerAgent[self.current_lower_agent_id].record_training_sample(state, agent_action, reward, next_state, episode_over, symptom_dist=self.disease_to_symptom_dist[self.id2disease[self.current_lower_agent_id]])

        # samples of master agent.
        state_rep = state_to_representation_last(state=state,
                                                 action_set=self.action_set,
                                                 slot_set=self.slot_set,
                                                 disease_symptom=self.disease_symptom,
                                                 max_turn=self.parameter["max_turn"])
        next_state_rep = state_to_representation_last(state=next_state,
                                                      action_set=self.action_set,
                                                      slot_set=self.slot_set,
                                                      disease_symptom=self.disease_symptom,
                                                      max_turn=self.parameter["max_turn"])
        master_reward = reward
        self.experience_replay_pool.append((state_rep, self.current_lower_agent_id, master_reward, next_state_rep, episode_over, self.behave_prob))

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))
        for key, lower_agent in self.id2lowerAgent.items():
            self.id2lowerAgent[key].flush_pool()
