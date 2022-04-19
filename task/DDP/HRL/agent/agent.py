# -*- coding: utf-8 -*-
"""
Basic agent class that other complicated agent, e.g., rule-based agent, DQN-based agent.
"""

import numpy as np
import copy
import sys, os
from collections import deque
sys.path.append(os.getcwd().replace("HRL/agent",""))
from  OpenMedicalChatBox.HRL import dialogue_configuration
from OpenMedicalChatBox.HRL.agent.utils import state_to_representation_last, reduced_state_to_representation_last

class Agent(object):
    """
    Basic class of agent.
    """
    def __init__(self, slot_set, disease_symptom, parameter, disease_as_action=True):
        self.parameter = parameter
        symptom_set = set()
        for key, v in disease_symptom.items():
            # print(key, len(v['symptom'].keys()))
            try:
                symptom_set = symptom_set | set(list(v['Symptom'].keys()))
            except:
                symptom_set = symptom_set | set(list(v['symptom'].keys()))
        # exit(0)

        self.slot_set = slot_set
        # self.disease_symptom = disease_symptom
        self.experience_replay_pool = deque(maxlen=parameter.get("experience_replay_pool_size"))
        self.parameter = parameter
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        #disease_as_action = self.parameter.get("disease_as_action")
        self.action_space = self._build_action_space(disease_symptom,disease_as_action)
        self.disease_symptom = self.disease_symptom_clip(disease_symptom, 2.5, parameter)


        self.agent_action = {
            "turn":1,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def initialize(self):
        """
        Initializing an dialogue session.
        :return: nothing to return.
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn":None,
            "action":None,
            "request_slots":{},
            "inform_slots":{},
            "explicit_inform_slots":{},
            "implicit_inform_slots":{},
            "speaker":"agent"
        }

    def next(self, *args, **kwargs):
        """
        Taking action based on different methods, e.g., DQN-based AgentDQN, rule-based AgentRule.
        Detail codes will be implemented in different sub-class of this class.
        :param state: a vector, the representation of current dialogue state.
        :param turn: int, the time step of current dialogue session.
        :return: the agent action, a tuple consists of the selected agent action and action index.
        """
        raise NotImplementedError('The `next` function of agent has not been implemented.')

    def train(self, batch):
        """
        Training the agent.
        Detail codes will be implemented in different sub-class of this class.
        :param batch: the sample used to training.
        :return:
        """
        raise NotImplementedError('The `train` function of agent has not been implemented.')

    def _build_action_space(self, disease_symptom, disease_as_action):
        """
        Building the Action Space for the RL-based Agent.
        All diseases are treated as actions.
        :return: Action Space, a list of feasible actions.
        """
        feasible_actions = []
        # Adding the request actions. And the slots are extracted from the links between disease and symptom,
        # i.e., disease_symptom
        slot_set = []
        for disease, v in disease_symptom.items():
            try:
                slot_set = slot_set + list(v["Symptom"])
            except:
                slot_set = slot_set + list(v["symptom"])
        slot_set = list(set(slot_set))
        for slot in self.slot_set:
            if slot != "disease":
                feasible_actions.append({'action': 'request', 'inform_slots': {}, 'request_slots': {slot: dialogue_configuration.VALUE_UNKNOWN},"explicit_inform_slots":{}, "implicit_inform_slots":{}})

        # Diseases as actions: inform + disease.
        if self.parameter.get("agent_id").lower() == "agenthrljoint":
            if disease_as_action is True:
                for disease in sorted(disease_symptom.keys()):
                    feasible_actions.append({'action': 'inform', 'inform_slots': {"disease": disease}, 'request_slots': {}, "explicit_inform_slots": {}, "implicit_inform_slots": {}})
        elif self.parameter.get("agent_id").lower() == "agenthrljoint2":
            if disease_as_action is True:
                for disease in sorted(disease_symptom.keys()):
                    feasible_actions.append({'action': 'inform', 'inform_slots': {"disease":disease}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}})
            #else:
            #    feasible_actions.append({'action': "return", 'inform_slots': {}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}})
        else:
            #print("#########################")
            if disease_as_action is True:
                for disease in sorted(disease_symptom.keys()):
                    feasible_actions.append({'action': 'inform', 'inform_slots': {"disease":disease}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}})
            else:
                feasible_actions.append({'action': "inform", 'inform_slots': {"disease":None}, 'request_slots': {},"explicit_inform_slots":{}, "implicit_inform_slots":{}})
        return feasible_actions

    @staticmethod
    def disease_symptom_clip(disease_symptom, denominator, parameter):
        """
        Keep the top min(symptom_num, max_turn//denominator) for each disease, and the related symptoms are sorted
        descendent according to their frequencies.
        Args:
            disease_symptom: a dict, key is the names of diseases, and the corresponding value is a dict too which
                contains the index of this disease and the related symptoms.
            denominator: int, the number of symptoms for each diseases is  max_turn // denominator.
            parameter: the super-parameter.
        Returns:
             and dict, whose keys are the names of diseases, and the values are dicts too with two keys: {'index', symptom}
        """
        max_turn = parameter.get('max_turn')
        temp_disease_symptom = copy.deepcopy(disease_symptom)
        for key, value in disease_symptom.items():
            try:
               symptom_list = sorted(value['Symptom'].items(), key=lambda x: x[1], reverse=True)
            except:
                symptom_list = sorted(value['symptom'].items(), key=lambda x: x[1], reverse=True)
            symptom_list = [v[0] for v in symptom_list]
            symptom_list = symptom_list[0:min(len(symptom_list), int(max_turn / float(denominator)))]
            temp_disease_symptom[key]['Symptom'] = symptom_list
        return temp_disease_symptom

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over, **kwargs):
        agent_id = self.parameter.get("agent_id")
        if self.parameter.get("state_reduced"):
            state = reduced_state_to_representation_last(state=state, slot_set=self.slot_set, parameter = self.parameter) # sequence representation.
            next_state = reduced_state_to_representation_last(state=next_state, slot_set=self.slot_set, parameter = self.parameter)
        else:
            state = state_to_representation_last(state=state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
            next_state = state_to_representation_last(state=next_state, action_set=self.action_set, slot_set=self.slot_set, disease_symptom=self.disease_symptom, max_turn=self.parameter["max_turn"])
        self.experience_replay_pool.append((state, agent_action, reward, next_state, episode_over))

    def flush_pool(self):
        self.experience_replay_pool = deque(maxlen=self.parameter.get("experience_replay_pool_size"))

    def train_mode(self):
        """
        Set the agent as the train mode, i.e., the parameters will be updated and dropout will be activated.
        """
        raise NotImplementedError("The `train_mode` function of agent has not been implemented")

    def eval_mode(self):
        """
        Set the agent as the train mode, i.e., the parameters will be unchanged and dropout will be deactivated.
        """
        raise NotImplementedError("The `train_mode` function of agent has not been implemented")