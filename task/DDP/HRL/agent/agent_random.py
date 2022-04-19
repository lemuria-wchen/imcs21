# -*- coding: utf-8 -*-
"""
An agent that randomly choose an action from action_set.
"""
import random

import sys, os
sys.path.append(os.getcwd().replace("HRL/dialogue_system/agent",""))

from OpenMedicalChatBox.HRL.agent.agent import Agent


class AgentRandom(Agent):
    def __init__(self, action_set, slot_set, disease_symptom, parameter):
        super(AgentRandom, self).__init__(action_set=action_set, slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
        self.max_turn = parameter["max_turn"]

    def next(self, state,turn,greedy_strategy,**kwargs):
        self.agent_action["turn"] = turn
        action_index = random.randint(0, len(self.action_space)-1)
        agent_action = self.action_space[action_index]
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"
        return agent_action, action_index

    def train_mode(self):
        pass

    def eval_mode(self):
        pass