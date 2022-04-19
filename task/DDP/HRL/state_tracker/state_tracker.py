# -*- coding:utf-8 -*-
"""
State tracker of the dialogue system, which tracks the state of the dialogue during interaction.
"""

import sys, os
import copy
import json
sys.path.append(os.getcwd().replace("HRL/dialogue_system/state_tracker", ""))

from OpenMedicalChatBox.HRL import dialogue_configuration


class StateTracker(object):
    def __init__(self, user, agent,parameter):
        self.user = user
        self.agent = agent
        self._init()

    def get_state(self):
        return copy.deepcopy(self.state)
        # return self.state

    def state_updater(self, user_action=None, agent_action=None):
        assert (user_action is None or agent_action is None), "user action and agent action cannot be None at the same time."
        self.state["turn"] = self.turn
        if user_action is not None:
            self._state_update_with_user_action(user_action=user_action)
        elif agent_action is not None:
            self._state_update_with_agent_action(agent_action=agent_action)
        self.turn += 1

    def initialize(self):
        self._init()

    def _init(self):
        self.turn = 0
        self.state = {
            "agent_action":None,
            "user_action":None,
            "turn":self.turn,
            "current_slots":{
                "explicit_inform_slots":{},
                "implicit_inform_slots":{},
            },
            "history":[]
        }

    def set_agent(self, agent):
        self.agent = agent

    def _state_update_with_user_action(self, user_action):
        # Updating dialog state with user_action.
        self.state["user_action"] = user_action
        temp_action = copy.deepcopy(user_action)
        temp_action["current_slots"] = copy.deepcopy(self.state["current_slots"])# Save current_slots for every turn.
        self.state["history"].append(temp_action)
        for slot in user_action["inform_slots"].keys():
            if slot not in user_action["explicit_inform_slots"].keys():
                self.state["current_slots"]["implicit_inform_slots"][slot] = user_action["inform_slots"][slot]

        for slot in user_action["explicit_inform_slots"].keys():
            self.state["current_slots"]["explicit_inform_slots"][slot] = user_action["explicit_inform_slots"][slot]

        # Inform_slots.
        inform_slots = list(user_action["inform_slots"].keys())


        # TODO (Qianlong): explicit_inform_slots and implicit_inform_slots are handled differently.
        # Explicit_inform_slots.
        explicit_inform_slots = list(user_action["explicit_inform_slots"].keys())
        for slot in explicit_inform_slots:
            self.state["current_slots"]["explicit_inform_slots"][slot] = user_action["explicit_inform_slots"][slot]
        # Implicit_inform_slots.
        implicit_inform_slots = list(user_action["implicit_inform_slots"].keys())
        for slot in implicit_inform_slots:
            self.state["current_slots"]["implicit_inform_slots"][slot] = user_action["implicit_inform_slots"][slot]


    def _state_update_with_agent_action(self, agent_action):
        # Updating dialog state with agent_action.
        explicit_implicit_slot_value = copy.deepcopy(self.user.goal["goal"]["explicit_inform_slots"])
        explicit_implicit_slot_value.update(self.user.goal["goal"]["implicit_inform_slots"])

        self.state["agent_action"] = agent_action
        temp_action = copy.deepcopy(agent_action)
        temp_action["current_slots"] = copy.deepcopy(self.state["current_slots"])# save current_slots for every turn.
        self.state["history"].append(temp_action)