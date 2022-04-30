"""
Created on May 17, 2016

@author: xiul, t-zalipt
"""

import json
import OpenMedicalChatBox.KRDS.dialog_config as dialog_config
from .state_tracker import StateTracker


class DialogManager:
    """ A dialog manager to mediate the interaction between an agent and a customer """

    def __init__(self, agent, user, act_set, slot_set, params):
        self.agent = agent
        self.user = user
        self.act_set = act_set
        self.slot_set = slot_set  # slot for agent
        self.state_tracker = StateTracker()
        self.user_action = None
        self.reward = 0
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET
        self.hit_rate = 0
        self.params = params
         
    def initialize_episode(self):
        """ Refresh state for new dialog """

        self.reward = 0
        self.episode_over = False
        self.state_tracker.initialize_episode()
        self.user_action, self.goal = self.user.initialize_episode()
        self.state_tracker.update(user_action=self.user_action)
        self.hit_rate = 0
        self.agent.initialize_episode()
        return self.goal['consult_id']

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and user (agent first) """

        #   CALL AGENT TO TAKE HER TURN
        self.state = self.state_tracker.get_state_for_agent()
        self.agent_action, self.repeat = self.agent.state_to_action(self.state)
        
        #   Register AGENT action with the state_tracker
        self.state_tracker.update(agent_action=self.agent_action)


        #   CALL USER TO TAKE HER TURN
        self.sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        self.user_action, self.episode_over, self.dialog_status, hit = self.user.next(self.sys_action)

        self.reward = self.reward_function(self.dialog_status, hit)

        #   Update state tracker with latest user action
        if self.episode_over != True:
            self.state_tracker.update(user_action=self.user_action)


        #  Inform agent of the outcome for this timestep (s_t, a_t, r, s_{t+1}, episode_over)
        if record_training_data:
            self.agent.register_experience_replay_tuple(self.state, self.agent_action, self.reward,
                                                        self.state_tracker.get_state_for_agent(), self.episode_over)
        self.hit_rate += hit

        request = None
        if list(self.user_action['inform_slots'].keys()):
            request = list(self.user_action['inform_slots'].keys())[0]

        return self.episode_over, self.reward, self.dialog_status, self.hit_rate, request

    def reward_function(self, dialog_status, hit_rate):
        """ Reward Function 1: a reward function based on the dialog_status """
        if self.repeat:
            reward = self.params['reward_for_repeated_action']
        elif dialog_status == dialog_config.FAILED_DIALOG:
            reward = self.params['reward_for_reach_max_turn']  # -22
        elif dialog_status == dialog_config.SUCCESS_DIALOG:
            reward =  self.params['reward_for_success']  # 44
        elif dialog_status == dialog_config.WRONG_DISEASE:
            reward =  self.params['reward_for_fail']
        elif dialog_status == dialog_config.NO_OUTCOME_YET:
            reward =  self.params['reward_for_not_come_yet']
        else:
            reward =  hit_rate*self.params['reward_for_inform_right_symptom']
        return reward


