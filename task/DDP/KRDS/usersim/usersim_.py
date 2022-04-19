import random


class UserSimulator:
    """ Parent class for all user sims to inherit from """

    def __init__(self, sym_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.sym_dict = sym_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']

    def initialize_episode(self):
        """ Initialize a new episode (dialog)"""

        print("initialize episode called, generating goal")
        self.goal = random.choice(self.start_set)
        #self.goal['request_slots']['ticket'] = 'UNK'
        episode_over, user_action = self._sample_action()
        assert (episode_over != 1), ' but we just started'
        return user_action

    def next(self, system_action):
        pass