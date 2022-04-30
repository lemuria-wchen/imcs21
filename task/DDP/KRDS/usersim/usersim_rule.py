"""
Created on May 14, 2016

a rule-based user simulator

-- user_goals_first_turn_template.revised.v1.p: all goals
-- user_goals_first_turn_template.part.movie.v1.p: moviename in goal.inform_slots
-- user_goals_first_turn_template.part.nomovie.v1.p: no moviename in goal.inform_slots

@author: xiul, t-zalipt
"""

from .usersim_ import UserSimulator
import argparse, json, random, copy
import OpenMedicalChatBox.KRDS.dialog_config as dialog_config


class RuleSimulator(UserSimulator):
    """ A rule-based user simulator for testing dialog policy """

    def __init__(self, sym_dict=None, act_set=None, slot_set=None, start_set=None, params=None):
        """ Constructor shared by all user simulators """

        self.sym_dict = sym_dict # all symptoms
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set

        self.max_turn = params['max_turn']

        self.data_split = params['data_split']
        self.hit = 0
        self.repeat = 0
        # self.left_goal = start_set

    def initialize_episode(self):
        """ Initialize a new episode (dialog)
        state['history_slots']: keeps all the informed_slots
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0
        # self.state['hit_slots'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        # self.goal =  random.choice(self.start_set)
        self.goal = self._sample_goal(self.start_set)
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        """ Debug: build a fake goal mannually """
        # self.debug_falk_goal()

        # sample first action
        user_action = self.start_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action, self.goal

    def start_action(self):
        self.state['diaact'] = "request"
        self.state['request_slots']['disease'] = 'UNK'
        #print(self.goal)
        if len(self.goal['explicit_inform_slots']) > 0:
            for slot in self.goal['explicit_inform_slots']:
                if self.goal['explicit_inform_slots'][slot] == True or self.goal['explicit_inform_slots'][slot] == '1':
                    self.state['inform_slots'][slot] = dialog_config.TRUE
                elif self.goal['explicit_inform_slots'][slot] == False or self.goal['explicit_inform_slots'][slot] == '0':
                    self.state['inform_slots'][slot] = dialog_config.FALSE
                else:
                    self.state['inform_slots'][slot] = dialog_config.NOT_SURE
        start_action = {}
        start_action['diaact'] = self.state['diaact']
        start_action['inform_slots'] = self.state['inform_slots']
        start_action['request_slots'] = self.state['request_slots']
        start_action['turn'] = self.state['turn']
        return start_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        sample_goal = random.choice(self.start_set[self.data_split])
        return sample_goal


    def next(self, system_action):
        """ Generate next User Action based on last System Action """
        self.hit = 0
        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.state['turn']+=2
        sys_act = system_action['diaact']
        
        if 0 < self.max_turn < self.state['turn']+3:
            self.dialog_status = dialog_config.FAILED_DIALOG
            self.episode_over = True
            self.state['diaact'] = "closing"
        else:
            self.state['history_slots'].update(self.state['inform_slots']) # add inform slot to history
            self.state['inform_slots'].clear()
            
            if sys_act == "inform":
                self.response_inform(system_action)
            #elif sys_act == "multiple_choice":
            #    self.response_multiple_choice(system_action)
            elif sys_act == "request":
                self.response_request(system_action)

            elif sys_act == "thanks":
                self.response_thanks(system_action)
            # elif sys_act == "confirm_answer":
            #     self.response_confirm_answer(system_action)
            #elif sys_act == "closing":
            #     self.episode_over = True
            #     self.state['diaact'] = "thanks"

        response_action = {}
        response_action['diaact'] = self.state['diaact']
        response_action['inform_slots'] = self.state['inform_slots']
        response_action['request_slots'] = self.state['request_slots']
        response_action['turn'] = self.state['turn']

        # add NL to dia_act
        # self.add_nl_to_action(response_action)

        # if len(self.goal['implicit_inform_slots'].keys()) == 0:
        #     hit_rate = 0.0
        # else:
        #     hit_rate = float(self.state['hit_slots'])/len(self.goal['implicit_inform_slots'].keys())
        # print(self.hit)

        return response_action, self.episode_over, self.dialog_status, self.hit



    def response_thanks(self, system_action):
        """ Response for Thanks (System Action) """

        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG
        # fail if no diagnosis or wrong diagnosis
        if self.state['request_slots']['disease'] == 'UNK' or self.state['request_slots']['disease'] != self.goal['disease_tag']:
            self.dialog_status = dialog_config.FAILED_DIALOG
        self.state['diaact'] = "closing"
        

    def response_request(self, system_action):
        """ Response for Request (System Action) """
        if len(system_action['request_slots'].keys()) > 0:
            slot = list(system_action['request_slots'].keys())[0]
            # answer slot in the goal
            if slot in self.goal['implicit_inform_slots'].keys():
                self.hit = 1
                # self.state['hit_slots'] += 1
                if self.goal['implicit_inform_slots'][slot] == True or self.goal['implicit_inform_slots'][slot] == '1':
                    self.state['diaact'] = "confirm"
                    self.state['inform_slots'][slot] = dialog_config.TRUE
                elif self.goal['implicit_inform_slots'][slot] == False or self.goal['implicit_inform_slots'][slot] == '0':
                    self.state['diaact'] = "deny"
                    self.state['inform_slots'][slot] = dialog_config.FALSE
                else:
                    self.state['diaact'] = "not_sure"
                    self.state['inform_slots'][slot] = dialog_config.NOT_SURE
            else:
                self.state['diaact'] = "not_sure"
                self.state['inform_slots'][slot] = dialog_config.NOT_SURE
                self.dialog_status = dialog_config.NO_OUTCOME_YET

    # response to diagnosis
    def response_inform(self, system_action):
        #self.state['diaact'] = "thanks"
        self.episode_over = True
        self.dialog_status = dialog_config.SUCCESS_DIALOG
        # fail if no diagnosis or wrong diagnosis
        self.state['request_slots']['disease'] = system_action['inform_slots']['disease']
        if self.state['request_slots']['disease'] == 'UNK' or self.state['request_slots']['disease'] != self.goal['disease_tag']:
            self.dialog_status = dialog_config.WRONG_DISEASE
            
        self.state['diaact'] = "thanks"



def main(params):
    user_sim = RuleSimulator()
    user_sim.initialize_episode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print("User Simulator Parameters:")
    print(json.dumps(params, indent=2))

    main(params)
