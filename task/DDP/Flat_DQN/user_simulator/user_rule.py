# -*- coding: utf-8 -*-
"""
User simulator which is based on rules.
"""

import sys, os
sys.path.append(os.getcwd().replace("Flat_DQN/dialogue_system/user_simulator",""))
from OpenMedicalChatBox.Flat_DQN.user_simulator.user import User

class UserRule(User):
    def __init__(self, goal_set, disease_syptom, exam_set, parameter):
        super(UserRule,self).__init__(goal_set=goal_set,
                                      disease_symptom=disease_syptom,
                                      exam_set = exam_set,
                                      parameter=parameter)