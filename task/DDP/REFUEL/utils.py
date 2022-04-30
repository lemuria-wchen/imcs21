import math
import numpy as np
import torch
import random
from torch.distributions import Categorical

def reb_generate(origin_state, origin_goal):
    #origin_goal = origin_goal.unsqueeze(0)

    forward = (origin_goal*(origin_state+1e-12).log()).sum()


    #origin_state_back = 1 - origin_state + 1e-12
    #origin_state_back = origin_state_back.unsqueeze(0)
    back = ((1-origin_goal)*(1+1e-12-origin_state).log()).sum()

    return -(forward + back)

def env_generate(state):
    env_record = (state*(state+1e-12).log()).sum()
    return -env_record

def random_generate(Action, mode):
    if mode == 'train':
        j = Categorical(Action)
        action = j.sample()
        prob = -j.log_prob(action)
    else:
        j = torch.argmax(Action, axis = 1)
        action = j
        prob = torch.ones(len(action))
    return action, prob

