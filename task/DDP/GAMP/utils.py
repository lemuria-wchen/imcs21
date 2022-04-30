import math
import numpy as np
import torch
import random
import copy
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

def state_representation_abs(state, new_state, length, num_slot):
    state_vector = torch.zeros((num_slot))
    for i in range(length):
        try:
            state_vector[state[i]] = 1
        except:
            pass
    try:
        state_vector[new_state] = 1
    except:
        pass
    return state_vector

def dataset_generate(dataset, slot_set, disease_set):
    current_slots_store = list()
    disease_store = list()
    for state in dataset:
        ######################
        # Current_slots rep.
        #####################
        current_slots = copy.deepcopy(state["goal"]["explicit_inform_slots"])
        current_slots.update(state["goal"]["implicit_inform_slots"])
        # Not one hot
        current_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in current_slots.keys():
            # different values for different slot values.
            if slot in slot_set:
                if current_slots[slot] == '1' or current_slots[slot] == True:
                    current_slots_rep[slot_set[slot]] = 1.0
                elif current_slots[slot] == '0' or current_slots[slot] == False:
                    current_slots_rep[slot_set[slot]] = -1.0
                elif current_slots[slot] == '2':
                    current_slots_rep[slot_set[slot]] = 0
        disease = disease_set[state['disease_tag']]
        current_slots_store.append(current_slots_rep)
        disease_store.append(disease)
    return current_slots_store, disease_store

def seq_generate(state, action_tensor, data_length, goal, num_slot):
    current_slots_rep = np.zeros((num_slot))
    before_slots_rep = np.zeros((num_slot))
    for i in range(data_length):
        slot = int(state[i])
        try: 
            if goal[slot] == '1' or goal[slot] == True:
                current_slots_rep[slot] = 1.0
                before_slots_rep[slot] = 1.0
            elif goal[slot] == '0' or goal[slot] == False:
                current_slots_rep[slot] = -1.0
                before_slots_rep[slot] = -1.0
            elif goal[slot] == '2':
                current_slots_rep[slot] = 0
                before_slots_rep[slot] = 0
        except:
            pass
    action = int(action_tensor)
    try:
        if goal[action] == '1' or goal[action] == True:
            current_slots_rep[action] = 1.0
        elif goal[action] == '0' or goal[action] == False:
            current_slots_rep[action] = -1.0
        elif goal[action] == '2':
            current_slots_rep[action] = 0
    except:
        pass
    return torch.Tensor(before_slots_rep), torch.Tensor(current_slots_rep)

def generate_new_state(state, action, data_length, goal, num_slot):
    new_state = list()
    fake_symptom = list()
    length = data_length.size(0)
    for i in range(length):
        new_seq = list(state[i].cpu())
        new_action = action[i]
        if new_action == num_slot:
            new_seq.insert(data_length[i], 0)
        else:
            new_seq.insert(data_length[i], new_action)
        new_state.append(new_seq)
        if int(new_action) not in goal[i].keys() and new_action != num_slot:
            fake_symptom.append(state_representation_abs(state[i], action[i], data_length[i], num_slot))
    return new_state, fake_symptom