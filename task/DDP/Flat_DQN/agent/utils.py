# coding: utf8

import numpy as np
import copy
import sys, os
import traceback
sys.path.append(os.getcwd().replace("Flat_DQN/agent",""))
from . import dialogue_configuration


def state_to_representation_history(state, slot_set, disease_symptom, max_turn):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses history utterances to get dialogue state representation, including every utterance
    of each turn, very inform/requested slot of each turn.
    :param state: Dialogue state that needs to be mapped.
    :return: Dialogue state representation with 2-rank, which is a sequence of all utterances representations.
    """
    # TODO (Qianlong): mapping state to representation using one-hot. Including state["history"] and
    # TODO (Qianlong): state["current_slots"] of each turn.
    # （0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN）考虑生成一个sequence，每一个元素包含（action_rep, request_slots_rep,inform_slots_rep, explicit_inform_slots_rep,
    # implicit_slots_rep, turn_rep, current_slots_rep )
    # （2）与定电影票相同，并不考虑state中的history，只用user_action, agent_action, current_slots, 数据库查询结果，turn来
    # 生成当前的state_rep.
    # 现在使用的方法是生成一个sequence，但是sequence需要进一步处理，如LSTM， 然后再提供给。

    ###########################################################################################
    # One-hot representation for the current state using state["history"].
    ############################################################################################
    history = state["history"]
    state_rep = []
    for index in range(0, len(history), 1):
        temp_action = history[index]
        # Action rep.
        action_rep = np.zeros(len(action_set.keys()))
        action_rep[action_set[temp_action["action"]]] = 1.0

        # Request slots rep.
        request_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["request_slots"].keys():
            request_rep[slot_set[slot]] = 1.0

        # Inform slots rep.
        inform_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["inform_slots"].keys():
            inform_slots_rep[slot_set[slot]] = 1.0

        # Explicit_inform_slots rep.
        explicit_inform_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["explicit_inform_slots"].keys():
            explicit_inform_slots_rep[slot_set[slot]] = 1.0

        # Implicit_inform_slots rep.
        implicit_inform_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["implicit_inform_slots"].keys():
            implicit_inform_slots_rep[slot_set[slot]] = 1.0

        # Turn rep.
        turn_rep = np.zeros(max_turn)
        turn_rep[temp_action["turn"] - 1] = 1.0

        # Current_slots rep.
        current_slots = copy.deepcopy(temp_action["current_slots"]["inform_slots"])
        current_slots.update(temp_action["current_slots"]["explicit_inform_slots"])
        current_slots.update(temp_action["current_slots"]["implicit_inform_slots"])
        current_slots.update(temp_action["current_slots"]["proposed_slots"])
        current_slots.update(temp_action["current_slots"]["agent_request_slots"])
        current_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in current_slots.keys():
            if current_slots[slot] == True:
                current_slots_rep[slot_set[slot]] = 1.0
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                current_slots_rep[slot_set[slot]] = -1.0
            elif current_slots[slot] == dialogue_configuration.I_DENY:
                current_slots_rep[slot_set[slot]] = 2

        state_rep.append(np.hstack((action_rep, request_rep, inform_slots_rep, explicit_inform_slots_rep,
                                    implicit_inform_slots_rep, turn_rep, current_slots_rep)).tolist())
    return state_rep


def state_to_representation_last(state, slot_set, disease_symptom, max_turn):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses informed/requested slots that user has informed and requested up to this turn .
    :param state: Dialogue state
    :return: Dialogue state representation with 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-rank, which is a vector representing dialogue state.
    """
    ######################
    # Current_slots rep.
    #####################
    current_slots = copy.deepcopy(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])


    # Not one hot
    current_slots_rep = np.zeros(len(slot_set.keys()))
    for slot in current_slots.keys():
        # different values for different slot values.
        if slot in slot_set:
            if current_slots[slot] == '1':
                current_slots_rep[slot_set[slot]] = 1.0
            elif current_slots[slot] == '0':
                current_slots_rep[slot_set[slot]] = -1.0
            elif current_slots[slot] == '2':
                current_slots_rep[slot_set[slot]] = 2.0
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                current_slots_rep[slot_set[slot]] = -2.0
    #############
    # Turn rep.
    #############
    turn_rep = np.zeros(max_turn)
    try:
        turn_rep[state["turn"]] = 1.0
    except:
        pass
    state_rep = np.hstack((current_slots_rep, turn_rep))
    return state_rep


def reduced_state_to_representation_last(state, slot_set, parameter):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses informed/requested slots that user has informed and requested up to this turn .
    :param state: Dialogue state
    :return: Dialogue state representation with 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-rank, which is a vector representing dialogue state.
    """
    ######################
    # Current_slots rep.
    #####################
    try:
        slot_set.pop("disease")
    except:
        pass

    current_slots = copy.deepcopy(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])

    # one hot
    if parameter.get('data_type') == 'simulated':
        current_slots_rep = np.zeros((len(slot_set.keys()),3))

        for slot in slot_set:
            if slot in current_slots.keys():
                if current_slots[slot] is True:
                    temp_slot = [1,0,0]
                elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    temp_slot = [0,1,0]
                else:
                    temp_slot = [0,0,1]
                #print(current_slots[slot], temp_slot)
            else:
                temp_slot = [0,0,0]
            current_slots_rep[slot_set[slot], :] = temp_slot
            #print( temp_slot)
    elif parameter.get("data_type") == 'real':
        current_slots_rep = np.zeros((len(slot_set.keys()), 3))

        for slot in slot_set:
            if slot in current_slots.keys():
                if current_slots[slot] == '1' or current_slots[slot] == True:
                    temp_slot = [1, 0, 0]
                elif current_slots[slot] == '2' or current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    temp_slot = [0, 1, 0]
                # else:
                #    temp_slot = [0,0,1]
                elif current_slots[slot] == '0' or current_slots[slot] == False:
                    temp_slot = [0, 0, 1]
                else:
                    temp_slot = [0, 0, 0]

                # print(current_slots[slot], temp_slot)
            else:
                temp_slot = [0, 0, 0]
            current_slots_rep[slot_set[slot], :] = temp_slot
    else:
        raise ValueError

    '''
    for slot in current_slots.keys():
        # different values for different slot values.
        #print(current_slots)
        if slot in slot_set:
            #temp_slot = [0,0,0]
            if current_slots[slot] is True:
                temp_slot = [1,0,0]
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                temp_slot = [0,1,0]
                #print('****************************************************************')
            else:
                temp_slot = [0,0,1]
                #print('****************************************************************')
                #print(slot, current_slots[slot])
            current_slots_rep[slot_set[slot],:] = temp_slot
    '''
    # one-hot vector for each symptom.
    # current_slots_rep = np.zeros((len(slot_set.keys()),4))
    # for slot in current_slots.keys():
    #     # different values for different slot values.
    #     if current_slots[slot] == True:
    #         current_slots_rep[slot_set[slot]][0] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == False:
    #         current_slots_rep[slot_set[slot]][0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == 'UNK':
    #         current_slots_rep[slot_set[slot]][2] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
    #         current_slots_rep[slot_set[slot]][3] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    # current_slots_rep = np.reshape(current_slots_rep, (len(slot_set.keys())*4))

    #############
    # Turn rep.
    #############
    #turn_rep = state["turn"]

    # state_rep = np.hstack((current_slots_rep, wrong_diseases_rep, user_action_rep, user_inform_slots_rep, user_request_slots_rep, agent_action_rep, agent_inform_slots_rep, agent_request_slots_rep, turn_rep))
    state_rep = current_slots_rep.reshape(1,len(slot_set.keys())*3)[0]
    return state_rep

