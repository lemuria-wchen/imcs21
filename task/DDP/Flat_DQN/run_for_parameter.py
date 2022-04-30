# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
import random
import json
import torch

sys.path.append(os.getcwd().replace("Flat_DQN/run", ""))

from Flat_DQN.agent import AgentRandom
from Flat_DQN.agent import AgentDQN
from Flat_DQN.agent import AgentRule
from Flat_DQN.run.utils import verify_params

from Flat_DQN.run import RunningSteward


def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        if s.lower() == 'true':
            return True
        else:
            return False


def run(parameter):
    """
    The entry function of this code.

    Args:
        parameter: the super-parameter

    """
    print(json.dumps(parameter, indent=2))
    time.sleep(2)
    slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
    action_set = pickle.load(file=open(parameter["action_set"], "rb"))
    disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))
    steward = RunningSteward(parameter=parameter, checkpoint_path=parameter["checkpoint_path"])

    print('action_set', action_set)
    warm_start = parameter.get("warm_start")
    warm_start_epoch_number = parameter.get("warm_start_epoch_number")
    train_mode = parameter.get("train_mode")
    agent_id = parameter.get("agent_id")
    simulate_epoch_number = parameter.get("simulate_epoch_number")

    # Warm start.
    if warm_start == True and train_mode == True:
        print("warm starting...")
        agent = AgentRule(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                          parameter=parameter)
        steward.dialogue_manager.set_agent(agent=agent)
        steward.warm_start(epoch_number=warm_start_epoch_number)
    # exit()
    if agent_id.lower() == 'agentdqn':
        agent = AgentDQN(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                         parameter=parameter)
    elif agent_id.lower() == 'agentrandom':
        agent = AgentRandom(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                            parameter=parameter)
    elif agent_id.lower() == 'agentrule':
        agent = AgentRule(action_set=action_set, slot_set=slot_set, disease_symptom=disease_symptom,
                          parameter=parameter)
    else:
        raise ValueError(
            'Agent id should be one of [AgentRule, AgentDQN, AgentRandom, AgentHRL, AgentWithGoal, AgentWithGoalJoint].')

    steward.dialogue_manager.set_agent(agent=agent)
    if train_mode is True:  # Train
        steward.simulate(epoch_number=simulate_epoch_number, train_mode=train_mode)
    else:  # test
        for index in range(simulate_epoch_number):
            steward.evaluate_model(dataset='test', index=index)

gpu = '2'
agent_id='AgentDQN'
wfrs_list = [0, 0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
gamma_list = [0.99]
epsilon_list = [0.2]

for wfrs in wfrs_list:
 for gamma in gamma_list:
  for epsilon in epsilon_list:
    disease_number = 4

    parser = argparse.ArgumentParser()
    parser.add_argument("--disease_number", dest="disease_number", type=int, default=disease_number,
                        help="the number of disease.")

    # simulation configuration
    parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=2000,
                        help="The number of simulate epoch.")
    parser.add_argument("--simulation_size", dest="simulation_size", type=int, default=100,
                        help="The number of simulated sessions in each simulated epoch.")
    parser.add_argument("--evaluate_session_number", dest="evaluate_session_number", type=int, default=1000,
                        help="the size of each simulate epoch when evaluation.")
    parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=10000,
                        help="the size of experience replay.")
    parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=100,
                        help="the hidden_size of DQN.")
    parser.add_argument("--warm_start", dest="warm_start", type=boolean_string, default=False,
                        help="Filling the replay buffer with the experiences of rule-based agents. {True, False}")
    parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int, default=30,
                        help="the number of epoch of warm starting.")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=30, help="the batch size when training.")
    parser.add_argument("--log_dir", dest="log_dir", type=str, default="./../../../log/",
                        help="directory where event file of training will be written, ending with /")
    parser.add_argument("--epsilon", dest="epsilon", type=float, default=epsilon, help="The greedy probability of DQN")
    parser.add_argument("--gamma", dest="gamma", type=float, default=gamma,
                        help="The discount factor of immediate reward in RL.")
    parser.add_argument("--train_mode", dest="train_mode", type=boolean_string, default=True,
                        help="Runing this code in training mode? [True, False]")

    #  Save model, performance and dialogue content ? And what is the path if yes?
    parser.add_argument("--save_performance", dest="save_performance", type=boolean_string, default=True,
                        help="save the performance? [True, False]")
    parser.add_argument("--save_model", dest="save_model", type=boolean_string, default=True,
                        help="Save model during training? [True, False]")
    parser.add_argument("--save_dialogue", dest="save_dialogue", type=boolean_string, default=False,
                        help="Save the dialogue? [True, False]")

    parser.add_argument("--run_id", dest='run_id', type=int, default=0, help='the id of this running.')

    # Learning rate for dqn.
    parser.add_argument("--dqn_learning_rate", dest="dqn_learning_rate", type=float, default=0.0001,
                        help="the learning rate of dqn.")

    # agent to use.
    # parser.add_argument("--agent_id", dest="agent_id", type=str, default='AgentWithGoalJoint', help="The agent to be used:[AgentRule, AgentDQN, AgentRandom, AgentHRL, AgentHRLGoal]")
    parser.add_argument("--agent_id", dest="agent_id", type=str, default=agent_id,
                        help="The agent to be used:[AgentRule, AgentDQN, AgentRandom, AgentHRL, AgentHRLGoal]")

    # goal set, slot set, action set.
    max_turn = 22
    parser.add_argument("--action_set", dest="action_set", type=str, default='./../../data/real_world/action_set.p',
                        help='path and filename of the action set')
    parser.add_argument("--slot_set", dest="slot_set", type=str, default='./../../data/real_world/slot_set.p',
                        help='path and filename of the slots set')
    parser.add_argument("--goal_set", dest="goal_set", type=str, default='./../../data/real_world/goal_set.p',
                        help='path and filename of user goal')
    parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,
                        default="./../../data/real_world/disease_symptom.p",
                        help="path and filename of the disease_symptom file")
    parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, help="the max turn in one episode.")
    parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=max_turn + 477,
                        help="the input_size of DQN.")
    # parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=2438, help="the input_size of DQN.")
    parser.add_argument("--reward_for_not_come_yet", dest="reward_for_not_come_yet", type=float, default=-1)
    parser.add_argument("--reward_for_success", dest="reward_for_success", type=float, default=2 * max_turn)
    parser.add_argument("--reward_for_fail", dest="reward_for_fail", type=float, default=-1 * max_turn)
    parser.add_argument("--reward_for_inform_right_symptom", dest="reward_for_inform_right_symptom", type=float,
                        default=-1)
    parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=boolean_string, default=False,
                        help="Success reward minus the number of left slots as the final reward for a successful session.{True, False}")

    parser.add_argument("--gpu", dest="gpu", type=str, default=gpu, help="The id of GPU on the running machine.")
    parser.add_argument("--check_related_symptoms", dest="check_related_symptoms", type=boolean_string, default=False,
                        help="Check the realted symptoms if the dialogue is success? True:Yes, False:No")
    parser.add_argument("--dqn_type", dest="dqn_type", default="DQN", type=str, help="[DQN, DoubleDQN")

    # noisy channel
    parser.add_argument("--noisy_channel", dest="noisy_channel", type=boolean_string, default=False,
                        help="noisy channel for user action?")
    parser.add_argument("--error_prob", dest="error_prob", type=float, default=0.05,
                        help="Error probability when applying noisy channel?")


    # reward shapping
    parser.add_argument("--weight_for_reward_shaping", dest='weight_for_reward_shaping', type=float, default=wfrs,
                        help="weight for reward shaping. 0 means no reward shaping.")

    args = parser.parse_args()
    parameter = vars(args)


    params = verify_params(parameter)
    gpu_str = params["gpu"]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str  # '0,0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN,2'
    torch.cuda.manual_seed(12345)
    torch.manual_seed(12345)
    params["run_info"] = params["run_info"] + '_params'
    print(params['run_info'])
    run(parameter=parameter)