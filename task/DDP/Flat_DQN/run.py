# -*- coding:utf-8 -*-

import time
import argparse
import pickle
import sys, os
import random
import json
import torch
#import wandb
sys.path.append(os.getcwd().replace("Flat_DQN/run",""))

from .agent import AgentRandom
from .agent import AgentDQN
from .agent import AgentRule
from .utils import verify_params
from OpenMedicalChatBox.Flat_DQN.running_steward import RunningSteward


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    if s.lower() == 'true':
        return True
    else:
        return False
class Flat_DQN:
    def __init__(self, dataset_path, model_save_path, model_load_path, lr = 0.0005, groups = 4, cuda_idx = 0, train_mode = True, max_turn = 10, \
        reward_for_not_come_yet = 0, reward_for_success = 20, reward_for_fail = 0, reward_for_inform_right_symptom = 6, reward_for_reach_max_turn = -100, \
            reward_for_repeated_action = -4, epoch_number = 5000, epoch_size = 100, experience_replay_size = 10000, batch_size = 100, discount_factor = 1, warm_start = False, warm_start_number = 30, greedy = 0.1):

        disease_number = 10
        parser = argparse.ArgumentParser()
        parser.add_argument("--disease_number", dest="disease_number", type=int,default=disease_number,help="the number of disease.")

        # simulation configuration
        parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=epoch_number, help="The number of simulate epoch.")
        parser.add_argument("--simulation_size", dest="simulation_size", type=int, default=epoch_size, help="The number of simulated sessions in each simulated epoch.")
        parser.add_argument("--evaluate_session_number", dest="evaluate_session_number", type=int, default=412, help="the size of each simulate epoch when evaluation.") # valid 371, test 412
        parser.add_argument("--experience_replay_pool_size", dest="experience_replay_pool_size", type=int, default=experience_replay_size, help="the size of experience replay.")
        parser.add_argument("--hidden_size_dqn", dest="hidden_size_dqn", type=int, default=512, help="the hidden_size of DQN.")
        parser.add_argument("--warm_start", dest="warm_start",type=boolean_string, default=warm_start, help="Filling the replay buffer with the experiences of rule-based agents. {True, False}")
        parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int, default=warm_start_number, help="the number of epoch of warm starting.")
        parser.add_argument("--batch_size", dest="batch_size", type=int, default=batch_size, help="the batch size when training.")
        parser.add_argument("--log_dir", dest="log_dir", type=str, default="./../../../log/", help="directory where event file of training will be written, ending with /")
        parser.add_argument("--epsilon", dest="epsilon", type=float, default=greedy, help="The greedy probability of DQN")
        parser.add_argument("--gamma", dest="gamma", type=float, default=discount_factor, help="The discount factor of immediate reward in RL.")
        parser.add_argument("--gamma_worker", dest="gamma_worker", type=float, default=0.9, help="The discount factor of immediate reward of the lower agent in Flat_DQN.")
        parser.add_argument("--train_mode", dest="train_mode", type=boolean_string, default=train_mode, help="Runing this code in training mode? [True, False]")
        parser.add_argument('--groups', dest='groups', type=str, default=groups, help='the data type is either simulated or real')
        parser.add_argument('--sequential_sampling', dest='sequential_sampling', type=boolean_string, default=True, help='True-seq sampling, False-random')

        #  Save model, performance and dialogue content ? And what is the path if yes?
        parser.add_argument("--save_performance",dest="save_performance", type=boolean_string, default=False, help="save the performance? [True, False]")
        parser.add_argument("--save_model", dest="save_model", type=boolean_string, default=True,help="Save model during training? [True, False]")
        parser.add_argument("--save_model_path", dest="save_model_path", type=str, default=model_save_path)
        parser.add_argument("--load_model_path", dest="load_model_path", type=str, default=model_load_path)
        parser.add_argument("--save_dialogue", dest="save_dialogue", type=boolean_string, default=False, help="Save the dialogue? [True, False]")
        parser.add_argument("--save_record", dest="save_record", type=boolean_string, default=False, help="Save the record? [True, False]")
        parser.add_argument("--disease_remove", dest="disease_remove", type=boolean_string, default=False, help="Whether to predict a disease.")


        parser.add_argument("--run_id", dest='run_id', type=int, default=0, help='the id of this running.')
        parser.add_argument("--save_experience_pool", dest="save_experience_pool",type=boolean_string, default=False,help="Save experience replay")

        # goal set, slot set, action set.
        max_turn = 2*max_turn
        file0 = dataset_path
        parser.add_argument("--data_files", dest="data_files", type=str, default='MZ-10',help='path and filename of the action set')
        parser.add_argument("--action_set", dest="action_set", type=str, default=file0+'/action_set.p',help='path and filename of the action set')
        parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
        parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
        parser.add_argument("--test_set", dest="test_set", type=str, default=file0+'/goal_test_set.p',help='path and filename of the test set')
        parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")
        parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, help="the max turn in one episode.")
        parser.add_argument("--input_size_dqn", dest="input_size_dqn", type=int, default=340, help="the input_size of DQN.") # for 4-label


        # reward design
        parser.add_argument("--reward_for_not_come_yet", dest="reward_for_not_come_yet", type=float,default=reward_for_not_come_yet)
        parser.add_argument("--reward_for_success", dest="reward_for_success", type=float,default=reward_for_success)
        parser.add_argument("--reward_for_fail", dest="reward_for_fail", type=float,default=reward_for_fail)
        parser.add_argument("--reward_for_inform_right_symptom", dest="reward_for_inform_right_symptom", type=float,default=reward_for_inform_right_symptom)
        parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=boolean_string, default=False,help="Success reward minus the number of left slots as the final reward for a successful session.{True, False}")
        parser.add_argument("--reward_for_reach_max_turn", dest="reward_for_reach_max_turn", type=float, default=reward_for_reach_max_turn)
        parser.add_argument("--reward_for_repeated_action", dest='reward_for_repeated_action', type=float, default= reward_for_repeated_action, help='the reward for repeated action')
        parser.add_argument("--weight_for_reward_shaping", dest='weight_for_reward_shaping', type=float, default=1, help="weight for reward shaping. 0 means no reward shaping.")

        # agent to use and DQN setting.
        parser.add_argument("--agent_id", dest="agent_id", type=str, default='agentdqn', help="The agent to be used:[AgentRule, AgentDQN, AgentRandom]")
        parser.add_argument("--gpu", dest="gpu", type=str, default=str(cuda_idx),help="The id of GPU on the running machine.")
        parser.add_argument("--check_related_symptoms", dest="check_related_symptoms", type=boolean_string, default=False, help="Check the realted symptoms if the dialogue is success? True:Yes, False:No")
        parser.add_argument("--dqn_type", dest="dqn_type", default="DQN", type=str, help="[DQN, DoubleDQN, DuelingDQN")
        parser.add_argument("--dqn_learning_rate", dest="dqn_learning_rate", type=float, default=lr, help="the learning rate of dqn.")

        parser.add_argument("--state_reduced", dest="state_reduced", type=boolean_string, default=False, help="whether to reduce the state dimension")

        # Flat_DQN configurations with hrl_new and hrl_joint
        parser.add_argument("--disease_as_action", dest="disease_as_action", type=boolean_string, default=True, help="if False then we use a classifier to inform disease")
        parser.add_argument("--classifier_type", dest="classifier_type", type=str, default="deep_learning", help="the classifier type is among machine_learning and deep_learning")
        parser.add_argument("--use_all_labels", dest="use_all_labels", type=boolean_string, default=False, help="whether to use more than one groups in Flat_DQN")
        parser.add_argument('--file_all', dest="file_all", type=str, default=file0, help='the path for groups of diseases')
        parser.add_argument("--label_all_model_path", dest="label_all_model_path", type=str, default='./../../data/best_models_reduced')

        args = parser.parse_args()
        self.parameter = vars(args)


    def run(self):
        """
        The entry function of this code.

        Args:
            parameter: the super-parameter

        """
        parameter = self.parameter
        params = verify_params(parameter)
        gpu_str = params["gpu"]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str#  '0,0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN,2'

        print(json.dumps(parameter, indent=2))
        time.sleep(2)
        slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        disease_symptom = pickle.load(file=open(parameter["disease_symptom"], "rb"))

        #print('action_set', action_set)
        warm_start = parameter.get("warm_start")
        warm_start_epoch_number = parameter.get("warm_start_epoch_number")
        train_mode = parameter.get("train_mode")

        simulate_epoch_number = parameter.get("simulate_epoch_number")
        steward = RunningSteward(parameter=parameter,checkpoint_path=parameter["checkpoint_path"])

        # Warm start.
        if warm_start == True and train_mode == True:
            print("warm starting...")
            agent = AgentRule(slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)
            steward.dialogue_manager.set_agent(agent=agent)
            steward.warm_start(epoch_number=warm_start_epoch_number)
        agent = AgentDQN(slot_set=slot_set,disease_symptom=disease_symptom,parameter=parameter)

        steward.dialogue_manager.set_agent(agent=agent)

        if train_mode is True: # Train
            steward.simulate(epoch_number=simulate_epoch_number, train_mode=train_mode)
        else: # test
            for index in range(simulate_epoch_number):
                steward.evaluate_model(dataset='test', index=index)


if __name__ == "__main__":
    HRL_test = Flat_DQN(dataset_path = '/remote-home/czhong/RL/OpenMedicalChatBox_install/Data/mz4/HRL//', model_save_path = './simulate', model_load_path = './simulate', cuda_idx = 1, train_mode = True)
    HRL_test.run()
