# -*- coding:utf-8 -*-
import time
import argparse
import pickle
import sys, os
import random
import json
import torch
from .agent import Agent
os.chdir(os.path.dirname(sys.argv[0]))
class REFUEL:
    def __init__(self, dataset_path, model_save_path, model_load_path, cuda_idx, train_mode = True, epoch_number = 5000, batch_size = 64, max_turn = 10,\
        reward_shaping = 0.25,  reward_for_success = 20, reward_for_fail = -1,  reward_for_reach_max_turn = -1, \
            rebuild_factor = 10, entropy_factor = 0.007, discount_factor = 0.99, lr = 0.0001):
        parser = argparse.ArgumentParser()
        #file0='./data/dxy_dataset/dxy_dataset/100symptoms//'
        #file0 = './data/data/dataset/label/'
        #file0='./data/new_data/mz10/allsymptoms/'
        file0 = dataset_path
        parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
        parser.add_argument("--disease_set", dest="disease_set", type=str, default=file0+'/disease_set.p',help='path and filename of the disease set')

        parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
        parser.add_argument("--goal_set_test", dest="goal_set_test", type=str, default=file0+'/goal_test_set.p',help='path and filename of user goal')

        parser.add_argument("--train_mode", dest="train_mode", type=bool, default=train_mode, help="Runing this code in training mode? [True, False]")
        parser.add_argument("--load_old_model", dest="load", type=bool, default=False)
        parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=epoch_number, help="The number of simulate epoch.")
        parser.add_argument("--model_savepath", dest="model_savepath", type=str, default=model_save_path, help="The path for save model.")
        parser.add_argument("--model_loadpath", dest="model_loadpath", type=str, default=model_load_path, help="The path for save model.")
        parser.add_argument("--batch_size", dest="batch_size", type=int, default=batch_size, help="The batchsize.")
        parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, help="The maxturn.")


        parser.add_argument("--wrong_prediction_reward", dest="n", type=int, default=reward_for_fail)

        parser.add_argument("--reward_shaping", dest="phi", type=int, default=reward_shaping)

        parser.add_argument("--Correct_prediction_reward", dest="m", type=int, default=reward_for_success)
        parser.add_argument("--reward_for_reach_max_turn", dest="out", type=int, default=reward_for_reach_max_turn)

        parser.add_argument("--rebulid_factor", dest="beta", type=int, default=rebuild_factor)
        parser.add_argument("--entropy_factor", dest="yita", type=int, default=entropy_factor)
        parser.add_argument("--discount_factor", dest="gamma", type=int, default=discount_factor)

        parser.add_argument("--cuda_idx", dest="cuda_idx", type=int, default=cuda_idx)
        parser.add_argument('--lr', type=float, default=lr)
        parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
        parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
        args = parser.parse_args()
        self.parameter = vars(args)

    def run(self):
        parameter = self.parameter
        slot_set = pickle.load(file=open(parameter["slot_set"], "rb"))
        disease_set = pickle.load(file=open(parameter["disease_set"], "rb"))
        train_mode = parameter.get("train_mode")
        simulate_epoch_number = parameter.get("simulate_epoch_number")

        agent = Agent(slot_set, disease_set, parameter)


        if train_mode:
            best_success_rate_test = agent.train(simulate_epoch_number)
            print('SC = ', best_success_rate_test)
            
        else:
            #agent.load(parameter['model_savepath'] + '/newest/')
            agent.load(parameter['model_loadpath'] )
            success_rate_test, avg_turns_test, avg_object_test, avg_recall, avg_out = agent.simulation_epoch(mode = 'test', epoch = 0, simulate_epoch_number = 1)
            # self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out
            print(success_rate_test, avg_turns_test, avg_object_test, avg_recall, avg_out)

if __name__ == '__main__':
    Refuel_test = REFUEL(dataset_path = 'D:\Documents\DISC\OpenMedicalChatBox\Data\mz10\\', model_save_path = './simulate', model_load_path = './simulate', cuda_idx = 1, train_mode = True)
    Refuel_test.run()
    