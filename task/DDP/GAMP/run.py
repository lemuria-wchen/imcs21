# -*- coding:utf-8 -*-
import time
import argparse
import pickle
import sys, os
import random
import json
import torch
from OpenMedicalChatBox.GAMP.agent import Agent
from OpenMedicalChatBox.GAMP.running_steward import RunningSteward

class GAMP:
    def __init__(self, dataset_path, model_save_path, model_load_path, cuda_idx = 0, epoch_number = 1000, train_mode = True, max_turn = 10 ,batch_size = 64, lr = 0.0001):
        parser = argparse.ArgumentParser()
        file0 = dataset_path
        #file0='./Data/new_data/mz10/'
        #file0='./Data/dxy_dataset/dxy_dataset/'
        parser.add_argument("--slot_set", dest="slot_set", type=str, default=file0+'/slot_set.p',help='path and filename of the slots set')
        parser.add_argument("--disease_set", dest="disease_set", type=str, default=file0+'/disease_set.p',help='path and filename of the disease set')

        parser.add_argument("--goal_set", dest="goal_set", type=str, default=file0+'/goal_set.p',help='path and filename of user goal')
        parser.add_argument("--goal_set_test", dest="goal_set_test", type=str, default=file0+'/goal_test_set.p',help='path and filename of user goal')
        parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default=file0+"/disease_symptom.p",help="path and filename of the disease_symptom file")

        parser.add_argument("--train_mode", dest="train_mode", type=bool, default=train_mode, help="Running this code in training mode? [True, False]")
        parser.add_argument("--load_old_model", dest="load", type=bool, default=False)
        parser.add_argument("--simulate_epoch_number", dest="simulate_epoch_number", type=int, default=epoch_number, help="The number of simulate epoch.")
        parser.add_argument("--model_savepath", dest="model_savepath", type=str, default=model_save_path, help="The path for save model.")
        parser.add_argument("--load_path", dest="load_path", type=str, default=model_load_path, help="The path for load model.")
        parser.add_argument("--batch_size", dest="batch_size", type=int, default=batch_size, help="The batchsize.")
        parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, help="The maxturn.")

        parser.add_argument("--cuda_idx", dest="cuda_idx", type=int, default=cuda_idx)
        parser.add_argument('--lr', type=float, default=lr)
        parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
        parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
        args = parser.parse_args()
        self.parameter = vars(args)

    def run(self):
        slot_set = pickle.load(file=open(self.parameter["slot_set"], "rb"))
        disease_set = pickle.load(file=open(self.parameter["disease_set"], "rb"))
        train_mode = self.parameter.get("train_mode")
        simulate_epoch_number = self.parameter.get("simulate_epoch_number")
        goal_set = pickle.load(file=open(self.parameter["goal_set"], "rb"))
        agent = Agent(slot_set, disease_set, goal_set, self.parameter)


        if train_mode:
            agent.warm_start()
            best_success_rate_test = agent.train(simulate_epoch_number)
            print('SC = ', best_success_rate_test)
            
        else:
            agent.load(self.parameter['load_path'])
            #agent.load(parameter['model_savepath'] )
            success_rate_test, avg_turns_test, avg_object_test, hits, outs = agent.simulation_epoch(mode = 'test', epoch = 0, simulate_epoch_number = 1)
            print(success_rate_test, avg_turns_test, avg_object_test, hits, outs)

if __name__ == '__main__':

    GAMP_test = GAMP(dataset_path = 'D:\Documents\DISC\OpenMedicalChatBox\Data\mz10\\', model_save_path = './simulate', model_load_path = './simulate', cuda_idx = 1, train_mode = True)
    GAMP_test.run()