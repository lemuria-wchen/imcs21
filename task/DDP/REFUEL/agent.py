from .network import REFUEL_net, init_weights
import torch
import random
import pickle
import math
import copy
from torch import optim
from .dataloader import get_loader
import sys
import multiprocessing as mp
import math
import numpy as np
import OpenMedicalChatBox.REFUEL.utils as utils 
import os
import time

def progress_bar(bar_len, SR, avg_turn, avg_obj, avg_recall, avg_out, SRT, ATT, avg_recall_t,  best_SR, avg_out_t, currentNumber, wholeNumber):
    # 20, success_rate, avg_turns, avg_object, success_rate_test, avg_turns_test, best_success_rate_test, best_avg_turns_test, i, simulate_epoch_number
    """
    bar_len 进度条长度
    currentNumber 当前迭代数
    wholeNumber 总迭代数
    """
    filled_len = int(round(bar_len * currentNumber / float(wholeNumber)))
    percents = round(100.0 * currentNumber / float(wholeNumber), 1)
    bar = '\033[32;1m%s\033[0m' % '>' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(\
        '[%d/%d][%s] %s%s \033[31;1mSR\033[0m = %3f \033[36;1mAvg_turn\033[0m= %3f \033[36;1mAvg_obj\033[0m= %3f \033[36;1mAvg_recall\033[0m= %3f \033[36;1mAvg_out\033[0m= %3f \033[33;1mSR_t\033[0m= %3f \033[33;1mAT_t\033[0m= %3f \033[33;1mrecall_t\033[0m= %3f \033[33;1mout_t\033[0m= %3f \033[33;1mBest_SR\033[0m= %3f  \r' %\
         (int(currentNumber),int(wholeNumber), bar, '\033[32;1m%s\033[0m' % percents, '%', SR, avg_turn, avg_obj, avg_recall, avg_out, SRT, ATT, avg_recall_t, avg_out_t, best_SR))
    sys.stdout.flush()


class Agent(object):
    def __init__(self, slot_set, disease_set, parameter):
        self.num_slot = len(slot_set)
        self.num_disease = len(disease_set)
        self.slot_set = slot_set
        self.idx2sym = dict()
        for key, value in self.slot_set.items():
            self.idx2sym[value] = key
        self.disease_set = disease_set
        self.goal_set_path =  parameter['goal_set']
        self.goal_test_path =  parameter['goal_set_test']
        self.batch_size = parameter['batch_size']
        self.max_turn = parameter['max_turn']
        self.n = parameter['n']
        self.m = parameter['m']
        self.beta = parameter['beta']
        self.gamma = parameter['gamma']
        self.out = parameter['out']
        self.phi = parameter['phi']
        self.yita = parameter['yita']
        self.load_model = parameter['load']
        
        self.parameter = parameter
        self.num_cores = int(mp.cpu_count())
        
        self.run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
        self.path = self.parameter['model_savepath'] + self.run_time
        if self.parameter['train_mode'] and not(os.path.exists(self.path)):
            os.makedirs(self.path)
        self.build_model()
        #print("本地计算机有: " + str(self.num_cores) + " 核心")

    def build_model(self, mode = 'normal'):
        self.current_REFUEL = REFUEL_net(input_size= self.num_slot, output_size=self.num_slot + self.num_disease, rebulid_size=self.num_slot)
        self.target_REFUEL = REFUEL_net(input_size= self.num_slot, output_size=self.num_slot + self.num_disease, rebulid_size=self.num_slot)
        self.optimizer_para = list(self.current_REFUEL.parameters()) 
        self.optimizer = optim.Adam(self.optimizer_para, self.parameter['lr'])
        self.device = torch.device('cuda:' + str(self.parameter['cuda_idx']) if torch.cuda.is_available() else 'cpu')
        self.current_REFUEL.to(self.device)
        self.target_REFUEL.to(self.device)
        init_weights(self.current_REFUEL, mode)
        if self.load_model:
            self.load(self.parameter['model_loadpath'] )
            print('Load path....')
        

    def update_state(self, action, goal, goal_disease, state, turn):
        episode_over = torch.ones(len(action))
        new_state = copy.deepcopy(state)
        goal_symptom = goal
        reward = torch.zeros(len(action))
        acc_flag = torch.zeros(len(action))
        reach_max_turn = torch.zeros(len(action))
        
        for i in range(len(action)):
            if turn == self.max_turn and action[i] < self.num_slot:
                episode_over[i] = 0
                reward[i] = self.out
                reach_max_turn[i] = 1
                                    
                    #elif state[i][simulate[i]] != 0:
                    #    status[i] = 4
                    #    reward[turn][i] = self.n                  
            else: 
                if action[i] < self.num_slot:
                    if goal_symptom[i][action[i]]:
                        new_state[i][action[i]] = 1.0
                    else:
                        new_state[i][action[i]] = -1.0
                elif (int(action[i]) - self.num_slot) == goal_disease[i]:
                    reward[i] = self.m
                    episode_over[i] = 0
                    acc_flag[i] = 1
                else:
                    reward[i] = self.n
                    episode_over[i] = 0
                
        return new_state, reward, episode_over, acc_flag, reach_max_turn

    def simulate(self, origin_state, goal, goal_disease, mode):
        # goal : vector
        loss = 0
        total_turn = 0
        acc = 0
        reach_max = 0
        length = origin_state.size(0)
        #####change
        rebuild_record_list = list()
        action_prob_list = list()
        state_record = torch.zeros((self.max_turn+2, length, self.num_slot))
        reward_gamma_list = torch.zeros((self.max_turn+1, length)).to(self.device)
        loss_total = 0
        reward_record = list()
        prob_record = list()
        action_record = list()
        episode_over_list = list()
        final_state = torch.zeros(length, self.num_slot)

        acc_flag = torch.zeros(length)
        turn = 1
        state = origin_state
        episode_over_list.append(np.ones(length))
        reward_record.append(np.zeros(length))
        
        final_status = torch.zeros(length)

        ######## Symptom stage ########
        ##### 调整reward和state的下标，如论文
        for i in range(0, self.max_turn):
            
            # copy
            action_prob, rebuild_record = self.current_REFUEL(state)  
                
            
            action_prob_list.append(action_prob)
            rebuild_record_list.append(rebuild_record)

            action, prob = utils.random_generate(action_prob, mode)
            prob_record.append(prob)
            action_record.append(np.array(action.cpu()))
            new_state, reward, episode_over, acc_flag_turn, reach_max_turn = self.update_state(action, goal, goal_disease, state, turn)
            
            episode_over = episode_over * episode_over_list[i]
           
            episode_over_list.append(episode_over)
            acc_flag = acc_flag + acc_flag_turn
            reward_record.append(reward)
            state_record[i+1,:,:] = new_state 
            state = copy.deepcopy(new_state)
            turn += 1
        # 每个对话持续了多少轮
        episode_over_list = torch.tensor(episode_over_list, device=self.device)
        reward_record = torch.tensor(reward_record, device=self.device)
        rebuild_record_list = torch.stack(rebuild_record_list).to(self.device)
        prob_record = torch.stack(prob_record).to(self.device)
        action_prob_list =  torch.stack(action_prob_list).to(self.device)
        
        over_turn = torch.sum(episode_over_list, axis = 0).long()
        reach_max_turn = reach_max_turn.to(self.device)
        
        for j in range(length):
            final_status[j] = reward_record[over_turn[j], j]
            final_state[j] = state_record[over_turn[j], j]
        # 判断 max_turn 之前对话是否结束
        reach_max_turn = reach_max_turn * episode_over_list[i-1,:]
        acc += final_status[final_status > 0].sum()
        reach_max += reach_max_turn.sum()
        hit_match = torch.sum(final_state == 1) - torch.sum(origin_state != 0)
        all_match = torch.sum(abs(goal)) - torch.sum(abs(origin_state))
        total_turn += torch.sum(over_turn)
        if mode == 'train':
            for i in reversed(range(0, self.max_turn)):
                # at terminal reward shaping should be 0
                reward_shaping = (self.gamma * self.phi * torch.sum((state_record[i+1,:,:] == 1.0), axis = 1) - self.phi * torch.sum((state_record[i,:,:] == 1.0), axis = 1)).to(self.device)
                reward_shaping = reward_shaping *  episode_over_list[i+1,:]
                # reward_gamma_list ---- (self.max_turn+2, length)
                # 这里只要是涉及到回溯遍历的都要考虑到episode over的情况
                reward_gamma_list[i, :] = episode_over_list[i] * (reward_record[i+1, :]  + reward_shaping + self.gamma * reward_gamma_list[i+1, :])
                loss +=  self.beta * utils.reb_generate(rebuild_record_list[i, :], goal) + self.yita * utils.env_generate(action_prob_list[i, :])
                # 能矩阵直接相乘吗？使用episode over
            
            loss = torch.sum(reward_gamma_list[:self.max_turn, :] * (prob_record))
            
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                self.optimizer.step()
                self.current_REFUEL.zero_grad()
            
            loss_total = loss.item()

        return loss_total, hit_match, all_match, acc, reach_max, total_turn, reward_gamma_list[0], np.array(final_state)
    
    def train_network(self, eps = 1e-10):

        pass
    
    def simulation_epoch(self, mode, epoch, simulate_epoch_number):
        if mode == 'train':
            dataset = get_loader(self.slot_set,  self.disease_set, self.goal_set_path, batch_size=self.batch_size, mode = 'train')
        else:
            dataset = get_loader(self.slot_set,  self.disease_set, self.goal_test_path, batch_size=self.batch_size, mode = 'test')
        success_count = 0
        total_object = 0
        total_hit = 0
        total_all_imp = 0
        total_rewards = 0
        total_env = 0
        length = 0
        total_simulate = 0
        out_match = 0
        record = {}

        # 多核运算
        hit_num = 0
        #pool = mp.Pool(self.num_cores)
        for i, (origin_state, goal, goal_disease, consult_id) in enumerate(dataset):

            temp_object = 0

            origin_state = origin_state.to(self.device)
            
            goal = goal.to(self.device)
            goal_disease = goal_disease.to(self.device)
            
            loss, hit_match, all_match, acc_flag, reach_max_turn, turn, rewards, final_state  = self.simulate(origin_state, goal, goal_disease, mode)
            
            total_object += loss
            success_count += acc_flag
            out_match += reach_max_turn
            total_simulate += turn
            total_hit += hit_match
            total_all_imp += all_match
            length += origin_state.size(0)
            
            final_state = np.array(final_state)

            for j in range(len(consult_id)):
                goal_idx = np.where(goal[j].cpu() != 0)
                origin_idx = np.where(origin_state[j].cpu() != 0)
                symptom_idx = np.where(final_state[j] == 1)
                symptom_idx = np.setdiff1d(np.intersect1d(symptom_idx, goal_idx), origin_idx)
                symptom_record = list(set([self.idx2sym[k] for k in symptom_idx]))
                
                record[consult_id[j]] = symptom_record

                hit_num += len(symptom_record)
            if mode == 'train':
                progress_bar(10, self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out, self.success_rate_test, self.avg_turns_test, self.avg_recall_test, self.best_success_rate_test, self.avg_outs_test, i + epoch * len(dataset), simulate_epoch_number* len(dataset))
                #self.save(self.parameter['model_savepath'] + '/newest/')
        '''
        if mode == 'test':
            pickle.dump(file=open('./records/' + self.parameter['model_loadpath'].split('/')[-3] + '/' + self.run_time + '.p', 'wb'), obj=record)
            #print(i)
        '''
        return round(float(success_count)/length, 3), round(float(total_simulate)/length, 3), round(float(total_object)/length, 3), round(float(total_hit/total_all_imp), 3), round(float(out_match/length) , 3), record
    def train(self, simulate_epoch_number):
        self.best_success_rate_test = 0
        self.best_avg_turns_test = 0
        self.success_rate = self.avg_turns = self.avg_object = self.avg_recall = self.avg_out = 0
        self.success_rate_test = self.avg_turns_test = self.avg_object_test = self.avg_recall_test = self.avg_outs_test = 0
        for epoch in range(simulate_epoch_number):

            self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out, record = self.simulation_epoch(mode = 'train', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            self.success_rate_test, self.avg_turns_test, self.avg_object_test, self.avg_recall_test, self.avg_outs_test, record = self.simulation_epoch(mode = 'test', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            if self.best_success_rate_test < self.success_rate_test:
                self.best_success_rate_test = self.success_rate_test
                self.best_avg_turns_test = self.avg_turns_test
                self.save(self.path, epoch)
                pickle.dump(file=open('./records/' + self.parameter['model_savepath'].split('/')[-2] + '/' + self.run_time + '.p', 'wb'), obj=record)
            
            # write
            #wandb.log({'success_rate' : self.success_rate, 'avg_turns' : self.avg_turns, 'avg_object' : self.avg_object, 'avg_recall' : avg_recall, 'avg_out': self.avg_out, \
            #           'success_rate_test' : self.success_rate_test, 'avg_turns_test' : self.avg_turns_test, 'avg_recall_test' : self.avg_recall_test, 'avg_outs_test' : self.avg_outs_test})
        self.save(self.path, epoch)    
        return self.best_success_rate_test
    
    def save(self, path, epoch):
        model_file_name = os.path.join(path, "s" + str(float(self.success_rate_test)) + "_obj" + str(self.avg_object) + "_t" + str(self.avg_turns_test)\
                    + "_mr" + str(self.avg_recall_test) + "_outs" + str(self.avg_outs_test) + "_e-" + str(epoch) + ".pkl")

        torch.save(self.current_REFUEL.state_dict(), model_file_name)
 
    def load(self, path):
        self.current_REFUEL.load_state_dict(torch.load(path))
        self.target_REFUEL.load_state_dict(torch.load(path))
        




        

