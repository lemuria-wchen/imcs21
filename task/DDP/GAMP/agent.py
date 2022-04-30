import OpenMedicalChatBox
from .network import Discriminator, Generator, Inference, init_weights
import torch
import random
import pickle
import math
import copy
from torch import optim
from .dataloader import get_loader, LSTM_get_loader, Dis_get_loader, Inf_get_loader
import sys
import multiprocessing as mp
import math
import numpy as np
import OpenMedicalChatBox.GAMP.utils as utils
import os
import time
from collections import deque
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
def progress_bar(bar_len, SR, avg_turn, avg_obj, avg_recall, avg_out, SRT, ATT, avg_recall_t,  best_SR, avg_out_t, best_correct_num_dis, currentNumber, wholeNumber):
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
        '[%d/%d][%s] %s%s \033[31;1mSR\033[0m = %3f \033[36;1mAvg_turn\033[0m= %3f \033[36;1mAvg_obj\033[0m= %3f \033[36;1mAvg_recall\033[0m= %3f \033[36;1mAvg_out\033[0m= %3f \033[33;1mSR_t\033[0m= %3f \033[33;1mAT_t\033[0m= %3f \033[33;1mrecall_t\033[0m= %3f \033[33;1mout_t\033[0m= %3f \033[33;1mBest_SR\033[0m= %3f \033[33;1mBest_Dis\033[0m= %3f \r' %\
         (int(currentNumber),int(wholeNumber), bar, '\033[32;1m%s\033[0m' % percents, '%', SR, avg_turn, avg_obj, avg_recall, avg_out, SRT, ATT, avg_recall_t, avg_out_t, best_SR, best_correct_num_dis))
    sys.stdout.flush()

def progress_bar_warm_gen(bar_len, SR, loss, best_SR, currentNumber, wholeNumber):
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
        '[%d/%d][%s] %s%s \033[31;1mSR\033[0m = %3f \033[36;1mLoss\033[0m= %3f Best_SR\033[0m = %3f \033[36;1m \r' %\
         (int(currentNumber),int(wholeNumber), bar, '\033[32;1m%s\033[0m' % percents, '%', SR, loss, best_SR))
    sys.stdout.flush()

class Agent(object):
    def __init__(self, slot_set, disease_set, goal_set, parameter):
        self.num_slot = len(slot_set)
        self.num_disease = len(disease_set)
        self.slot_set = slot_set
        self.disease_set = disease_set
        for idx, disease in enumerate(list(disease_set.keys())):
            self.disease_set[disease] =  idx
        self.goal_set_path =  parameter['goal_set']
        self.goal_test_path =  parameter['goal_set_test']
        self.batch_size = parameter['batch_size']
        self.max_turn = parameter['max_turn']
        self.goal_set = goal_set
        self.load_model = parameter['load']
        self.idx2sym = dict()
        for key, value in self.slot_set.items():
            self.idx2sym[value] = key

        self.parameter = parameter
        self.num_cores = int(mp.cpu_count())
        
        self.run_time = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
        self.path = self.parameter['model_savepath'] + self.run_time
        self.fake_seq = deque(maxlen=1000)
        
        if self.parameter['train_mode'] and not(os.path.exists(self.path)):
            os.makedirs(self.path)
        
        self.build_model()
        #print("本地计算机有: " + str(self.num_cores) + " 核心")

    def build_model(self, mode = 'normal'):
        self.Discriminator = Discriminator(input_size= self.num_slot)
        self.Generator = Generator(slot_set = self.slot_set, input_size= 64, hidden_size=self.num_slot + 1)
        self.Inference = Inference(input_size= self.num_slot, output_size=self.num_disease)
        self.optimizer_gen = optim.Adam(self.Generator.parameters(), 0.0001) 
        self.optimizer_gen_pre = optim.Adam(self.Generator.parameters(), 0.01) 
        self.optimizer_dis = optim.Adam(self.Discriminator.parameters(), 0.01)
        self.optimizer_inf = optim.Adam(self.Inference.parameters(), 0.01)
        self.device = torch.device('cuda:' + str(self.parameter['cuda_idx']) if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')
        self.Discriminator.to(self.device)

        self.Generator.to(self.device)
        self.Inference.to(self.device)
        init_weights(self.Discriminator, mode)
        init_weights(self.Generator, mode)
        init_weights(self.Inference, mode)
        if self.load_model:
            self.load(self.parameter['load_path'])
        

    def update_state(self, action, goal, goal_disease, state, turn, data_length, tau = 0.995, lbd = 0.5, epi = 0.5, mode='train'):
        episode_over = torch.ones(len(action))
        fake_symptom = list()
        length = action.size(0)
        new_state = copy.deepcopy(state)
        acc_flag = torch.zeros(len(action))
        reach_max_turn = torch.zeros(len(action))
        new_state_seq = list()
        disease_prob = list()
        R_D = torch.zeros(len(action))
        R_M = torch.zeros(len(action))
        R_F = torch.zeros(len(action))
        new_state, fake_symptom = utils.generate_new_state(state, action, data_length, goal, self.num_slot)
        self.fake_seq.extend(fake_symptom)
        
        # 2021.12.20
        CELoss = torch.nn.CrossEntropyLoss()
        new_state = torch.LongTensor(new_state).to(self.device)

        for i in range(len(action)):
            before_symptom, current_symptom = utils.seq_generate(state[i], action[i], data_length[i], goal[i], self.num_slot)
            before_symptom = before_symptom.to(self.device)
            current_symptom = current_symptom.to(self.device)
            before_predict = self.Inference(before_symptom.unsqueeze(0))
            current_predict = self.Inference(current_symptom.unsqueeze(0))
            R_M[i] = CELoss(before_predict, goal_disease[i].view(-1)).item() - CELoss(current_predict, goal_disease[i].view(-1)).item()

            current_seq_present = utils.state_representation_abs(state[i], action[i], data_length[i], self.num_slot)
            new_state_seq.append(current_seq_present)
            disease_prob.append(before_predict)
        new_state_seq_total = torch.stack(new_state_seq).to(self.device)
        disease_prob = torch.stack(disease_prob).to(self.device)
        R_D = self.Discriminator(new_state_seq_total)
        R_M = R_M.to(self.device)
        R_F = (1 - lbd) * R_M + lbd * (R_D.view(-1) - epi)
        for i in range(len(action)):
            if torch.max(disease_prob[i], dim = 1).values > tau:
                if torch.max(disease_prob[i], dim = 1).indices == goal_disease[i]:
                    acc_flag[i] = 1    
                    episode_over[i] = 0
                else:
                    acc_flag[i] = -100 
                    episode_over[i] = 0
                continue
                
            if turn == self.max_turn and action[i] < self.num_slot:
                episode_over[i] = 0
                reach_max_turn[i] = 1
                if torch.max(disease_prob[i], dim = 1).indices == goal_disease[i]:
                    acc_flag[i] = 1
                    #elif state[i][simulate[i]] != 0:
                    #    status[i] = 4
                    #    reward[turn][i] = self.n                  
            else: 
                if action[i] < self.num_slot:
                    data_length[i] += 1 
                else:
                    if torch.max(disease_prob[i], dim = 1).indices == goal_disease[i]:
                        acc_flag[i] = 1
                    episode_over[i] = 0
                
        return new_state, R_F, episode_over, acc_flag, reach_max_turn

    def g_step(self, origin_state, origin_data_length, goal, goal_disease, mode):
        # goal : vector
        fake_sequence = list()
        loss = 0
        total_turn = 0
        acc = 0
        reach_max = 0
        length = origin_state.size(0)
        
        data_length = copy.deepcopy(origin_data_length)
        #####change
        #rebuild_record_list = list()
        action_prob_list = list()
        state_record = torch.zeros((self.max_turn+2, length, self.num_slot))
        #reward_gamma_list = torch.zeros((self.max_turn+1, length)).to(self.device)
        loss_total = 0
        reward_record = list()
        prob_record = list()
        #action_record = torch.zeros((self.max_turn+2, length))
        episode_over_list = list()
        final_state = torch.zeros(length, self.num_slot)
        action_list = list()
        acc_flag = torch.zeros(length)
        turn = 1
        state = origin_state
        episode_over_list.append(np.ones(length))
        #reward_record.append(np.zeros(length))
        
        final_status = torch.zeros(length)

        ######## Symptom stage ########
        ##### 调整reward和state的下标，如论文
        for i in range(0, self.max_turn):
            fake_symptom = list()
            # copy
            action_prob,(h_n,c_n) = self.Generator(state, data_length)
            '''
            output_max = torch.argmax(output, dim = 1)
            # 如果梯度会有问题的话也可以用argmax
            action_prob = output[output_max]
            '''
            action, prob = utils.random_generate(action_prob, mode)
            prob_record.append(prob)
                            
            action_prob_list.append(action_prob)
            action_list.append(np.array(action.cpu()))
            new_state, reward, episode_over, acc_flag_turn, reach_max_turn = self.update_state(action, goal, goal_disease, state, turn, data_length)
            
            episode_over = episode_over * episode_over_list[i]
           
            episode_over_list.append(episode_over)
            acc_flag = acc_flag + acc_flag_turn
            reward_record.append(reward.cpu().detach().numpy())
            #state_record[i+1,:,:] = new_state 
            state = copy.deepcopy(new_state)
            turn += 1
        # 每个对话持续了多少轮
        episode_over_list = torch.tensor(episode_over_list, device=self.device)[0:-1]
        reward_record = torch.tensor(reward_record, device=self.device)
        prob_record = torch.stack(prob_record).to(self.device)
        action_prob_list =  torch.stack(action_prob_list).to(self.device)
        
        over_turn = torch.sum(episode_over_list, axis = 0).long()
        reach_max_turn = reach_max_turn.to(self.device)
        
        for j in range(length):
            final_state[j] = state_record[over_turn[j], j]
        # 判断 max_turn 之前对话是否结束
        reach_max_turn = reach_max_turn * episode_over_list[i,:]
        acc += len(acc_flag[acc_flag > 0])
        reach_max += reach_max_turn.sum()
        hit_match = 0
        all_match = 0
        all_already_find = []
        for i in range(length):
            already_find = []
            for j in range(origin_data_length[i], data_length[i]+1):
                if int(new_state[i][j]) in goal[i].keys() and int(new_state[i][j]) not in already_find:
                    hit_match += 1
                    already_find.append(int(new_state[i][j]))
            all_already_find.append(already_find)
            all_match += len(goal[i].keys()) - origin_data_length[i]
        total_turn += torch.sum(over_turn)
        if mode == 'train':            
            loss = torch.sum(episode_over_list * reward_record * prob_record)
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                self.optimizer_gen.step()
                self.Generator.zero_grad()
            loss_total = loss.item()

        return loss_total, hit_match, all_match, acc, reach_max, total_turn, all_already_find
    
    def d_steps(self, index):
        best_correct_num_dis = 0
        real_symptom = random.sample(self.real_symptom, 200)
        fake_symptom = random.sample(self.fake_seq, 200) 
        BCELoss = torch.nn.BCELoss()
        total_symptom = real_symptom + fake_symptom
        label = [1 for j in real_symptom] + [0 for i in fake_symptom] 
        dis_dataset = Dis_get_loader(total_symptom, label)
        for i in range(25):
            correct_num  = 0
            total_loss = 0
            length = 0
            for j, (state, label) in enumerate(dis_dataset):
                state = state.to(self.device)
                label = label.float().to(self.device)
                #label = torch.FloatTensor(label).to(self.device)
                length += state.size(0)
                output = self.Discriminator(state).squeeze(-1) 
                loss = BCELoss(output, label)
                self.optimizer_dis.zero_grad()
                loss.backward()
                self.optimizer_dis.step()
                total_loss += loss.item()

                output_max = torch.where(output < 0.5, torch.zeros_like(output), torch.ones_like(output))
                for j in range(output_max.size()[0]):
                    if output_max[j] == label[j]:
                        correct_num += 1
            if best_correct_num_dis < correct_num/length:
                best_correct_num_dis = correct_num/length
        return best_correct_num_dis
    
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
        
        #pool = mp.Pool(self.num_cores)
        for i, (origin_state, goal, goal_disease, data_length, consult_id) in enumerate(dataset):

            
            temp_object = 0

            origin_state = origin_state.to(self.device)
            origin_length = copy.deepcopy(data_length)
            goal = goal
            goal_disease = goal_disease.to(self.device)
            
            loss, hit_match, all_match, acc_flag, reach_max_turn, turn, action_record = self.g_step(origin_state, origin_length, goal, goal_disease, mode)
            
            
            total_object += loss
            success_count += acc_flag
            out_match += reach_max_turn
            total_simulate += turn
            total_hit += hit_match
            total_all_imp += all_match
            length += origin_state.size(0)
            #print(i)
            if self.parameter['train_mode'] != 'train':
                for i in range(len(consult_id)):
                    symptom_record = [self.idx2sym[j] for j in action_record[i]]
                    record[consult_id[i]] = symptom_record

        '''
        if self.parameter['train_mode'] != 'train':
            pickle.dump(file=open('./records/' + self.parameter['load_path'].split('/')[-3] + '/' + self.parameter['load_path'].split('/')[-2] + '.p', 'wb'), obj=record)
            #print(i)
        '''
        return round(float(success_count)/length, 3), round(float(total_simulate)/length, 3), round(float(total_object)/length, 3), round(float(total_hit/int(total_all_imp)), 3), round(float(out_match/length) , 3), record
    
    def train(self, simulate_epoch_number):
        self.best_success_rate_test = 0
        self.best_avg_turns_test = 0
        self.success_rate = self.avg_turns = self.avg_object = self.avg_recall = self.avg_out = 0
        self.success_rate_test = self.avg_turns_test = self.avg_object_test = self.avg_recall_test = self.avg_outs_test = 0
        for epoch in range(simulate_epoch_number):

            self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out, record = self.simulation_epoch(mode = 'train', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out, record = self.simulation_epoch(mode = 'train', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            best_correct_num_dis = self.d_steps(epoch)
            self.success_rate_test, self.avg_turns_test, self.avg_object_test, self.avg_recall_test, self.avg_outs_test, record = self.simulation_epoch(mode = 'test', epoch = epoch, simulate_epoch_number = simulate_epoch_number)
            if self.best_success_rate_test < self.success_rate_test:
                self.best_success_rate_test = self.success_rate_test
                self.best_avg_turns_test = self.avg_turns_test
                self.save(self.path, epoch)
                pickle.dump(file=open( self.parameter['model_savepath'].split('/')[-2] + '/records/' + '/' + self.run_time + '.p', 'wb'), obj=record)
            
            # write
            #wandb.log({'success_rate' : self.success_rate, 'avg_turns' : self.avg_turns, 'avg_object' : self.avg_object, 'avg_recall' : self.avg_recall, 'avg_out': self.avg_out, \
            #           'success_rate_test' : self.success_rate_test, 'avg_turns_test' : self.avg_turns_test, 'avg_recall_test' : self.avg_recall_test, 'avg_outs_test' : self.avg_outs_test})

            progress_bar(10, self.success_rate, self.avg_turns, self.avg_object, self.avg_recall, self.avg_out, self.success_rate_test, self.avg_turns_test, self.avg_recall_test, self.best_success_rate_test, self.avg_outs_test, best_correct_num_dis, epoch, simulate_epoch_number)
                #self.save(self.parameter['model_savepath'] + '/newest/')
        self.save(self.path, epoch)
        return self.best_success_rate_test
    
    def save(self, path, epoch):
        model_file_name = os.path.join(path, "s" + str(float(self.success_rate_test)) + "_obj" + str(self.avg_object) + "_t" + str(self.avg_turns_test)\
                    + "_mr" + str(self.avg_recall_test) + "_outs" + str(self.avg_outs_test) + "_e-" + str(epoch))

        torch.save(self.Generator.state_dict(), model_file_name + '_Generator.pkl')
        torch.save(self.Discriminator.state_dict(), model_file_name + '_Discriminator.pkl')
        torch.save(self.Inference.state_dict(), model_file_name + '_Inference.pkl')
 
    def load(self, path):
        self.Generator.load_state_dict(torch.load(path+ '_Generator.pkl'))
        self.Discriminator.load_state_dict(torch.load(path+ '_Discriminator.pkl'))
        self.Inference.load_state_dict(torch.load(path+ '_Inference.pkl'))

    def warm_start(self):
        simplefilter(action='ignore', category=FutureWarning)
        LSTM_dataset = LSTM_get_loader(self.slot_set,  self.disease_set, self.goal_set_path, batch_size=128, mode = 'train')
        print("########## Warm-Start the generator ############")
        best_correct_num_gen = 0
        best_correct_num_dis = 0
        best_correct_num_inf = 0
        loss = 0
        CELoss = torch.nn.CrossEntropyLoss()
        BCELoss = torch.nn.BCELoss()
        for i in range(100):
            correct_num  = 0
            total_loss = 0
            length = 0
            for j, (origin_state, goal_symptom, data_length) in enumerate(LSTM_dataset):
                origin_state = origin_state.to(self.device)
                data_length = data_length.to(self.device)
                length += origin_state.size(0)
                output,(h_n,c_n) = self.Generator(origin_state, data_length)
                loss = CELoss(output, goal_symptom)
                self.optimizer_gen_pre.zero_grad()
                loss.backward()
                self.optimizer_gen_pre.step()
                total_loss += loss.item()

                output_max = torch.argmax(output, dim = 1)
                for j in range(output_max.size()[0]):
                    if output_max[j] == goal_symptom[j]:
                        correct_num += 1
            if best_correct_num_gen < correct_num/length:
                best_correct_num_gen = correct_num/length
            progress_bar_warm_gen(20, correct_num/length, total_loss/length, best_correct_num_gen, i, 100)
        
        self.real_symptom = list()
        fake_symptom = list()
        print("#### Now is generating fake sample and true sample ####")
        for j, (origin_state, goal_symptom, data_length) in enumerate(LSTM_dataset):
            origin_state = origin_state.to(self.device)
            output,(h_n,c_n) = self.Generator(origin_state, data_length)
            output_max = torch.argmax(output, dim = 1)
            
            for i in range(origin_state.size(0)):
                real_state = utils.state_representation_abs(origin_state[i], goal_symptom[i], data_length[i], self.num_slot)
                #real_state = torch.ones(41)
                self.real_symptom.append(real_state)
                if goal_symptom[i] == len(self.slot_set) or output_max[i] == len(self.slot_set) or goal_symptom[i] == output_max[i]:
                    continue

                fake_state = utils.state_representation_abs(origin_state[i], output_max[i], data_length[i], self.num_slot)
                #fake_state = torch.zeros(41)
                fake_symptom.append(fake_state)
        real_symptom = self.real_symptom
        total_symptom = real_symptom + fake_symptom
        label = [1 for j in real_symptom] + [0 for i in fake_symptom] 
        dis_dataset = Dis_get_loader(total_symptom, label)

        print("#### Now training the discriminator ####")
        for i in range(100):
            correct_num  = 0
            total_loss = 0
            length = 0
            for j, (state, label) in enumerate(dis_dataset):
                state = state.to(self.device)
                label = label.float().to(self.device)
                #label = torch.FloatTensor(label).to(self.device)
                length += state.size(0)
                output = self.Discriminator(state).squeeze(-1) 
                loss = BCELoss(output, label)
                self.optimizer_dis.zero_grad()
                loss.backward()
                self.optimizer_dis.step()
                total_loss += loss.item()

                output_max = torch.where(output < 0.5, torch.zeros_like(output), torch.ones_like(output))
                for j in range(output_max.size()[0]):
                    if output_max[j] == label[j]:
                        correct_num += 1
            if best_correct_num_dis < correct_num/length:
                best_correct_num_dis = correct_num/length
            progress_bar_warm_gen(20, correct_num/length, total_loss/length, best_correct_num_dis, i, 100)

        print("#### Now is generating fake sample and true sample ####")
        
        training_set = self.goal_set['train']
        state, disease = utils.dataset_generate(training_set, self.slot_set, self.disease_set)
        Inf_dataset = Inf_get_loader(state, disease)
        print("#### Now training the inference ####")
        for i in range(15):
            correct_num  = 0
            total_loss = 0
            length = 0
            max_prob = 0
            for j, (state, label) in enumerate(Inf_dataset):
                state = state.to(self.device)
                label = label.to(self.device)
                length += state.size(0)
                output = self.Inference(state.float())
                loss = CELoss(output, label)
                self.optimizer_inf.zero_grad()
                loss.backward()
                self.optimizer_inf.step()
                total_loss += loss.item()

                output_max = torch.argmax(output, dim = 1)

                for j in range(output_max.size()[0]):
                    max_prob += output[j][output_max[j]]
                    if output_max[j] == label[j]:
                        correct_num += 1
            max_prob = max_prob / length
            if best_correct_num_inf < correct_num/length:
                best_correct_num_inf = correct_num/length
            progress_bar_warm_gen(20, correct_num/length, max_prob, best_correct_num_inf, i, 15)


