import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from OpenMedicalChatBox.KRDS.qlearning.layers import NoisyLinear
import OpenMedicalChatBox.KRDS.dialog_config as dialog_config

    


class Knowledge_Graph_Reasoning(nn.Module):
    def __init__(self, num_actions, dise_start, act_cardinality, slot_cardinality, dise_sym_mat, sym_dise_mat, sym_prio, device):
        super(Knowledge_Graph_Reasoning, self).__init__()
        self.device = device
        self.num_actions = num_actions
        self.dise_start = dise_start
        self.act_cardinality = act_cardinality
        self.slot_cardinality = slot_cardinality
        self.dise_sym_mat = dise_sym_mat
        self.sym_dise_mat = sym_dise_mat
        self.sym_prio = sym_prio
    def forward(self, state):
        current_slots_rep = state[:, (2*self.act_cardinality+self.dise_sym_mat.size(0)+1):(2*self.act_cardinality+self.slot_cardinality)]
        # print("slot", self.slot_cardinality)
        # print("slot shape", current_slots_rep.size())
        
        batch_size = state.size(0)
        dise_num = self.dise_sym_mat.size(0)
        sym_num = self.dise_sym_mat.size(1)
        dise_start = self.dise_start
        sym_start = self.dise_start + dise_num

        sym_prio_ = self.sym_prio.repeat(batch_size,1).view(batch_size, -1)

        zeros = torch.zeros(current_slots_rep.size()).to(self.device)

        # not request->use prio prob
        # print('sym_prio_: ',sym_prio_)
        # print('current_slots_rep: ', current_slots_rep)
        sym_prio_prob = torch.where(current_slots_rep == 0, sym_prio_, current_slots_rep)
        # not sure->use prio prob
        sym_prio_prob = torch.where(sym_prio_prob == -2, sym_prio_, sym_prio_prob)
        #sym_prio_prob = torch.where(sym_prio_prob == -1, zeros, sym_prio_prob)
        # print("sym_prio_prob", sym_prio_prob)

        dise_prob = torch.matmul(sym_prio_prob, self.sym_dise_mat)
        sym_prob = torch.matmul(dise_prob, self.dise_sym_mat)

        action = torch.zeros(batch_size, self.num_actions).to(self.device)
        action[:, dise_start:sym_start] = dise_prob
        action[:, sym_start:] = sym_prob
        # print("knowledge action", action)
        return action

class KR_DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, num_actions, relation_init, dise_start, act_cardinality, slot_cardinality,  sym_dise_pro, dise_sym_pro, sym_prio, device):
        super(KR_DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.dise_start = dise_start
        self.act_cardinality = act_cardinality
        self.slot_cardinality = slot_cardinality
        self.sym_dise_mat = sym_dise_pro
        self.dise_sym_mat = dise_sym_pro
        self.sym_prio = sym_prio

        self.fc1 = nn.Linear(self.input_shape, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_actions)
        self.tran_mat = Parameter(torch.Tensor(relation_init.size(0),relation_init.size(1)))
        self.knowledge_branch = Knowledge_Graph_Reasoning(self.num_actions, self.dise_start, self.act_cardinality, self.slot_cardinality, 
            self.dise_sym_mat, self.sym_dise_mat, self.sym_prio, device)

        self.tran_mat.data = relation_init

        #self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.tran_mat.data.uniform_(-stdv, stdv)
    def forward(self, state, sym_flag):
        # print('sym_flag.size(): ', sym_flag.size())
        x = F.relu(self.fc1(state))
        x = self.fc2(x)

        rule_res = self.knowledge_branch(state)
        relation_res = torch.matmul(x, F.softmax(self.tran_mat, 0))
        # dqn+knowledge+relation
        x = torch.sigmoid(x) + torch.sigmoid(relation_res) + rule_res

        x = x * sym_flag
        
        return x

    def predict(self, x, sym_flag):
        with torch.no_grad():
            a = self.forward(x, sym_flag).max(1)[1].view(1, 1)
        return a.item()


    
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy=False, sigma_init=0.5):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.noisy = noisy
        self.body = body(input_shape, num_actions, noisy, sigma_init)

        self.fc1 = nn.Linear(self.body.feature_size(), 512) 
        self.fc2 = nn.Linear(512, self.num_actions) 

    def forward(self, x):
        x = self.body(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_noise(self):
        if self.noisy:
            self.body.sample_noise()
            self.fc1.sample_noise()
            self.fc2.sample_noise()

    def predict(self, x):
        # print(self.fc1.weight)
        with torch.no_grad():
            self.sample_noise()
            a = self.forward(x).max(1)[1].view(1, 1)
        return a.item()
