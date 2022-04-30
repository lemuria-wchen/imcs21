import logging
import pandas as pd
import random
from torch.utils import data
from .utils import *
import torch
import pickle
import argparse
import logging
import os
from datetime import datetime
from torch.backends import cudnn
import torch.nn.utils.rnn as rnn_utils
import copy

class DataLoader(data.Dataset):

    def __init__(self, slot_set,  disease_set, goal_test_path, mode='train'):
        self.goal_test = pickle.load(file=open(goal_test_path, "rb"))
        self.mode = mode
        self.disease_set = dict()

        self.dataset = self.goal_test[mode]
        self.num_slot = len(slot_set)
        for idx, disease in enumerate(list(disease_set.keys())):
            self.disease_set[disease] =  idx
        self.num_disease = len(disease_set)
        self.slot_set = slot_set
        #self.disease_set = disease_set

        logging.info("image count in {} path :{}".format(self.mode, len(self.dataset)))
        
    def init_setting(self, state):
        goal = dict()
        
        explicit_symptom = list(state['goal']['explicit_inform_slots'])
        origin_state = torch.LongTensor([self.slot_set[symptom] for symptom in explicit_symptom])
        for key, value in state['goal']['explicit_inform_slots'].items():
            goal[self.slot_set[key]] = value
        for key, value in state['goal']['implicit_inform_slots'].items():
            goal[self.slot_set[key]] = value
        return origin_state, goal 
    
    def __getitem__(self, index):
        data = self.dataset[index]
        origin_state, goal = self.init_setting(data)
        goal_disease = self.disease_set[data['disease_tag']]
        consult_id = data['consult_id']
        return origin_state, goal, goal_disease, consult_id

    def __len__(self):
        return len(self.dataset)


def get_loader(slot_set,  disease_set, goal_test_path, batch_size=16, num_workers=4, mode='train', shuffle = True):
    """
        Builds and returns Data loader.
        :param batch_size:
        :param num_workers:
        :param mode:
        :param augmentation_prob:
        :return:
    """
    dataset = DataLoader(slot_set, disease_set, goal_test_path,
                          mode=mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    return data_loader

def collate_fn(data_tuple):
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    
    label = torch.LongTensor(label)
    data_length = torch.LongTensor([len(q) for q in data])
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)
    return data, label, data_length

def collate_fn_real(data_tuple):
    data_tuple.sort(key=lambda x: len(x[0]), reverse=True)
    data = [sq[0] for sq in data_tuple]
    label = [sq[1] for sq in data_tuple]
    disease_tag = torch.LongTensor([sq[2] for sq in data_tuple])
    consult_id = [sq[3] for sq in data_tuple]
    data_length = torch.LongTensor([len(q) for q in data])
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)
    return data, label, disease_tag, data_length, consult_id

class LSTM_DataLoader(data.Dataset):

    def __init__(self, slot_set, disease_set, goal_test_path, mode='train'):
        self.goal_test = pickle.load(file=open(goal_test_path, "rb"))
        self.mode = mode
        self.disease_set = dict()

        self.dataset = self.goal_test[mode]
        self.num_slot = len(slot_set)
        for idx, disease in enumerate(list(disease_set.keys())):
            self.disease_set[disease] =  idx
        self.num_disease = len(disease_set)
        self.slot_set = slot_set
        #self.disease_set = disease_set
        self.state, self.symptom = self.init_setting(self.dataset)
        logging.info("image count in {} path :{}".format(self.mode, len(self.dataset)))
        
    def init_setting(self, dataset):
        exist_symptom = list()
        goal_symptom = list()
        for state in dataset:
            explicit_symptom = list(state['goal']['explicit_inform_slots'])
            implicit_inform_slots = list(state['goal']['implicit_inform_slots'])
            origin_state = [self.slot_set[symptom] for symptom in explicit_symptom]
            expand_state = [self.slot_set[symptom] for symptom in implicit_inform_slots]

            for i in range(len(expand_state)):
                exist_symptom.append(torch.LongTensor(origin_state + expand_state[:i:]))
                goal_symptom.append(expand_state[i])
            '''
            exist_symptom.append(torch.LongTensor(origin_state + expand_state))
            goal_symptom.append(len(self.slot_set))
            '''
        return exist_symptom, torch.Tensor(goal_symptom) 
    
    def __getitem__(self, index):
        origin_state = self.state[index]
        goal_sym = self.symptom[index]

        # for the reason  that torch classification criterion can only accept classes label [0, C-1]

        return origin_state, goal_sym

    def __len__(self):
        return len(self.dataset)

class Dis_DataLoader(data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data_ = self.data[index]
        label_ = self.label[index]

        # for the reason  that torch classification criterion can only accept classes label [0, C-1]

        return data_, label_

    def __len__(self):
        return len(self.data)

class Inf_DataLoader(data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data_ = self.data[index]
        label_ = self.label[index]

        # for the reason  that torch classification criterion can only accept classes label [0, C-1]

        return data_, label_

    def __len__(self):
        return len(self.data)

def get_loader(slot_set,  disease_set, goal_test_path, batch_size=16, num_workers=0, mode='train', shuffle = True):

    dataset = DataLoader(slot_set, disease_set, goal_test_path,
                          mode=mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn_real)
    return data_loader


def LSTM_get_loader(slot_set,  disease_set, goal_test_path, batch_size=16, num_workers=0, mode='train', shuffle = True):

    dataset = LSTM_DataLoader(slot_set, disease_set, goal_test_path,
                          mode=mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader

def Dis_get_loader(data_list, label, batch_size=16, num_workers=0,  shuffle = True):

    dataset = Dis_DataLoader(data_list, label)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    return data_loader

def Inf_get_loader(data_list, label, batch_size=16, num_workers=0, mode='train', shuffle = True):

    dataset = Inf_DataLoader(data_list, label)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    return data_loader
if __name__ == '__main__':
    pass