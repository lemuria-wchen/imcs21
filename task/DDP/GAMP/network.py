# -*- coding: utf8 -*-

import torch
import torch.nn.functional
import os
import numpy as np
from torch.nn import init
from torch.nn.modules.activation import Sigmoid
import pickle
def init_weights(net, init_type='normal', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # hasattr: Return whether the object has an attribute with the given name.

            if init_type == 'normal':
                init.normal_(m.weight.data, mean = 0.0, std = 0.01)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
            # specially initialize the parameters for batch normalization

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class Discriminator(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    dont know whether the non-linear is right
    """
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        # different layers. Two layers.
        self.dis = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1),
            
        )
        self.sig = torch.nn.Sigmoid()
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        #embedding = self.Encoder_layer(x)
        x1 = self.dis(x)
        x2 = self.sig(x1)
        return x2

class Generator(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    dont know whether the non-linear is right
    """
    def __init__(self, slot_set, input_size, hidden_size):
        super(Generator, self).__init__()
        self.slot_set = slot_set
        self.embedding = torch.nn.Embedding(len(self.slot_set), input_size)
        self.LSTM_layer = torch.nn.LSTM(input_size = input_size, hidden_size=hidden_size, batch_first=True)
        self.softmax = torch.nn.Softmax(dim=0)
    def forward(self, idx, length):
                   
        data_input = self.embedding(idx)
        out, (h_n, c_n) = self.LSTM_layer(data_input) 
        output = torch.zeros((out.size(0), out.size(2)))
        output_soft = torch.softmax(out, dim = 2)
        for i in range(out.size(0)):
            output[i,:] = output_soft[i, length[i]-1, :]

        return output, (h_n, c_n)

class Inference(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Inference, self).__init__()
        # different layers. Two layers.
        self.dis = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, output_size),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        #embedding = self.Encoder_layer(x)
        x1 = self.dis(x)
        return x1

