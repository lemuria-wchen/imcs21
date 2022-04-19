# -*- coding: utf8 -*-

import torch
import torch.nn.functional
import os
import numpy as np
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            # hasattr: Return whether the object has an attribute with the given name.

            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            '''
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
            '''
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
            # specially initialize the parameters for batch normalization

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class REFUEL_net(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    dont know whether the non-linear is right
    """
    def __init__(self, input_size, output_size, rebulid_size):
        super(REFUEL_net, self).__init__()
        # different layers. Two layers.
        
        self.Encoder_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
        )
        self.Decoder_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_size),
            torch.nn.Softmax(dim=1)
        )
        self.Rebuild_layer = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, rebulid_size),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        if torch.cuda.is_available():
            x.cuda()
        #embedding = self.Encoder_layer(x)
        x1 = self.Encoder_layer(x)
        Action = self.Decoder_layer(x1)
        Rebuild = self.Rebuild_layer(x1)
        return Action, Rebuild

