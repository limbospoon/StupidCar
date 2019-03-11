# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as opt
import torch.autograd as ag

from torch.autograd import Variable

#Creating NN architecture

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        #Hidden nuerons
        x = func.relu(self.fc1(state))
        
        #Output neurons
        q_values = self.fc2(x)
        return q_values