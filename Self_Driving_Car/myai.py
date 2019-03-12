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
    
#Implenting experience replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        #Reshape list
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#Implementing Deep Q learning model
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = opt.Adam(self.model.parameters(), lr = 0.01)
        self.last_state = torch.tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = func.softmax(self.model(Variable(state, volatile = True)) * 7)
        action = probs.multinomial()
        return action.data[0,0]
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #Grab the chosen action
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = func.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
        
        
        
        
        
        
        
        