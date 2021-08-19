#AI for self Driving Car 

#Importing Libaries
import numpy as np
import pandas as pd

#Import pytorch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import variable

#Creating The architecture

class Network(nn.Module):
    
    #Initialize the object
    def __init__(self, input_size, nb_action ):     #input_size= number of input neurons, output_size=No of output neuron
        super(Network, self).__init__()             #inherit from nn.module to use tools of module
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30 )                           #Full connection all hidden layer neurons connected to input(Liner) with hidden(2nd) layer
        self.fc2 = nn.Linear(30, nb_action )                            #connection of hidden layer with output layer  
        
    #Activate the neuron i.e perform fwd propagation
    def forward(self, state):  #input of neural net= state 
        #Activate hidden neuron
        x = F.relu(self.fc1(state, self))
        #return output neuron
        q_values = self.fc1(x)
        return q_values
    
#Implement Experince Replay
class replayMemo(object):
    
    def __init__(self, capacity):
        self.capacity = capacity     #max no of instances you want to have in the memory replay
        self.memory = []                #initialize empty list to store the transtion in it(100 events)
   
    #will append new event in the memory and make sure it has only 100 last events
         
    def push(self, event):      #event: tupleof four events (last state, new state st+1, last action ac, the last reward rt )
        self.memory.append(event)
         
        
































    

        
    
        
    


