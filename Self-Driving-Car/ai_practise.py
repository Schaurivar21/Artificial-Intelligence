#AI for self Driving Car 

#Importing Libaries
import numpy as np
import pandas as pd
import random
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
class ReplayMemo(object):
    
    def __init__(self, capacity):
        self.capacity = capacity        #max no of instances you want to have in the memory replay
        self.memory = []                #initialize empty list to store the transtion in it(100 events)
   
    #will append new event in the memory and make sure it has only 100 last events
         
    def push(self, event):              #event: tupleof four events (last state, new state st+1, last action ac, the last reward rt )
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    #to get random samples from memory
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))                  #zip is like resahape list function
        return map(lambda x: variable( torch.cat(x, 0)), samples)               #variable lambda is a function on which will will apply each of the samples
                                                                                #x: is going to be the variable of the function lambda
                                                                                # lambda func to return (that will convert the sample variables in torch tensor)
        
#Deep-Q learning Model implementation 

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):                           #initialize all the req object for deep q network input_size: no of dimension related to your input states 
        self.gamma = gamma
        self.reward_window = []                                                 #sliding window of mean of last 100 rewards                                                                 #nb_action :no of action that car can go  #gamma: delay coefficient
        #create Neural network
        self.model = Network(input_size, nb_action)                             #obecj of network class
        self.memory = ReplayMemo(100*1000)
        #optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)        #obj of Dqn class # connecting adam with Dqn model #learning rate decay
        self.last_state = torch.Tensor(input_size).unsqueeze(0)                 #unsqueeze for adding fake dim with index 0 #vector of 5 dim # 3-signal of 3 sensors and orinentation & - orinentation one more fake din corressponding with batch
        self.last_action = 0                                                    #can be 0 1 2 into angle of rotaion (0, 20 degree, -20 degree) 
        self.last_reward = 0                                                    #btw -1 to 1
        
    #func to select the right action each time  
    def select_action(self, state):  
        #get the best action to play as well as exploring other ways
        probs = F.softmax(self.model(variable(state, volatile = True))*7 )          #wrap state tensor into torch variable, not be using gradient of the state volatile = true it will exclude gradient associated with this input state
        #q values are output of NN 
        #T= 7 :higher the temperature parameter is more sure will be the NN closer to 1 
        action = probs.multinomial()                                            #ramdom draw from probabality distribution to get action        
        return action.data[0,0]                                                 #data stored in 0,0 miltinomial prob
    
    
    #process of fwd and bwd propagation     
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):                         #marcov desicion process
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)               #the actions that were chosen we use .gather
        next_outputs = self.model(batch_next_state).detach().max(1)[0]                                                     #            
        target = self.gamma*next_outputs + batch_reward
        td_loss =  F.smooth_l1_loss(outputs, target)                           #td temproral difference
        #back prop the error
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)                              #back propagate
        self.optimizer.step()                                                  #update the weight

    
    def update(self, reward, new_signal):   
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)              #new signal is input of 3 state of sensor and 2 orientations                                          
        #update memory 
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.LongTensor([self.last_reward])))
        
        
        
        
        
        
        
        
        
        
        
































    

        
    
        
    


