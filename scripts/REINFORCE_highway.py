import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import highway_env

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    # env = gym.make('CartPole-v1')
    env = gym.make('highway-bs-v0')

    pi = Policy()
    score = 0.0
    print_interval = 20
    
    ho_prob = 1e-9
    tele_total_rewards = []
    tran_total_rewards = []
    ho_rwd = 0.0 
    
    
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            print('a',a)
            print('item',a.item())
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            print('info',info)
            tele_total_rewards.append(info['agents_tele_all_rewards'])
            tran_total_rewards.append(info['agents_tran_all_rewards'])
            ho_prob = info['agents_ho_prob']
            
        pi.train_net()
        ho_rwd += (ho_prob)
        tel_reward_all_mean = np.mean(tele_total_rewards)
        tran_reward_all_mean = np.mean(tran_total_rewards)
        ho_prob = 1e-9
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg reward : {},avg tele reward: {}, avg tran reward {}, avg ho prob {}".format(n_epi, score/print_interval,tel_reward_all_mean,tran_reward_all_mean,ho_rwd/print_interval))
            score = 0.0
            ho_prob = 1e-9
            tele_total_rewards = []
            tran_total_rewards = []
            ho_rwd = 0.0 

    env.close()
    
if __name__ == '__main__':
    main()