import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import numpy as np
import matplotlib.pyplot as plt
import visdom

import os
import random
from abc import ABC, abstractmethod


class SortingAgent(ABC):
    """Base sorting agent"""
    def __init__(self, arr):
        self.arr = arr

    def switch_elements(self, idx_1, idx_2):
        temp = self.arr[idx_1]
        self.arr[idx_1] = self.arr[idx_2]
        self.arr[idx_2] = temp

    @abstractmethod
    def update(self):
        pass


class RandomAgent(SortingAgent):
    """Sorts with a random policy"""
    def update(self):
        self.switch_elements(
            random.randint(0, len(self.arr)-1),
            random.randint(0, len(self.arr)-1),
        )


def get_score(arr, idx):
    """Score in immediate area"""
    score = 0
    score += -1 if idx - 1 >= 0 and not arr[idx] >= arr[idx-1] else 1
    score += -1 if idx + 1 < len(arr) and not arr[idx] <= arr[idx+1] else 1
    return score

def get_sortedness(arr):
    score = 0
    for i in range(len(arr)-1):
        score += arr[i+1] - arr[i]
    return score


class DQN(nn.Module):
    """Q table approximater"""
    def __init__(self, n):
       super(DQN, self).__init__()
       self.net = nn.Sequential(
           nn.Linear(n, 32),
           nn.ReLU(),
           nn.Linear(32, 64),
           nn.ReLU(),
           nn.Linear(64, n**2),
           nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.net(x)


class DQAgent(SortingAgent):
    """Deep Q learning agent"""
    def __init__(self, arr, is_train=False):
        super(DQAgent, self).__init__(arr)
        self.is_train = is_train
        self.dqn = DQN(len(arr))
        self.discount = 0.8
        self.loss_f = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)

        self.total_loss = 0

    def update(self):

        if self.is_train:
            # Gamble during training
            if random.random() > 0.1:

                self.optimizer.zero_grad()
                
                q_predict, res = torch.max(self.dqn(torch.Tensor(self.arr)), 0)
                idx_1 = res // len(self.arr)
                idx_2 = res % len(self.arr)
                self.switch_elements(idx_1, idx_2)
                
                # Update dqn params with bellman
                with torch.no_grad():
                    q_prime = torch.max(self.dqn(torch.Tensor(self.arr)))
                reward = get_sortedness(arr)
                q_target = reward + self.discount * q_prime

                loss = self.loss_f(q_predict, q_target)
                loss.backward()
                self.optimizer.step()

                self.total_loss += loss.item()
                
            else:
                self.switch_elements(
                    random.randint(0, len(self.arr)-1),
                    random.randint(0, len(self.arr)-1),
                )
            
        else:
            with torch.no_grad():
                _, res = torch.max(self.dqn(torch.Tensor(self.arr)), 0)
                self.switch_elements(
                    res // len(self.arr),
                    res % len(self.arr),
                )
            

# === MAIN ===

vis = visdom.Visdom()

arr = list(range(10))
random.shuffle(arr)
agent = DQAgent(arr, is_train=True)
update_rate = 10000
n_iter = 150000
save_path = './data'

arr_log = []
loss_log = []
for i in range(n_iter):
    
    agent.update()

    # Update visdom and save params
    if (i + 1) % update_rate == 0:
        loss_log.append(agent.total_loss / (i + 1))
        vis.line(
            Y=np.array(loss_log),
            X=np.array([update_rate*x for x in range(1, len(loss_log)+1)]),
            opts=dict(
                title='DQN Average Loss',
                webgl=True,
            ),
            win='Losses',
        )

        arr_log.insert(0, f"<tr><td>{i + 1}</td><td>{agent.arr}</td></tr>")
        vis.text(
            '<table>'+''.join(arr_log)+'</table>',
            win="Result",
        )

        torch.save(agent.dqn.state_dict(), os.path.join(save_path, f"dqn_{i+1}"))
