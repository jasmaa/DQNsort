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

# === Testing reward funcs ===

def get_local_score(arr, idx):
    """Check if considered sorted in local area"""
    score = 0
    score += -1 if idx - 1 >= 0 and not arr[idx] >= arr[idx-1] else 1
    score += -1 if idx + 1 < len(arr) and not arr[idx] <= arr[idx+1] else 1
    return score

def get_ascending_reward(arr):
    """Reward based on number of ascending items"""
    score = 0
    for i in range(len(arr)-1):
        score += arr[i+1] - arr[i]
    return score

def get_inplace_reward(arr):
    """Reward based on position of item"""
    score = 0
    for i, val in enumerate(arr):
        score += 1 if i == val else 0
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

    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.dqn.eval()

    def update(self):

        if self.is_train:
            # Gamble during training
            if random.random() > 0.1:

                self.optimizer.zero_grad()

                old_score = get_inplace_reward(self.arr)
                
                q_predict, res = torch.max(self.dqn(torch.Tensor(self.arr)), 0)
                idx_1 = res // len(self.arr)
                idx_2 = res % len(self.arr)
                self.switch_elements(idx_1, idx_2)

                new_score = get_inplace_reward(self.arr)
                
                # Update dqn params with bellman
                with torch.no_grad():
                    q_prime = torch.max(self.dqn(torch.Tensor(self.arr)))
                reward = new_score - old_score
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
                actions = self.dqn(torch.Tensor(self.arr))
                _, res = torch.max(actions, 0)
                self.switch_elements(
                    res // len(self.arr),
                    res % len(self.arr),
                )

                print(actions)
                print(res // len(self.arr), res % len(self.arr))
                print(self.arr)
                print("---")
            

# === MAIN ===
if __name__ == "__main__":
    
    vis = visdom.Visdom()

    arr = list(range(10))
    agent = DQAgent(arr, is_train=True)

    n_epoch = 500
    n_iter = 10000
    update_rate = 100
    save_rate = 5
    save_path = './data'

    loss_log = []
    for i_epoch in range(n_epoch):
        
        # Reset agent
        agent.total_loss = 0
        random.shuffle(agent.arr)
        arr_log = []
        for i_iter in range(n_iter):

            agent.update()

            if (i_iter + 1) % update_rate == 0:
                arr_log.insert(0, f"<tr><td>{i_iter + 1}</td><td>{agent.arr}</td></tr>")
                vis.text(
                    '<table>'+''.join(arr_log)+'</table>',
                    win="Result",
                )

        # Update visdom and save params
        if agent.is_train and (i_epoch + 1) % save_rate == 0:
            loss_log.append(agent.total_loss / n_iter)
            vis.line(
                Y=np.array(loss_log),
                X=np.array([save_rate*x for x in range(1, len(loss_log)+1)]),
                opts=dict(
                    title='DQN Average Loss',
                     webgl=True,
                ),
                win='Losses',
            )
            
            torch.save(agent.dqn.state_dict(), os.path.join(save_path, f"dqn_{i_epoch + 1}"))
