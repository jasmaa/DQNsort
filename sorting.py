import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

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

    def update(self):
        
        if self.is_train:
            # Gamble if training
            if random.random() > 0.5:

                self.optimizer.zero_grad()
                
                q_predict, res = torch.max(self.dqn(torch.Tensor(self.arr)), 0)
                idx_1 = res // len(self.arr)
                idx_2 = res % len(self.arr)
                before_score = get_score(self.arr, idx_1) + get_score(self.arr, idx_2)
                self.switch_elements(idx_1, idx_2)
                after_score = get_score(self.arr, idx_1) + get_score(self.arr, idx_2)
                
                # Update dqn params with bellman
                with torch.no_grad():
                    q_prime = torch.max(self.dqn(torch.Tensor(self.arr)))
                reward = after_score - before_score
                q_target = reward + self.discount * q_prime

                loss = self.loss_f(q_predict, q_target)
                loss.backward()
                self.optimizer.step()
                
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
arr = list(range(20))
random.shuffle(arr)
agent = DQAgent(arr, is_train=True)

for i in range(9999):
    print(agent.arr)
    agent.update()
