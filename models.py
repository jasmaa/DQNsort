import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import os
import random
from abc import ABC, abstractmethod
from collections import namedtuple

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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """Replay memory from dqn tutorial"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        """Push transition into memory"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample minibatch from memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQAgent(SortingAgent):
    """Deep Q learning agent"""

    def __init__(self, arr, is_train=False, batch_size=32):
        super(DQAgent, self).__init__(arr)
        self.is_train = is_train
        self.dqn = DQN(len(arr))
        self.discount = 0.8
        self.loss_f = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-6)

        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(200)

        self.steps = 0
        self.total_loss = 0

    def reset(self):
        """Reset agent"""
        self.replay_memory = ReplayMemory(64)
        self.steps = 0
        self.total_loss = 0

    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.dqn.eval()

    def update(self):

        if self.is_train:
            # Exploit or explore
            if random.random() > 1 - self.steps/500:
                with torch.no_grad():
                    state = torch.Tensor(self.arr)
                    old_score = get_inplace_reward(self.arr)
                    
                    _, action = torch.max(self.dqn(torch.Tensor(self.arr)), 0)
                    idx_1 = action // len(self.arr)
                    idx_2 = action % len(self.arr)
                    self.switch_elements(idx_1, idx_2)

                    next_state = torch.Tensor(self.arr)
                    new_score = get_inplace_reward(self.arr)
                    reward = new_score - old_score
            else:
                state = torch.Tensor(self.arr)
                old_score = get_inplace_reward(self.arr)

                action = random.randint(0, len(self.arr)**2 - 1)
                idx_1 = action // len(self.arr)
                idx_2 = action % len(self.arr)
                self.switch_elements(idx_1, idx_2)

                next_state = torch.Tensor(self.arr)
                new_score = get_inplace_reward(self.arr)
                reward = new_score - old_score
            
            # Update memory and steps
            self.replay_memory.push(state, action, next_state, reward)
            self.steps += 1

            # Train DQN
            if len(self.replay_memory) >= self.batch_size:
                # Extract minibatch
                minibatch = self.replay_memory.sample(self.batch_size)
                state_batch = torch.stack([ t.state for t in minibatch ])
                action_batch = [t.action for t in minibatch]
                next_state_batch = torch.stack([ t.next_state for t in minibatch ])
                reward_batch = [t.reward for t in minibatch]

                # Zero optimizer
                self.optimizer.zero_grad()

                # Get q values from net and bellman
                q_pred = self.dqn(state_batch)
                with torch.no_grad():
                    q_target = self.dqn(next_state_batch)
                    for i in range(self.batch_size):
                        q_target[i][action_batch[i]] = reward_batch[i] + self.discount * torch.max(q_target[i])
                    
                # Calculate loss and optimize
                loss = self.loss_f(q_pred, q_target)
                loss.backward()
                self.optimizer.step()

                self.total_loss += loss.item()
            
        else:
            with torch.no_grad():
                actions = self.dqn(torch.Tensor(self.arr))
                _, action_idx = torch.max(actions, 0)
                self.switch_elements(
                    action_idx // len(self.arr),
                    action_idx % len(self.arr),
                )

                print(actions)
                print(action_idx // len(self.arr), action_idx % len(self.arr))
                print(self.arr)
                print("---")
