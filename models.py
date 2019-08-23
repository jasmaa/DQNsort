import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

import os
import random
from abc import ABC, abstractmethod
from collections import namedtuple

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    def init(self, arr):
        super(RandomAgent, self).__init__()
        self.arr = arr
    
    def update(self):
        self.switch_elements(
            random.randint(0, len(self.arr)-1),
            random.randint(0, len(self.arr)-1),
        )

# === Reward funcs ===

def get_local_score(arr, idx):
    """Check if considered sorted in local area"""
    score = 0
    score += -1 if idx - 1 >= 0 and not arr[idx] >= arr[idx-1] else 1
    score += -1 if idx + 1 < len(arr) and not arr[idx] <= arr[idx+1] else 1
    return score

def get_ascending_score(arr):
    """Reward based on number of ascending items"""
    score = 0
    for i in range(len(arr)-1):
        score += arr[i+1] - arr[i]
    return score

def get_inplace_score(arr):
    """Reward based on position of item"""
    score = 0
    for i, val in enumerate(arr):
        score += 1 if i == val else 0
    return score

def get_reward(prev_state, state):
    """Reward function"""

    prev_arr = prev_state.tolist()
    arr = state.tolist()

    # Bonus for completion
    modifier = 0
    if arr == sorted(arr):
        modifier = 10
   
    return get_inplace_score(arr) - get_inplace_score(prev_arr) + modifier

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
    """Replay memory"""

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

    def clear(self):
        self.memory.clear()
        self.position = 0

    def __len__(self):
        return len(self.memory)


class DQAgent(SortingAgent):
    """Deep Q learning agent"""

    def __init__(self, arr, discount=0.99, is_train=False, lr=1e-4, batch_size=32):
        super(DQAgent, self).__init__(arr)
        self.is_train = is_train
        self.dqn = DQN(len(arr))
        self.discount = discount
        self.loss_f = nn.MSELoss()
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(1000000)

        self.steps = 0
        self.total_loss = 0

    def reset(self):
        """Reset agent"""
        self.steps = 0
        self.total_loss = 0

    def load_model(self, path):
        self.dqn.load_state_dict(torch.load(path))
        self.dqn.eval()

    def update(self):

        # Training
        if self.is_train:
            
            state = torch.Tensor(self.arr)
            # Exploit or explore
            working_epsilon = 0.1 if self.steps > 1e6 else -self.steps*0.9/1e6 + 1
            if random.random() > 1 - working_epsilon:
                with torch.no_grad():
                    _, action = torch.max(self.dqn(torch.Tensor(self.arr)), 0)
                    idx_1 = action // len(self.arr)
                    idx_2 = action % len(self.arr)
            else:
                action = random.randint(0, len(self.arr)**2 - 1)
                idx_1 = action // len(self.arr)
                idx_2 = action % len(self.arr)
                
            self.switch_elements(idx_1, idx_2)
            next_state = torch.Tensor(self.arr)
            reward = get_reward(state, next_state)
            
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
                        # Do bellman
                        l = next_state_batch[i].tolist()
                        q_target[i][action_batch[i]] = reward_batch[i] + self.discount * torch.max(q_target[i])
                    
                # Calculate loss and optimize
                loss = self.loss_f(q_pred, q_target)
                loss.backward()
                self.optimizer.step()

                self.total_loss += loss.item()

        # Testing
        else:
            with torch.no_grad():
                q_values = self.dqn(torch.Tensor(self.arr))
                q_value, action = torch.max(q_values, 0)
                self.switch_elements(
                    action // len(self.arr),
                    action % len(self.arr),
                )
