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


class RandomSortingAgent(SortingAgent):
    """Sorts with a random policy"""
    def update(self):
        self.switch_elements(
            random.randint(0, len(self.arr)-1),
            random.randint(0, len(self.arr)-1),
        )
    

# === MAIN ===

agent = RandomSortingAgent(list(range(10)))

for i in range(10):
    print(agent.arr)
    agent.update()
