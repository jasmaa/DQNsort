import torch
import random
from sorting import DQAgent

arr = list(range(10))
agent = DQAgent(arr, is_train=False)
agent.load_model('./data/dqn_500')

random.shuffle(arr)

while True:
    agent.update()
