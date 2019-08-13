import torch
import random
from models import DQAgent

arr = list(range(10))
agent = DQAgent(arr, is_train=False)
agent.load_model('./data/dqn_latest.pt')

random.shuffle(arr)

while range(10000):
    print(agent.arr)
    if agent.arr == sorted(agent.arr):
        break
    agent.update()

print("Done!")
