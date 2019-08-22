import torch
import random
from models import DQAgent

arr = list(range(5))
random.shuffle(arr)

agent = DQAgent(arr, is_train=False)
agent.load_model('./data/dqn_latest.pt')

while range(10000):
    print(agent.arr)
    if agent.arr == sorted(agent.arr):
        break
    agent.update()

print("Done!")
