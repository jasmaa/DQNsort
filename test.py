import torch
import random
from models import DQAgent, RandomAgent, BubbleSortAgent
import utils

def test():
    """Test DQNsort"""
    
    arr = list(range(5))
    random.shuffle(arr)

    agent = DQAgent(arr, is_train=False)
    agent.load_model('./data/dqn_latest.pt')

    while range(50):
        print(agent.arr)
        if agent.arr == sorted(agent.arr):
            break
        agent.update()

    print("Done!")

def test_compare(make_gif):
    """Test comparing DQN sort"""
    
    arr = list(range(5))
    random.shuffle(arr)

    rand_agent = RandomAgent(arr.copy())
    bubble_agent = BubbleSortAgent(arr.copy())
    dqn_agent = DQAgent(arr.copy(), is_train=False)
    dqn_agent.load_model('./data/dqn_latest.pt')

    agents = [
        rand_agent,
        bubble_agent,
        dqn_agent,
    ]
    agent_names = [
        "Random",
        "Bubble",
        "DQN",
    ]
    imgs = []

    is_done = False
    winner = -1
    while not is_done:

        print("Random :", rand_agent.arr)
        print("Bubble :", bubble_agent.arr)
        print("DQNsort:", dqn_agent.arr)
        print("===")
        if make_gif:
            imgs.append(utils.visualize_agents(agents, agent_names))
        
        for i, agent in enumerate(agents):
            agent.update()

            if agent.arr == sorted(agent.arr):
                is_done = True
                winner = i
                break
            
    print("Random :", rand_agent.arr)
    print("Bubble :", bubble_agent.arr)
    print("DQNsort:", dqn_agent.arr)
    print("===")
    if make_gif:
        end_img = utils.visualize_agents(agents, agent_names)
        imgs += [end_img] * 10

    print(agent_names[winner], "wins!")
    if make_gif:
        # Generate visuals
        utils.imgs2gif(imgs)
