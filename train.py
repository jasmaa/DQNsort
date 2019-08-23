import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import visdom

from models import DQAgent

def train(use_visdom, n_epoch, n_iter, update_rate, save_rate, discount, lr, batch_size):
    """Trains DQNsort"""
    
    vis = visdom.Visdom() if use_visdom else None

    if not os.path.exists('./data'):
        os.mkdir('./data')
        print("Data directory created")
    else:
        print("Data directory already exists")

    arr = list(range(5))
    agent = DQAgent(arr, is_train=True)

    loss_log = []

    print("Start training...")
    for i_epoch in range(n_epoch):
        
        # Reset agent
        agent.reset()
        random.shuffle(agent.arr)
        arr_log = []
        for i_iter in range(n_iter):

            # Quick exit when sorted
            if agent.arr == sorted(agent.arr):
                break

            agent.update()

            if (i_iter + 1) % update_rate == 0:
                arr_log.insert(0, f"<tr><td>{i_iter + 1}</td><td>{agent.arr}</td></tr>")

                if use_visdom:
                    vis.text(
                        '<table>'+''.join(arr_log)+'</table>',
                        win="Result",
                    )
        avg_loss = agent.total_loss / (agent.steps + 1)
        print("Epoch: {0}/{1}\tDQN Loss: {2:.2f}".format(
            i_epoch + 1,
            n_epoch,
            avg_loss,
        ))

        # Update visdom and save params
        if agent.is_train and (i_epoch + 1) % save_rate == 0:
            loss_log.append(avg_loss)
            
            if use_visdom:
                vis.line(
                    Y=np.array(loss_log),
                    X=np.array([save_rate*x for x in range(1, len(loss_log)+1)]),
                    opts=dict(
                        title='DQN Average Loss',
                         webgl=True,
                    ),
                    win='Losses',
                )
            
            torch.save(agent.dqn.state_dict(), f"./data/dqn_{i_epoch + 1}.pt")
            torch.save(agent.dqn.state_dict(), "./data/dqn_latest.pt")

    print("Done!")
