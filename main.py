import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import visdom

from models import DQAgent

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
        agent.reset()
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
