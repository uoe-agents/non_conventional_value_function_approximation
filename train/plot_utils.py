import matplotlib.pyplot as plt
import numpy as np

def plot_loss(losses_all,batch_size):
    print("Plotting DQN loss...")
    losses = np.array(losses_all)
    x_values = batch_size + np.arange(len(losses))
    plt.plot(x_values, losses, "-", alpha=0.7, label=f"DQN loss")
    plt.legend(loc="best")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.tight_layout(pad=0.3)
    plt.savefig(f"DQN_loss.pdf", format="pdf")
    # plt.show()