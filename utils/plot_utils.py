import matplotlib.pyplot as plt
import numpy as np

def plot_returns(values: list, stds: list, xlabel: str, ylabel: str, legend_names: list, eval_freq: int):
    """
    Plot values with respect to timesteps
    
    :param values (np.ndarray): numpy array of values to plot as y values
    :param std (np.ndarray): numpy array of stds of y values to plot as shading
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    x_values = eval_freq + np.arange(len(values[0])) * eval_freq
    for i in range(len(values)):
        plt.plot(x_values, values[i], "-", alpha=0.7, label=f"{legend_names[i]}")
        plt.fill_between(
            x_values,
            values[i] - stds[i],
            values[i] + stds[i],
            alpha=0.2,
            antialiased=True,
        )
    plt.legend(loc="lower left")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.tight_layout(pad=0.3)