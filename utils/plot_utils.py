import matplotlib.pyplot as plt
import numpy as np

def plot_returns(save_name: str, values: list, lower: list, upper: list, xlabel: str, ylabel: str, legend_names: list, eval_freq: int, markers: list, legend_outside=False):
    """
    Plot values with respect to timesteps
    
    :param values (np.ndarray): numpy array of values to plot as y values
    :param std (np.ndarray): numpy array of stds of y values to plot as shading
    :param xlabel (str): label of x-axis
    :param ylabel (str): label of y-axis
    :param legend_name (str): name of algorithm
    """
    x_values = eval_freq + np.arange(len(values[0])) * eval_freq
    plt.figure(figsize=(8,5))
    plt.rc('font', size=13)
    for i in range(len(values)):
        if markers is not None:
            plt.plot(x_values, values[i], "-", alpha=0.7, label=f"{legend_names[i]}", marker=markers[i])
        else: 
            plt.plot(x_values, values[i], "-", alpha=0.7, label=f"{legend_names[i]}")
        plt.fill_between(
            x_values,
            lower[i],
            upper[i],
            alpha=0.2,
            antialiased=True,
        )
    x_ticks = 2000 + np.arange(10) * 2000
    plt.xticks(x_ticks, rotation=30)
    plt.legend(loc="lower right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend_outside:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f"{save_name}.pdf", format="pdf", bbox_inches="tight")
