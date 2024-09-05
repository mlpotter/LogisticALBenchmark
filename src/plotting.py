import jax
from click.core import batch

jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KernelDensity

import imageio
import os.path as osp
import os
from copy import deepcopy
from tqdm import tqdm


def plot_confidence_interval(x, y, ax, color="r", label="_nolegend_"):
    # some confidence interval
    ci = 1.96 * np.std(y, axis=0) / np.sqrt(y.shape[0])

    y_mu = np.mean(y, axis=0)
    ax.plot(x, y_mu, color=color, label=label)
    ax.fill_between(x, np.clip(y_mu - ci,0,1), np.clip(y_mu + ci,0,1), color=color, alpha=.4)


def plot_times(time_matrix,uncertainty_dict,args):
    #
    cmap = mpl.colormaps['jet']

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, len(uncertainty_dict)))

    fig, ax = plt.subplots()
    query_strategies = list(uncertainty_dict.keys())
    data_sizes = (np.arange(args.n_queries)) * args.query_size + args.n_initial
    for idx in np.arange(time_matrix.shape[0]):
        plot_confidence_interval(data_sizes, time_matrix[idx], ax, color=colors[idx],
                                 label=query_strategies[idx])


    ax.set_title(f"{args.dataset} times")
    ax.set_xlabel("# Datapoints")
    ax.set_ylabel("Time (s)")
    fig.legend()
    fig.tight_layout()

    return fig,ax

def plot_scores(score_matrix,baseline_score,uncertainty_dict,args):
    #
    cmap = mpl.colormaps['jet']

    # Take colors at regular intervals spanning the colormap.
    colors = cmap(np.linspace(0, 1, len(uncertainty_dict)))

    fig, ax = plt.subplots()
    query_strategies = list(uncertainty_dict.keys())
    data_sizes = (np.arange(args.n_queries+1)) * args.query_size + args.n_initial
    ax.plot(data_sizes,[baseline_score]*len(data_sizes),label="baseline",color="black",linestyle="--")
    for idx in np.arange(score_matrix.shape[0]):
        plot_confidence_interval(data_sizes, score_matrix[idx], ax, color=colors[idx],
                                 label=query_strategies[idx])

    ax.set_xlabel("# Datapoints")
    ax.set_ylabel("Accuracy")
    fig.legend(loc='lower right')
    fig.tight_layout()

    return fig,ax