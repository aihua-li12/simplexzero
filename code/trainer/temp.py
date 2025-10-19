# main.py

from data import *
from model import *
import torch

import os
os.getcwd()

from plot import *
from sklearn.preprocessing import StandardScaler
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap



def plot_simplex_3d(data, alpha=0.7):
    """Visualizes 3D data points on a probability simplex with a specific aesthetic.

    Args:
        data (np.ndarray): A NumPy array of shape (n_samples, 3) where each
                           row represents a point on the probability simplex
                           (i.e., elements are non-negative and sum to 1).
        alpha (float): The transparency level for the scattered data points.
    """
    if data.shape[1] != 3:
        raise ValueError("Input data must have 3 features (columns).")
    if not np.allclose(np.sum(data, axis=1), 1):
        print("Warning: Not all data points sum to 1. They may not lie on the simplex.")

    # If input points lie outside the simplex bounds,
    # it will be visually contained within the plot's 3D box.
    data = np.clip(data, 0, 1)

    # --- Set up the plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.tick_params(axis='both', which='major', labelsize=12)

    # --- Apply styling to panes and grid ---
    pane_color = (0.95, 0.95, 0.95, 0.4)
    ax.xaxis.set_pane_color(pane_color)
    ax.yaxis.set_pane_color(pane_color)
    ax.zaxis.set_pane_color(pane_color)
    ax.grid(True)

    # --- Draw the simplex triangle surface and edges ---
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles=[[0, 1, 2]])
    ax.plot_trisurf(tri, vertices[:, 2], color='gray', alpha=0.15, shade=False)
    ax.plot([1,0,0,1], [0,1,0,0], [0,0,1,0], 'k-', linewidth=1.5)

    # --- Custom colormap definition ---
    my_colors = ["mediumblue", "cornflowerblue", "lightskyblue",
                 "yellow", "gold", "gold",
                 "lightsalmon", "orangered", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", my_colors)

    # --- Calculate colors and draw scatter plot ---
    color_values = (data[:, 1] * 0.5) + (data[:, 2] * 1.0)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=15, alpha=alpha, c=color_values, cmap=custom_cmap)

    # --- Set viewing angle, labels, title, and limits ---
    ax.view_init(elev=30., azim=45)
    ax.set_title('3D Probability Simplex Visualization', fontsize=20)
    ax.set_xlabel('Dim 1', fontsize=15)
    ax.set_ylabel('Dim 2', fontsize=15)
    ax.set_zlabel('Dim 3', fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # Show the plot
    plt.show()


dir = Dirichlet(dirich_param=torch.Tensor([1,1,1]))
plot_simplex_3d(dir.sample(500).numpy())




a = torch.tensor([2.0, 3.0, 4.0])
b = torch.tensor([3.0, 4.0, 5.0])

logB = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
B = torch.exp(logB)


a = torch.tensor(2.5)
b = torch.tensor(3.0)
logB = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
B = torch.exp(logB)
