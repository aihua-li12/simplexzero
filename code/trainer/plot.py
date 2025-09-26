# plot.py

# ================================================================================ #
#   This script contains plotting function for evaluating generated samples        #
# ================================================================================ #


from data import *
from model import *
import torch

import os

import matplotlib.pyplot as plt
import skbio
from skbio.stats.ordination import pcoa
from skbio.diversity.alpha import shannon
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.tri as mtri
from typing import cast, Iterable
from matplotlib.artist import Artist
from matplotlib.colors import LinearSegmentedColormap, Normalize


class Plot():
    """
    Args:
        x_obs: observed samples, (n_samples, dim)
        x_recon: reconstructed samples, (n_samples, dim)
        plot_save_dir: directory to save the plot
    """
    def __init__(self, x_obs:torch.Tensor, x_recon:torch.Tensor,
                 plot_save_dir:str|None=None, style:str="tableau-colorblind10"):
        self.x_obs = x_obs 
        self.x_recon = x_recon 
        self.plot_save_dir = plot_save_dir
        self.style = style
        if self.plot_save_dir is not None:
            os.makedirs(self.plot_save_dir, exist_ok=True)
        
        self.x_list = [x_obs, x_recon]
        self.title_list = ["observed", "reconstructed"]

    
    def _save(self, plot_name:str) -> None:
        """Save the plot as .pdf"""
        if self.plot_save_dir is not None:
            plot_path = os.path.join(self.plot_save_dir, plot_name+".pdf")
            plt.savefig(plot_path, dpi=300)


    def histogram(self, n_hist:int=4, plot_name:str="histogram"):
        """Histogram of relative abundance of each sample
        Args:
            n_hist: number of histograms in one row
        """
        with plt.style.context(self.style): 
            fig, axes = plt.subplots(2, n_hist, figsize=(10, 4))
            fig.suptitle('Relative Abundance of each sample', fontsize=12)
            for i, ax in enumerate(axes[0]):
                ax.hist(self.x_obs[i], bins=30, color='tab:blue', alpha=0.7)
                ax.set_title(f'Observed sample {i+1}', fontsize=10)
                ax.grid(True)
            for i, ax in enumerate(axes[1]):
                ax.hist(self.x_recon[i], bins=30, color='tab:orange', alpha=0.7)
                ax.set_title(f'Reconstructed sample {i+1}', fontsize=10)
                ax.grid(True)
            plt.tight_layout()
            self._save(plot_name)
            plt.show()

    def mean_variance(self, plot_name:str="mean-variance"):
        """Mean-variance plot"""
        with plt.style.context(self.style): 
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            fig.suptitle('Mean-variance plot', fontsize=12)
            for i, ax in enumerate(axes):
                epsilon = 0
                feature_mean = self.x_list[i].mean(dim=0) + epsilon # (dim,)
                feature_var = self.x_list[i].var(dim=0) + epsilon
                
                ax.scatter(feature_mean, feature_var, alpha=0.4, edgecolors='k', s=7)
                x_line = np.linspace(min(feature_mean.min(), feature_var.min()),
                                     max(feature_mean.max(), feature_var.max()), 100)
                ax.plot(x_line, x_line, color='red', alpha=0.6, 
                        linestyle='--', label='(Variance = Mean)')
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(f'Mean-Variance Plot ({self.title_list[i]})', fontsize=10)
                ax.set_xlabel('Mean (log scale)', fontsize=8)
                ax.set_ylabel('Variance (log scale)', fontsize=8)
                ax.grid(True, which="both", ls="--", linewidth=0.3)

            plt.tight_layout()
            self._save(plot_name)
            plt.show()

    def sparsity(self, plot_name:str="sparsity"):
        with plt.style.context(self.style): 
            plt.figure(figsize=(5, 3))
            plt.title('Sparsity (number of zeros)', fontsize=12)
            n_zeros_obs = (self.x_list[0]==0).sum(-1)
            n_zeros_recon = (self.x_list[1]==0).sum(-1)
            plt.hist(n_zeros_obs, bins=20, alpha=0.5, label = "Observed",
                     color='tab:blue', edgecolor = "k")
            plt.hist(n_zeros_recon, bins=20, alpha=0.5, label = "Reconstructed",
                     color='tab:orange', edgecolor = "k")
            plt.grid(True, which="both", ls="--", linewidth=0.3)
            
            plt.legend(loc='upper left', fontsize = 8)
            plt.tight_layout()
            self._save(plot_name)
            plt.show()
            






class PlotSimplex():
    """Creates and saves a GIF animation of samples on simplexes.
    Args:
        data: samples across time, (time, samples, dims)
        data_prior (optional): if provided, plot prior data samples on the left most subplot, (samples, dim)
        data_true (optional): if provided, plot true data samples on the right most subplot, (samples, dim)
        prior_tlt (optional): title of the subplot of prior data samples
        true_tlt (optional): title of the subplot of true/learned data samples
        plot_save_dir: the directory to save the .gif
        plot_save_name: the name to save the .gif
        pause_seconds: seconds to pause at the last frame
        fps: frame per second
    """
    def __init__(self, data:np.ndarray, 
                 data_prior:np.ndarray|None=None,
                 data_true:np.ndarray|None=None,
                 prior_tlt:str="Prior distribution",
                 true_tlt:str="True distribution",
                 title:str="Probability Path Over Simplex",
                 plot_save_dir:str='../../result/static',
                 plot_save_name:str='simplex',
                 pause_seconds:float=1.5, 
                 fps:int=5):
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir, exist_ok=True)
        
        self.n_times = data.shape[0]
        self.fps = fps
        self.pause_seconds = pause_seconds
        self.data = data

        # --- Set plot ---
        # one row of three subplots
        if data_prior is not None and data_true is not None:
            self.fig, (self.ax1, self.animation_ax, self.ax3) = plt.subplots(
                1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'}
            )
            self.ax1.tick_params(axis='both', which='major', labelsize=12)
            self.ax3.tick_params(axis='both', which='major', labelsize=12)
        # one single simplex
        else:
            self.fig = plt.figure(figsize=(10, 8))
            self.animation_ax = cast(Axes3D, self.fig.add_subplot(111, projection='3d'))
        

        self.fig.suptitle(title, fontsize=30)
        self.animation_ax.tick_params(axis='both', which='major', labelsize=12)

        # --- Custom colormap ---
        self.my_colors = ["mediumblue", "cornflowerblue", "lightskyblue", 
                          "yellow", "gold", "gold", 
                          "lightsalmon", "orangered", "red"]
        self.custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", self.my_colors)
        self.alphas = np.linspace(0.7, 0.3, self.n_times)

        # --- Create a custom frame sequence to add a pause at the end ---
        self.num_pause_frames = int(self.pause_seconds * self.fps)
        self.last_frame_index = self.n_times - 1
        self.frame_sequence = list(range(self.n_times)) + [self.last_frame_index] * self.num_pause_frames
        self.total_render_frames = len(self.frame_sequence)

        # Class to track progress for cleaner console output
        class ProgressTracker:
            def __init__(self, total): self.current = 0; self.total = total
            def step(self): self.current += 1; return self.current
        
        self.progress = ProgressTracker(self.total_render_frames)
    
        if data_prior is not None and data_true is not None:
            # Plot the prior samples on the left axis
            self.plot_single_simplex(self.ax1, data_prior, prior_tlt, self.alphas[0])
            # Plot the true samples on the right axis
            self.plot_single_simplex(self.ax3, data_true, true_tlt, self.alphas[-1])
        
        ani = FuncAnimation(self.fig, self.update, frames=self.frame_sequence, interval=100)

        output_file = os.path.join(plot_save_dir, plot_save_name+'.gif')
        writer = PillowWriter(fps=self.fps)
        ani.save(output_file, writer=writer)
        
        print(f"\nAnimation saved successfully as '{output_file}'")
        plt.close(self.fig)

    def plot_single_simplex(self, ax:Axes3D, points:np.ndarray, 
                            title:str, alpha:float) -> Artist:
        """Draw a styled simplex with scatter points on a given axis
        Args:
            ax: plot axes
            points: data points, (samples, dims)
            title: title of the plot
            alpha: transparency level
        """
        # Clear the axis for redrawing
        ax.cla()

        # If input points lie outside the simplex bounds,
        # it will be visually contained within the plot's 3D box.
        points = np.clip(points, 0, 1)

        # --- Re-apply styling that gets cleared ---
        pane_color = (0.95, 0.95, 0.95, 0.4)
        ax.xaxis.set_pane_color(pane_color) # type: ignore
        ax.yaxis.set_pane_color(pane_color) # type: ignore
        ax.zaxis.set_pane_color(pane_color) # type: ignore
        ax.grid(True)

        vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        tri = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles=[[0, 1, 2]])
        ax.plot_trisurf(tri, vertices[:, 2], color='gray', alpha=0.15, shade=False)
        ax.plot([1,0,0,1], [0,1,0,0], [0,0,1,0], 'k-', linewidth=1.5)

        # --- Set consistent viewing angle and labels ---
        ax.view_init(elev=30., azim=45)
        ax.set_title(title, fontsize=20)
        ax.set_xlabel('Dim 1', fontsize=15)
        ax.set_ylabel('Dim 2', fontsize=15)
        ax.set_zlabel('Dim 3', fontsize=15)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # --- Calculate colors and draw scatter plot ---
        color_values = (points[:, 1] * 0.5) + (points[:, 2] * 1.0)
        scatter_plot = ax.scatter(*points.T, s=15, alpha=alpha, c=color_values, cmap=self.custom_cmap)
        
        return scatter_plot


    def update(self, frame_index: int) -> Iterable[Artist]:
        current_step = self.progress.step()
        scatter_artist = self.plot_single_simplex(
            self.animation_ax, self.data[frame_index], f"Time step: {frame_index+1}", self.alphas[frame_index]
        )
        print(f"Rendering frame {current_step}/{self.total_render_frames}"+
              f"(Data from step {frame_index + 1})...")
        return (scatter_artist,)
    

