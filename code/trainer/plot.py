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
            



