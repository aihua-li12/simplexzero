# plot.py

# ================================================================================ #
#   This script contains plotting function for evaluating generated samples        #
# ================================================================================ #


from data import *
from model import *
import torch
from typing import Any, Dict

import os

import matplotlib.pyplot as plt
import skbio
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from skbio.diversity.alpha import shannon
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.tri as mtri
from typing import cast, Iterable
from matplotlib.artist import Artist
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon
from skbio.diversity import alpha_diversity



class Plot():
    """
    Args:
        samples: dictionary containing the observed samples and reconstructed samples
        x_obs: observed samples, (n_samples, dim)
        x_recon: reconstructed samples, (n_samples, dim)
        x_recon2 (optional): reconstructed samples (by other transformation)
        plot_save_dir: directory to save the plot
    """
    def __init__(self, 
                 samples_dict:Dict[str,torch.Tensor]|None=None,
                 plot_save_dir:str|None=None, 
                 style:str="tableau-colorblind10"):
        plt.style.use(style)
        # Set font properties 
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"], # Explicitly prefer Times New Roman
            "mathtext.fontset": "stix", # Math font
        })
        self.plot_save_dir = plot_save_dir

        palette = sns.color_palette("Paired")
        self.colors = [palette[7], palette[1], palette[0], palette[5], palette[3]]


        if isinstance(samples_dict, dict):
            self.x_list = list(samples_dict.values())
            self.title_list = list(samples_dict.keys())
            

    def _save(self, plot_name:str) -> None:
        """Save the plot as .png"""
        if self.plot_save_dir is not None:
            os.makedirs(self.plot_save_dir, exist_ok=True)
            plot_path = os.path.join(self.plot_save_dir, plot_name+".png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')


    def histogram(self, n_hist:int=4, plot_name:str="histogram"):
        """Histogram of relative abundance of each sample
        Args:
            n_hist: number of histograms in one row
        """
        # assert self.x_obs is not None
        # assert self.x_recon is not None
        # fig, axes = plt.subplots(2, n_hist, figsize=(10, 4))
        # fig.suptitle('Relative Abundance of each sample', fontsize=12)
        # for i, ax in enumerate(axes[0]):
        #     ax.hist(self.x_obs[i], bins=30, color='tab:blue', alpha=0.7)
        #     ax.set_title(f'Observed sample {i+1}', fontsize=10)
        #     ax.grid(True)
        # for i, ax in enumerate(axes[1]):
        #     ax.hist(self.x_recon[i], bins=30, color='tab:orange', alpha=0.7)
        #     ax.set_title(f'Reconstructed sample {i+1}', fontsize=10)
        #     ax.grid(True)
        # plt.tight_layout()
        # self._save(plot_name)
        # plt.show()

    def mean_variance(self, plot_name:str="mean-variance"):
        """Mean-variance plot"""
        fig, axes = plt.subplots(1, len(self.x_list), figsize=(4*len(self.x_list), 3.5))
        # fig.suptitle('Mean-variance plot', fontsize=12)
        for i, ax in enumerate(axes):
            epsilon = 0
            feature_mean = self.x_list[i].mean(dim=0) + epsilon # (dim,)
            feature_var = self.x_list[i].var(dim=0) + epsilon
            
            ax.scatter(feature_mean, feature_var, alpha=1, s=20, color=self.colors[i],
                       edgecolors='k', linewidth=0.5)
            x_line = np.linspace(min(feature_mean.min(), feature_var.min()),
                                    max(feature_mean.max(), feature_var.max()), 100)
            ax.plot(x_line, x_line, color='red', alpha=0.6, 
                    linestyle='--', label='(Variance = Mean)')
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(self.title_list[i], fontsize=14)
            ax.set_xlabel('Mean (log scale)', fontsize=14)
            ax.set_ylabel('Variance (log scale)', fontsize=14)
            ax.grid(True, which="major", ls="--", linewidth=0.3)

        plt.tight_layout()
        self._save(plot_name)
        # plt.show()
        plt.close()


    def beta_diversity(self, plot_name:str="pcoa"):
        fig, axes = plt.subplots(1, len(self.x_list), figsize=(4*len(self.x_list), 3.5))
        for i, ax in enumerate(axes):
            x = pd.DataFrame(self.x_list[i].numpy())
            x = x.loc[x.sum(axis=1)>0] # remove empty row

            bc_dist = beta_diversity("braycurtis", x)
            pcoa_object = pcoa(bc_dist, dimensions=2, seed=42)
            pcoa_df = pd.DataFrame(pcoa_object.samples)

            if np.mean(pcoa_df.values) < 0:
                pcoa_df *= -1 # fix the sign

            var_exp = np.array(pcoa_object.proportion_explained)
            var_exp1 = f'Variation explained {var_exp[0]*100:.1f}%'
            var_exp2 = f'Variation explained {var_exp[1]*100:.1f}%'

            sns.scatterplot(
                pcoa_df, x='PC1', y='PC2', ax=ax,
                color=self.colors[i], alpha=1, s=20,
                edgecolor='k', linewidth=0.5
            )

            ax.grid(True, which="major", ls="--", linewidth=0.3)
            ax.set_title(self.title_list[i], fontsize=14)
            ax.set_xlabel(f'PC1 - {var_exp1}', fontsize=14)
            ax.set_ylabel(f'PC2 - {var_exp2}', fontsize=14)
        plt.tight_layout()
        self._save(plot_name)
        # plt.show()
        plt.close()

    def _prepare_stackedbar_df(self, 
                               abundance:pd.DataFrame, 
                               x:torch.Tensor,
                               top_taxa_list:list,
                               n_subset:int=1000,
                               seed:int=123) -> pl.dataframe.frame.DataFrame:
        """Prepare the dataframe for making the stacked bar plot.
        Args:
            abundance: DataFrame from `preprocess()`.
            x: Element of `self.x_list` (e.g., x_obs, x_recon)
            top_taxa_list: The list of taxa to be plotted
            n_subset: Randomly subset the data and take n_subset samples. The full data is too huge to be plotted.
            seed: The random seed to randomly subset the data.
        Returns:
            Returns a polar dataframe where the first column is the sample_id, 
            and the rest of the columns are the relative abundance of different taxa in this taxonomic level.
            Sample ids are sorted according to their relative abundance in the most abundant taxa.
        """
        # Create polars dataframe
        tax_level = str(abundance.index.name)
        df = pd.DataFrame(
            x.numpy().T, 
            index=abundance.index, 
            columns=['sample'+str(i) for i in range(x.shape[0])]
        )
        df = pl.from_dataframe(df)

        # Pivot the data longer and compute the relative abundance for each sample
        df_rel_abun = df.unpivot(index=tax_level, variable_name='sample_id', value_name='relative_abundance')

        # Group taxa into top list or 'Other'
        df_processed = df_rel_abun.with_columns(
            pl.when(pl.col(tax_level).is_in(top_taxa_list))
            .then(pl.col(tax_level))
            .otherwise(pl.lit('Other'))
            .alias(tax_level)
        ).group_by(['sample_id', tax_level]).agg(
            pl.sum('relative_abundance')
        )

        # Pivot wider
        df_plot = df_processed.pivot(index='sample_id', on=tax_level, values='relative_abundance').fill_null(0)

        # Sort columns to have a consistent plotting order
        cols = [taxon for taxon in top_taxa_list if taxon in df_plot.columns]
        if 'Other' in df_plot.columns:
            df_plot = df_plot.select(['sample_id'] + cols + ['Other'])
        else:
            df_plot = df_plot.select(['sample_id'] + cols)


        # Randomly subset the data for plot
        df_plot = df_plot.sample(n=n_subset, seed=seed)
        
        # Sort the sample ids according to their relative abundance in the most abundant taxa
        # taxa_cols = df_plot.columns[1:]
        # tax_cols_sums = df_plot[:,1:].to_numpy().sum(axis=0)
        # sort_by = taxa_cols[np.argmax(tax_cols_sums)] # most abundant taxa
        sort_by = top_taxa_list[2]
        df_plot_sorted = df_plot.sort(sort_by, descending=True)
        
        return df_plot_sorted
    
    def stacked_bar(self, abundance:pd.DataFrame, n_taxa:int=5,
                    n_subset:int=1000, seed:int=123,
                    plot_name:str='stacked-bar'):
        
        # --- Step 1: Determine top taxa ONCE from the first dataset ---
        tax_level = str(abundance.index.name)
        x_first = self.x_list[0]
        df_first = pd.DataFrame(
            x_first.numpy().T, 
            index=abundance.index, 
            columns=['sample'+str(i) for i in range(x_first.shape[0])]
        )
        df_first_pl = pl.from_dataframe(df_first)
        df_first_means = df_first_pl.unpivot(index=tax_level, variable_name='sample_id', value_name='abundance') \
                                    .group_by(tax_level).agg(pl.mean('abundance').alias('mean_abundance'))
        
        top_taxa_to_plot = df_first_means.sort('mean_abundance', descending=True) \
                                        .head(n_taxa)[tax_level].to_list()
        top_taxa_to_plot.sort()
        all_taxa_for_legend = top_taxa_to_plot + ['Other']

        # --- Step 2: Create a consistent color map ---
        colors = plt.colormaps["RdYlBu"](np.linspace(0, 1, len(all_taxa_for_legend)))
        color_map = {taxon: color for taxon, color in zip(all_taxa_for_legend, colors)}

        fig, axes = plt.subplots(2, 2, figsize=(10*2, 3.5*2), sharey=True)
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            df_plot_sorted = self._prepare_stackedbar_df(
                abundance, self.x_list[i], top_taxa_to_plot, n_subset, seed
            )
            taxa_cols = df_plot_sorted.columns[1:]
            x_pos = np.arange(len(df_plot_sorted))
            bottom = np.zeros(len(df_plot_sorted), dtype=np.float32)

            for taxon in taxa_cols:
                values = df_plot_sorted[taxon].to_numpy()
                ax.bar(x_pos, values, bottom=bottom, label=taxon, width=1.0, color=color_map.get(taxon))
                bottom += values
            
            tick_interval = 200
            tick_locs = np.arange(0, len(df_plot_sorted)+1, tick_interval)
            ax.set_xticks(tick_locs)
            ax.set_xticklabels([str(loc) for loc in tick_locs])

            ax.set_xlabel('Sample Index', fontsize=18)
            ax.set_title(self.title_list[i], fontsize=18)
        
        # Set y-label only for the first subplot
        axes[0].set_ylabel('Proportion', fontsize=18)
        axes[2].set_ylabel('Proportion', fontsize=18)
        axes[0].set_ylim(0, 100)
        
        # --- Step 3: Create a single, shared legend for the entire figure ---
        handles, labels = axes[0].get_legend_handles_labels()
        # Use a dictionary to ensure legend order is correct and has no duplicates
        by_label = dict(zip(labels, handles))
        # Sort legend items based on the pre-defined order
        sorted_labels = [lbl for lbl in all_taxa_for_legend if lbl in by_label]
        sorted_handles = [by_label[lbl] for lbl in sorted_labels]
        
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        sorted_labels = [lbl for lbl in all_taxa_for_legend if lbl in by_label]
        sorted_handles = [by_label[lbl] for lbl in sorted_labels]

        legend = fig.legend(
            sorted_handles, sorted_labels,
            title="Taxon",
            loc='upper right',
            bbox_to_anchor=(1.05, 0.96),  # move slightly outside top-left
            fontsize=16,
        )
        legend.get_title().set_fontsize(18)

        plt.tight_layout(rect=(0, 0, 0.9, 1))
        
        self._save(plot_name)
        # plt.show()
        plt.close()


    def sparsity(self, plot_name:str="sparsity"):
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

    def sza(self, plot_name:str="sza"):
        def f(z1, z2):
            """Calculates the function f(z)_1 = max(z1, 0) / (max(z1, 0) + max(z2, 0)).
            Handles the case where the denominator is zero.
            """
            relu_z1 = np.maximum(z1, 0)
            relu_z2 = np.maximum(z2, 0)
            denominator = relu_z1 + relu_z2 + 1e-2

            # Avoid division by zero. Where the denominator is zero, the result is undefined.
            # We will return 0 in this case for plotting purposes.
            # A small epsilon is added to avoid issues with floating point precision.
            result = np.divide(relu_z1, denominator, 
                               out=np.zeros_like(relu_z1, dtype=float), 
                               where=denominator>1e-9)
            return result
        
        z1_values = np.linspace(-5, 10, 400)
        z2_fixed_values = [0, 0.5, 2, 5]

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
        linestyles = ['-', '--', ':', '-.']
        linewidths = [0.85, 1, 1, 1]
        _, ax = plt.subplots(figsize=(5, 2.5))
        for z2_val, color, style, w in zip(z2_fixed_values, colors, linestyles, linewidths):
            f_values = f(z1_values, z2_val)
            ax.plot(z1_values, f_values, label=f'$z_2 = {z2_val}$', 
                    linewidth=w, color=color, linestyle=style, alpha=1)
        ax.axhline(y=1, color='tab:gray', linestyle='--', linewidth=0.9, alpha=0.5)
        # ax.set_title('Simple-Zero Activation Function', fontsize=12)
        ax.set_xlabel('$z_1$', fontsize=12)
        ax.set_ylabel('$\\sigma_{\\text{sza}, 1}(\\boldsymbol{z})$', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        ax.grid(True, alpha=0.1)

        ax.set_ylim(-0.1, 1.1)

        plt.tight_layout(rect=(0, 0, 1, 1))
        self._save(plot_name)
        plt.show()
        



class PlotTree(Plot):
    """Plot the tree on a disk
    Args:
        tree: tree produced by DepthFirstSearch
    """

    def __init__(self, tree:dict, plot_save_dir:str|None="../../result"):
        super().__init__(plot_save_dir=plot_save_dir)
        self.tree = tree
        self.taxa_levels = ["Domain", "Phylum", "Class", "Order", "Family", "Genus"]


    def plot_matplotlib_radial(self, plot_name:str="tree"):
        style = "tableau-colorblind10"
        plt.style.use(style)
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        })

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect('equal')
        ax.axis('off')
        class_level_index = self.taxa_levels.index('Class')
        unique_names_to_color = self._get_unique_taxa_names_up_to_level(
            self.tree, class_level_index
        )

        my_colors = ["slategray", "tab:blue", 
                     "cornflowerblue", "lightskyblue",
                     "yellowgreen", 
                     "gold", "orange", 
                     "orangered", "salmon",
                     "mediumpurple", "plum"]
        colormap = LinearSegmentedColormap.from_list("my_custom_cmap", my_colors)
        colors = colormap(np.linspace(0, 1, len(unique_names_to_color)))
        self.color_map_dict = {
            name: color for name, color in zip(sorted(list(unique_names_to_color)), colors)
        }

        radii = {}
        radii['Domain'] = 0.9
        radii['Phylum'] = radii['Domain'] + 4
        radii['Class'] = radii['Phylum'] + 3
        radii['Order'] = radii['Class'] + 1.5
        radii['Family'] = radii['Order'] + 2
        radii['Genus'] = radii['Family'] + 3.5

        node_sizes = {
            "Domain": 300, "Phylum": 200, "Class": 100,
            "Order": 80, "Family": 50, "Genus": 50
        }
        text_sizes = {
            "Domain": 17, "Phylum": 12, "Family": 16, "Genus": 16,
        }

        self._draw_node_recursively(
            ax, self.tree, (0, 0), self.taxa_levels, radii,
            node_sizes, text_sizes, 0, 2*np.pi
        )
        fig.subplots_adjust(top=0.85, bottom=0.15)
        self._save(plot_name)
        print(f"✅ Final radial tree saved to '{plot_name}'")
        plt.show()

    def _get_unique_taxa_names_up_to_level(
            self, node: dict, target_level: int, current_level: int = 0
    ) -> set:
        if current_level > target_level: return set()
        names = set()
        for name, child_node in node.items():
            if name != '_asv_ids':
                names.add(name)
                names.update(
                    self._get_unique_taxa_names_up_to_level(
                        child_node, target_level, current_level+1
                    )
                )
        return names

    def _count_terminal_nodes(self, node: dict) -> int:
        children = {k: v for k, v in node.items() if k != '_asv_ids'}
        if not children:
            return 1
        count = 0
        for child_node in children.values():
            count += self._count_terminal_nodes(child_node)
        return count

    def _draw_node_recursively(
        self, ax, node: dict, parent_pos: tuple,
        levels: list, radii: dict, node_sizes: dict, text_sizes: dict,
        start_angle: float, end_angle: float, level_idx: int = 0,
        branch_color: str = 'gray'
    ):
        if level_idx >= len(levels): return

        current_level_name = levels[level_idx]
        radius = radii[current_level_name]

        children = {k: v for k, v in node.items() if k != '_asv_ids'}
        if not children: return

        total_terminals = self._count_terminal_nodes(node)
        if total_terminals == 0: return

        current_angle = start_angle
        class_level_index = levels.index('Class')

        for name, child_node in sorted(children.items()):
            child_terminals = self._count_terminal_nodes(child_node)
            angle_slice = (end_angle - start_angle) * (child_terminals / total_terminals)
            angle = current_angle + angle_slice / 2
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            node_color = branch_color
            if level_idx <= class_level_index:
                node_color = self.color_map_dict.get(name, branch_color)

            ax.plot([parent_pos[0], x], [parent_pos[1], y],
                    color=node_color, alpha=0.7, lw=0.8, zorder=1)

            if current_level_name in ["Domain", "Phylum"]:
                # For 'Domain' and 'Phylum', draw a rounded rectangle with text inside
                font_size = text_sizes.get(current_level_name, 18)
                ax.text(x, y, name, ha='center', va='center',
                        fontsize=font_size, color='white', weight='bold',
                        zorder=level_idx+5,
                        path_effects=[pe.withStroke(linewidth=1, foreground='black')],
                        bbox=dict(
                            boxstyle='round,pad=0.3',
                            fc=node_color,
                            ec='black',
                            lw=0.8
                        ))
            else:
                # For all other levels, draw the standard scatter plot circle
                ax.scatter(x, y, s=node_sizes.get(current_level_name, 20),
                           c=[node_color], zorder=level_idx+5, ec='black', lw=0.8)

            grand_children = {k: v for k, v in child_node.items() if k != '_asv_ids'}
            is_end_node = not grand_children

            if is_end_node:
                text_offset = 1.15
                text_x = x * text_offset
                text_y = y * text_offset
                rotation = np.rad2deg(angle)
                ha = 'left'
                font_size = text_sizes.get(current_level_name, 7)
                if 90 < rotation < 270:
                    rotation -= 180
                    ha = 'right'
                ax.text(text_x, text_y, name, ha=ha, va='center',
                        fontsize=font_size, rotation=rotation,
                        rotation_mode='anchor', color=node_color)
            
            self._draw_node_recursively(
                ax, child_node, (x, y), levels, radii,
                node_sizes, text_sizes, current_angle, current_angle+angle_slice,
                level_idx+1, branch_color=node_color
            )
            current_angle += angle_slice



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
    

class PlotCardiovascular(Plot):
    """PCoA and PCA plots for cardiovascular disease.
    Note: this function should be used after neccesary 
    data amplification and processing and subsetting.
    Args:
        meta: (n_samples,), where index are sample ids and values are disease types
        plot_save_dir: directory to save the plot
    """
    def __init__(self, meta:pd.Series, plot_save_dir:str|None=None,
                 style:str="tableau-colorblind10"):
        plt.style.use(style)
        # Set font properties 
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"], # Explicitly prefer Times New Roman
            "mathtext.fontset": "stix", # Math font
        })
        self.plot_save_dir = plot_save_dir
        
        meta.index.name = "#SampleID"
        self.meta = meta
        self.cat_var = str(meta.name)
    
    def pcoa_plot(self, data_genus:pd.DataFrame, plot_name:str="cardio_pcoa"):
        """PCoA plot.
        Args:
            data_genus: (n_samples, n_features), where index are sample ids 
                        and features have been filtered to relevant taxa.
        """
        data_genus.index.name = "#SampleID"
        
        # remove empty row in data_genus
        non_empty_samples_mask = data_genus.sum(axis=1) > 0
        data_genus = data_genus.loc[non_empty_samples_mask]
        
        # select metadata that corresponds to data_genus
        meta_genus = self.meta.loc[data_genus.index].rename(self.cat_var)
        meta_genus.index.name = "#SampleID"
        meta_genus.name = self.cat_var

        # beta diversity
        bc_dist = beta_diversity("braycurtis", data_genus, data_genus.index)
        pcoa_object = pcoa(bc_dist, dimensions=2, seed=42)
        pcoa_df = pd.DataFrame(pcoa_object.samples)
        pcoa_df.index.name = "#SampleID"
        pcoa_df = pcoa_df.join(meta_genus, on="#SampleID")

        var_exp = np.array(pcoa_object.proportion_explained)
        var_exp1 = f'Variation explained {var_exp[0]*100:.1f}%'
        var_exp2 = f'Variation explained {var_exp[1]*100:.1f}%'


        group_names = pcoa_df[self.cat_var].unique()
        palette = sns.color_palette("Paired")
        colors = [palette[0], palette[1], palette[9], palette[5], palette[3]]
        color_rectangle = 'red'

        fig, axes = plt.subplots(1, len(group_names), figsize=(8, 4), sharey=True)

        for i in range(len(group_names)):
            group_data = pcoa_df[pcoa_df[self.cat_var]==group_names[i]]
            ax = axes[i]
            sns.scatterplot(
                group_data, x='PC1', y='PC2', ax=axes[i],
                color=colors[i], alpha=1, s=30, 
                edgecolor='k', linewidth=0.5
            )
            ax.set_title(group_names[i])
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_xlim(-0.55, 0.55)
            # Add rectangle
            ax.add_patch(Rectangle(
                xy=(-0.02, -0.05),        # Bottom-left corner (x, y)
                width=0.55,             # Width of the rectangle
                height=0.42,             # Height of the rectangle
                facecolor='whitesmoke',
                alpha=0.8,              # Transparency of the fill
                edgecolor=color_rectangle,
                linestyle='--',
                linewidth=2,
                zorder=0                # Set zorder to 0 to draw it behind the points
            ))
        
        fig.suptitle('PCoA of Taxa by Cardiovascular Disease', fontsize=14)
        fig.supxlabel(f'PC1 - {var_exp1}', fontsize=12)
        fig.supylabel(f'PC2 - {var_exp2}', fontsize=12)
        plt.tight_layout()
        self._save(plot_name)
        plt.show()

    def pca_plot(self, latent_rep:torch.Tensor, plot_name:str='cardio_pca'):
        """PCA plot of the latent representations generated by VAE
        Args:
            latent_rep: (n_samples, n_features), latent representation
        """
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(latent_rep.detach().numpy())
        pca_df = pd.DataFrame(pcs, columns=['pc1', 'pc2'])
        pca_df.index.name = "#SampleID"
        pca_df.index = self.meta.index
        pca_df = pca_df.join(self.meta, on="#SampleID")

        pca.explained_variance_ratio_

        var_exp = pca.explained_variance_ratio_
        var_exp1 = f'PC1 - Variance explained {var_exp[0]*100:.1f}%'
        var_exp2 = f'PC2 - Variance explained {var_exp[1]*100:.1f}%'


        plt.figure(figsize=(7, 5))
        # Get unique groups and a color palette
        groups = pca_df[self.cat_var].unique()
        palette = sns.color_palette("Paired", n_colors=len(groups))
        color_map = dict(zip(groups, palette))

        # 2. Draw the blurred KDE shadow for each group
        for group, color in color_map.items():
            # Filter data for the current group
            group_data = pca_df[pca_df[self.cat_var] == group]
            sns.kdeplot(
                data=group_data, x='pc1', y='pc2',
                color=color,
                fill=True,       # This is the key to creating the shadow
                alpha=0.4,       # Controls the transparency of the shadow
                thresh=0.1,      # Hides the faint, outer edges of the density
                levels=5,         # Number of contour levels
                warn_singular=False
            )

        # 3. Draw the scatter plot on top of the shadows
        sns.scatterplot(
            data=pca_df, x='pc1', y='pc2',
            hue=self.cat_var, palette=color_map, 
            s=30, alpha=1,
            edgecolor='k', linewidth=0.5,
            legend=True 
        )

        # 4. Final plot adjustments
        # Using dummy values for explained variance for this example
        plt.xlabel(var_exp1, fontsize=16)
        plt.ylabel(var_exp2, fontsize=16)
        legend = plt.legend(title='Cardiovascular Disease', loc='lower left', fontsize=16)
        legend.get_title().set_fontsize(16)
        # plt.title('PCA of Latent Representations', fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        self._save(plot_name)
        plt.show()







def calculate_jsd(x_obs:torch.Tensor, x_recon:torch.Tensor):
    """Calculates the Jensen-Shannon Divergence between two sets of distributions.
    Args:
        x_obs: The original data of shape (n_samples, n_features).
        x_recon: The generated data of shape (n_samples, n_features).
    Returns:
        float: The Jensen-Shannon Divergence value. A lower value indicates more
        similar distributions.
    """
    # 1. Calculate the mean distribution for each dataset
    p_mean = np.mean(x_obs.numpy(), axis=0)
    q_mean = np.mean(x_recon.numpy(), axis=0)

    # 2. Compute the Jensen-Shannon distance (which is sqrt(JSD))
    # The base of the logarithm is `e` by default.
    js_distance = jensenshannon(p_mean, q_mean)

    # 3. Square the distance to get the divergence
    js_divergence = js_distance**2

    return js_divergence

def calculate_zeros(x_obs:torch.Tensor, x_recon:torch.Tensor, type:str):
    """Calculate sparsity error"""

    sparsity_obs = (x_obs.numpy() == 0).mean(-1)
    if type == 'sza':
        sparsity_recon = (x_recon.numpy() == 0).mean(-1)
    else:
        sparsity_recon = (x_recon.numpy() <= 0.01).mean(-1)
    
    return np.abs(np.mean(sparsity_obs) - np.mean(sparsity_recon))

# def calculate_alpha(x_obs:torch.Tensor, x_recon:torch.Tensor):

#     alpha_obs = alpha_diversity(metric='shannon', counts=x_obs.numpy()).values.mean()
#     alpha_recon = alpha_diversity(metric='shannon', counts=x_recon.numpy()).values.mean()

#     return np.abs(alpha_obs-alpha_recon)