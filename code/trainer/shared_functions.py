import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.nn.functional as F
import polars as pl
import torch.nn as nn
import torch


# Function to make stacked bar plot
def prepare_plot_dataframe(tax_level:str, n_taxa:int=5, n_subset:int=2000, seed:int=123,
                           x_recon:np.ndarray|None=None, abundance_df:pd.DataFrame|None=None) -> pl.dataframe.frame.DataFrame:
    """Get the dataframe corresponding to the required taxonomic level, 
    and prepare the dataframe for making the plot.
    
    Args:
        tax_level: One of ['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus'].
        n_taxa: Plot the n_taxa most abundant taxa. Other taxa are grouped as 'Other'.
        n_subset: Randomly subset the data and take n_subset samples. The full data is too huge to be plotted.
        seed: The random seed to randomly subset the data.
        x_recon: Reconstructed data of shape [n_samples, n_features].
        abundance_df: Abundance dataframe of shape [n_features, n_samples]. We will take the taxa names from it for x_recon.
    
    Returns:
        Returns a polar dataframe where the first column is the sample_id, 
        and the rest of the columns are the relative abundance of different taxa in this taxonomic level.
        Sample ids are sorted according to their relative abundance in the most abundant taxa.
    """
    if x_recon is None and abundance_df is None:
        # Extract aggregated data
        file_path = os.path.join('~/desktop/data-full/aggASV/', f"aggASV_{tax_level}.tsv")
        df = pl.read_csv(file_path, separator='\t')
    if x_recon is not None and abundance_df is not None:
        df = pl.from_numpy(abundance_df.index, schema=[tax_level]).with_columns(
            pl.from_numpy(x_recon, schema=["sample"+str(i) for i in range(x_recon.shape[0])])
        )

    # Pivot the data longer and compute the relative abundance for each sample
    df_long = df.unpivot(index=tax_level, variable_name='sample_id', value_name='abundance')
    df_rel_abun = df_long.with_columns(
        (pl.col('abundance') * 100 / pl.col('abundance').sum().over('sample_id')).alias('relative_abundance')
    )
    df_rel_abun = df_rel_abun.with_columns(
        # Taxonomic categories with one sample with zero count have relative abundance NaN. Replace with 0
        pl.col('relative_abundance').fill_nan(0)
    )


    # Compute the mean of relative abundance for each taxa across samples
    df_means = df_rel_abun.group_by(tax_level).agg(
        pl.mean('relative_abundance').alias('mean_abundance')
    )

    # Only plot taxa with the top several highest average relative abundance. Group others as 'Other'
    top_k = min(len(set(df_means[tax_level])), n_taxa)
    abundance_threshold = sorted(df_means['mean_abundance'].to_list(), reverse=True)[top_k-1]
    abundant_groups = df_means.filter(pl.col('mean_abundance') >= abundance_threshold)[tax_level]

    df_processed = df_rel_abun.with_columns(
        pl.when(pl.col(tax_level).is_in(abundant_groups))
        .then(pl.col(tax_level))
        .otherwise(pl.lit('Other'))
        .alias(tax_level)
    ).group_by(['sample_id', tax_level]).agg( # aggregate the relative abundance for "Other"
        pl.sum('relative_abundance')
    )

    # Pivot wider
    df_plot = df_processed.pivot(index='sample_id', on=tax_level, values='relative_abundance').fill_null(0)

    # Sort columns to have a consistent plotting order
    cols = sorted([col for col in df_plot.columns if col not in ['sample_id', 'Other']])
    if 'Other' in df_plot.columns:
        df_plot = df_plot.select(['sample_id'] + cols + ['Other'])
    else:
        df_plot = df_plot.select(['sample_id'] + cols)


    # Randomly subset the data for plot
    df_plot = df_plot.sample(n=n_subset, seed=seed)
    
    # Sort the sample ids according to their relative abundance in the most abundant taxa
    taxa_cols = df_plot.columns[1:]
    tax_cols_sums = df_plot[:,1:].to_numpy().sum(axis=0)
    sort_by = taxa_cols[np.argmax(tax_cols_sums)] # most abundant taxa
    df_plot_sorted = df_plot.sort(sort_by, descending=True)
    
    return df_plot_sorted


def make_stacked_bar_plot(tax_level:str, df_plot_sorted:pl.dataframe.frame.DataFrame, 
                          file_name:str|None=None,
                          file_dir:str='/Users/aihuahome/Desktop/agp-project/results/gan'):
    """Make stacked bar plot for the given taxanomic level"""
    plot_path = os.path.join(file_dir, f"stacked_{tax_level}_{file_name}.pdf")
    
    taxa_cols = df_plot_sorted.columns[1:]
    x = np.arange(len(df_plot_sorted)) # position of x-axis

    fig, ax = plt.subplots(figsize=(15,5))
    bottom = np.zeros(len(df_plot_sorted)) # track the bottom of bars
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(taxa_cols)))

    for i, taxon in enumerate(taxa_cols):
        values = df_plot_sorted[taxon].to_numpy()
        ax.bar(x, values, bottom=bottom, label=taxon, width=1.0, color=colors[i])
        bottom += values

    tick_interval = 200 # adjust the interval based on the sample size
    tick_locs = np.arange(0, len(df_plot_sorted) + 1, tick_interval)
    tick_labels = [str(loc) for loc in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_title('Relative Abundance ', fontsize=13)
    ax.legend(title=f'Taxonomic Levels [{tax_level}]', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.ylim(0, 100)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(plot_path, dpi=300)
    plt.show()


