# main.py

from data import *
from model import *
import torch

import os
os.getcwd()

from plot import *

# ----- 1. Load data -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abundance, metadata = preprocess(tax_level="Class", 
                                 agg_abundance_dir="../../data/aggregation",
                                 metadata_dir="../../data/matched")
metadata.set_index("#SampleID", inplace=True)
metadata = metadata[metadata['sample_type'] == 'Stool']



# ----- 2. Create data loader -----
dataset_sza = AbundanceDataset(abundance)
dataset_clr = AbundanceDataset(abundance, transformation='clr')

loader_creator = AbundanceLoader(batch_size=128, drop_last=False)
loader_sza = loader_creator.create_loader(dataset_sza)
loader_clr = loader_creator.create_loader(dataset_clr)
# x = batch_data = next(iter(loader))


n_features = dataset_sza.dim
n_samples = 1000 # number of generated samples
n_epochs = 10
latent_dim = 12
x_obs = dataset_sza.sample(n_samples, seed=1)


# ----- 2. VAE -----
torch.manual_seed(1)
vae_sza = VAE(n_features, latent_dim, use_batch_norm=True, n_layers=4)
vae_trainer_sza = VAETrainer(vae_sza, device)
vae_losses_sza = vae_trainer_sza.train(n_epochs, train_loader=loader_sza)
vae_recon_sza = vae_sza.sample(n_samples)

vae_clr = VAE(n_features, latent_dim, use_batch_norm=True, n_layers=4)
vae_trainer_clr = VAETrainer(vae_clr, device)
vae_losses_clr = vae_trainer_clr.train(n_epochs, train_loader=loader_clr)
vae_recon_clr = vae_clr.sample(n_samples)
vae_recon_clr = torch.softmax(vae_recon_clr, dim=-1)*100


samples_dict = {
    'Observed': x_obs,
    'Reconstructed (SZA)': vae_recon_sza,
    'Reconstructed (CLR)': vae_recon_clr
}
plot = Plot(samples_dict, "../../result/vae")
plot.mean_variance()
plot.beta_diversity()
plot.stacked_bar(abundance, n_taxa=5)




# ----- 3. GAN -----
torch.manual_seed(1)
g_nn_sza = Generator(latent_dim, n_features, n_layers=2)
d_nn_sza = Discriminator(n_features, n_layers=2)
gan_trainer_sza = GANTrainer(g_nn_sza, d_nn_sza, device)
gan_losses_sza = gan_trainer_sza.train(n_epochs, train_loader=loader_sza)
gan_recon_sza = g_nn_sza.sample(n_samples)


g_nn_clr = Generator(latent_dim, n_features, n_layers=2)
d_nn_clr = Discriminator(n_features, n_layers=2)
gan_trainer_clr = GANTrainer(g_nn_clr, d_nn_clr, device)
gan_losses_clr = gan_trainer_clr.train(n_epochs, train_loader=loader_clr)
gan_recon_clr = g_nn_clr.sample(n_samples)
gan_recon_clr = torch.softmax(gan_recon_clr, dim=-1)*100



samples_dict = {
    'Observed': x_obs,
    'Reconstructed (SZA)': gan_recon_sza,
    'Reconstructed (CLR)': gan_recon_clr
}
plot = Plot(samples_dict, "../../result/gan")
plot.mean_variance()
plot.beta_diversity()
plot.stacked_bar(abundance, 5)



# samples_dict = {
#     'Observed': x_obs,
#     'Generated (VAE)': vae_recon_sza,
#     'Generated (GAN)': gan_recon_sza
# }
# plot = Plot(samples_dict, "../../result/comparison")
# plot.mean_variance()

# plot.stacked_bar(abundance, 5)







# # ----- 4. Flow -----
# flow_model = FlowMatching(n_features, n_resiblocks=5)
# flow_trainer = FlowTrainer(flow_model, device)
# flow_losses = flow_trainer.train(n_epochs, train_loader=loader)

# flow_sampler = FlowSampler(flow_model, n_samples, n_steps=1000, simplex_aware=False)
# flow_recon = flow_sampler.simulate()
# flow_recon = CustomActivation()(flow_recon)












# # ----- 5. Diffusion -----
# score_model = ScoreMatching(n_features, n_resiblocks=5)
# score_trainer = ScoreTrainer(score_model, device)
# score_losses = score_trainer.train(n_epochs, train_loader=loader)
# diffusion_recon = DiffusionSampler(score_model, sigma = 1.0, n_samples=n_samples)









# ----- 6. Dirichlet Flow Matching (PMLR 2024) -----






def prepare_plot_dataframe(abundance:pd.DataFrame, n_taxa:int=5, 
                           n_subset:int=1000, seed:int=123) -> pl.dataframe.frame.DataFrame:
    """Prepare the dataframe for making the stacked bar plot.
    Args:
        abundance: DataFrame from `preprocess()`.
        n_taxa: Plot the n_taxa most abundant taxa. Other taxa are grouped as 'Other'.
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
        x_obs.numpy().T, 
        index=abundance.index, 
        columns=['sample'+str(i) for i in range(x_obs.shape[0])]
    )
    df = pl.from_dataframe(df)

    # Pivot the data longer and compute the relative abundance for each sample
    df_rel_abun = df.unpivot(index=tax_level, variable_name='sample_id', value_name='relative_abundance')

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


def stacked_bar(df_plot_sorted, n_taxa:int=5, 
                          n_subset:int=1000, seed:int=123):

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

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # plt.savefig(plot_path, dpi=300)
    plt.show()


stacked_bar(df_plot_sorted)