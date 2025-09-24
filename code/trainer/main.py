# main.py

from data import *
from model import *
import torch

import os
os.getcwd()

import matplotlib.pyplot as plt
import skbio
from skbio.stats.ordination import pcoa
from skbio.diversity.alpha import shannon
import seaborn as sns
from plot import *

# ----- 1. Load data -----
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abundance, metadata = preprocess(tax_level="Phylum", 
                                 agg_abundance_dir="../../data/aggregation",
                                 metadata_dir="../../data/matched")

    
dataset = AbundanceDataset(abundance)
loader_creator = AbundanceLoader(batch_size=256, drop_last=False)
loader = loader_creator.create_loader(dataset)
# x = batch_data = next(iter(loader))

# batch_data = batch_data[:2, :4]
# x = batch_data



n_features = dataset.dim
n_samples = 1000 # number of generated samples
n_epochs = 10
batch_print_freq = 500
latent_dim = 2

# ----- 2. VAE -----
vae = VAE(n_features, latent_dim)
vae_trainer = VAETrainer(vae, device)
vae_losses = vae_trainer.train(n_epochs, train_loader=loader, batch_print_freq=batch_print_freq)
vae_recon = vae.sample(n_samples)


# ----- 3. GAN -----
g_nn = Generator(latent_dim, n_features, use_batch_norm=False, n_layers=4)
d_nn = Discriminator(n_features, use_batch_norm=False, n_layers=4)

gan_trainer = GANTrainer(g_nn, d_nn, device)
gan_losses = gan_trainer.train(n_epochs, train_loader=loader, batch_print_freq=batch_print_freq)
gan_recon = g_nn.sample(n_samples)




# ----- 4. Flow -----
flow_model = FlowMatching(n_features, n_resiblocks=5)
flow_trainer = FlowTrainer(flow_model, device)
flow_losses = flow_trainer.train(n_epochs, train_loader=loader, batch_print_freq=batch_print_freq)
flow_recon = FlowSampler(flow_model, n_samples)



# ----- 5. Diffusion -----
score_model = ScoreMatching(n_features, n_resiblocks=5)
score_trainer = ScoreTrainer(score_model, device)
score_losses = score_trainer.train(n_epochs, train_loader=loader, batch_print_freq=batch_print_freq)
diffusion_recon = DiffusionSampler(flow_model, score_model, sigma = 1.0, n_samples=n_samples)




# ----- 6. Evaluation -----
x_obs = dataset.sample(n_samples)

plot = Plot(x_obs, flow_recon, "../../result/diffusion")
plot.histogram(4)
plot.mean_variance()
plot.sparsity()
