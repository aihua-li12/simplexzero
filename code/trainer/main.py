# main.py

from data import *
from model import *
import torch

import os
os.getcwd()

from plot import *

# ----- 1. Load data -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abundance, metadata = preprocess(tax_level="Family", 
                                 agg_abundance_dir="../../data/aggregation",
                                 metadata_dir="../../data/matched")
metadata.set_index("#SampleID", inplace=True)
metadata = metadata[metadata['sample_type'] == 'Stool']

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
vae_losses = vae_trainer.train(n_epochs, train_loader=loader)
vae_recon = vae.sample(n_samples)



# ----- 3. GAN -----
g_nn = Generator(latent_dim, n_features, use_batch_norm=False, n_layers=3)
d_nn = Discriminator(n_features, use_batch_norm=False, n_layers=3)

gan_trainer = GANTrainer(g_nn, d_nn, device)
gan_losses = gan_trainer.train(n_epochs, train_loader=loader)
gan_recon = g_nn.sample(n_samples)



# ----- 4. Flow -----
flow_model = FlowMatching(n_features, n_resiblocks=5)
flow_trainer = FlowTrainer(flow_model, device)
flow_losses = flow_trainer.train(n_epochs, train_loader=loader)

flow_sampler = FlowSampler(flow_model, n_samples, n_steps=1000, simplex_aware=False)
flow_recon = flow_sampler.simulate()
flow_recon = CustomActivation()(flow_recon)



# ----- 5. Diffusion -----
score_model = ScoreMatching(n_features, n_resiblocks=5)
score_trainer = ScoreTrainer(score_model, device)
score_losses = score_trainer.train(n_epochs, train_loader=loader)
diffusion_recon = DiffusionSampler(score_model, sigma = 1.0, n_samples=n_samples)




# ----- 6. Evaluation -----
x_obs = dataset.sample(n_samples)

plot = Plot(x_obs, gan_recon, "../../result/gan")
plot.histogram(4)
plot.mean_variance()
plot.sparsity()









file_path = '/home/jupyter/data/aggregation/AGP.taxonomyASV.parquet'
abundance_tax_pl = pl.read_parquet(file_path)