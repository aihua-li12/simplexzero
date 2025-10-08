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

loader_creator = AbundanceLoader(batch_size=256, drop_last=False)
loader_sza = loader_creator.create_loader(dataset_sza)
loader_clr = loader_creator.create_loader(dataset_clr)
# x = batch_data = next(iter(loader))



n_features = dataset_sza.dim
n_samples = 1000 # number of generated samples
n_epochs = 5
latent_dim = 2
x_obs = dataset_sza.sample(n_samples, seed=1)


# ----- 2. VAE -----
torch.manual_seed(1)
vae_sza = VAE(n_features, latent_dim, use_batch_norm=False, n_layers=4)
vae_trainer_sza = VAETrainer(vae_sza, device)
vae_losses_sza = vae_trainer_sza.train(n_epochs, train_loader=loader_sza)
vae_recon_sza = vae_sza.sample(n_samples)

vae_clr = VAE(n_features, latent_dim, use_batch_norm=False, n_layers=4)
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





# ----- 3. GAN -----
torch.manual_seed(1)
g_nn_sza = Generator(latent_dim, n_features, use_batch_norm=False, n_layers=4)
d_nn_sza = Discriminator(n_features, use_batch_norm=False, n_layers=4)
gan_trainer_sza = GANTrainer(g_nn_sza, d_nn_sza, device)
gan_losses_sza = gan_trainer_sza.train(n_epochs, train_loader=loader_sza)
gan_recon_sza = g_nn_sza.sample(n_samples)


g_nn_clr = Generator(latent_dim, n_features, use_batch_norm=False, n_layers=4)
d_nn_clr = Discriminator(n_features, use_batch_norm=False, n_layers=4)
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









