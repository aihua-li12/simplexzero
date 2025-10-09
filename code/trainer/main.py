# main.py

from data import *
from model import *
import torch

import os
os.getcwd()

from plot import *
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

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
def vae_helper(loader:DataLoader, type:str):
    torch.manual_seed(1)
    vae = VAE(n_features, latent_dim, use_batch_norm=True, n_layers=4)
    vae_trainer = VAETrainer(vae, device)
    vae_losses = vae_trainer.train(n_epochs, train_loader=loader)
    vae_recon = vae.sample(n_samples)
    if type == 'sza':
        return vae_recon
    if type == 'clr':
        return torch.softmax(vae_recon, dim=-1)*100
    
# vae_recon_sza = vae_helper(loader_sza, 'sza')
# vae_recon_clr = vae_helper(loader_clr, 'clr')


# samples_dict = {
#     'Observed': x_obs,
#     'Reconstructed (SZA)': vae_recon_sza,
#     'Reconstructed (CLR)': vae_recon_clr
# }
# plot = Plot(samples_dict, "../../result/vae")
# plot.mean_variance()
# plot.beta_diversity()
# plot.stacked_bar(abundance)




# ----- 3. GAN -----
def gan_helper(loader:DataLoader, type:str):
    torch.manual_seed(1)
    g_nn = Generator(latent_dim, n_features, n_layers=2)
    d_nn = Discriminator(n_features, n_layers=2)
    gan_trainer = GANTrainer(g_nn, d_nn, device)
    gan_losses = gan_trainer.train(n_epochs, train_loader=loader)
    gan_recon = g_nn.sample(n_samples)
    if type == 'sza':
        return gan_recon
    if type == 'clr':
        return torch.softmax(gan_recon, dim=-1)*100

# gan_recon_sza = gan_helper(loader_sza, 'sza')
# # gan_recon_clr = gan_helper(loader_clr, 'clr')



# samples_dict = {
#     'Observed': x_obs,
#     'Reconstructed (SZA)': gan_recon_sza
#     # 'Reconstructed (CLR)': gan_recon_clr
# }
# plot = Plot(samples_dict, "../../result/gan")
# plot.mean_variance()
# plot.beta_diversity()
# plot.stacked_bar(abundance)









# ----- 4. Flow -----
def flow_helper(loader:DataLoader, type:str):
    torch.manual_seed(1)
    flow_model = FlowMatching(n_features, hidden_dim=32, time_emb_dim=32, n_resiblocks=4)
    flow_trainer = FlowTrainer(flow_model, device)
    flow_losses = flow_trainer.train(n_epochs, train_loader=loader)

    flow_sampler = FlowSampler(flow_model, n_samples, n_steps=3000, simplex_aware=False)
    flow_recon = flow_sampler.simulate()
    if type == 'sza':
        return CustomActivation()(flow_recon)
    if type == 'clr':
        return torch.softmax(flow_recon, dim=-1)*100


# flow_recon_sza = flow_helper(loader_sza, "sza")
# flow_recon_clr = flow_helper(loader_clr, "clr")

# samples_dict = {
#     'Observed': x_obs, 
#     'Reconstructed (SZA)': flow_recon_sza,
#     'Reconstructed (CLR)': flow_recon_clr
# }
# plot = Plot(samples_dict, "../../result/flow")
# plot.mean_variance()
# plot.beta_diversity()
# plot.stacked_bar(abundance)




# samples_dict = {
#     'Observed': x_obs,
#     'Generated (VAE)': vae_recon_sza,
#     'Generated (GAN)': gan_recon_sza,
#     'Generated (Flow)': flow_recon_sza
# }
# plot = Plot(samples_dict, "../../result/comparison")
# plot.mean_variance()

# plot.stacked_bar(abundance, 5)






# ----- 5. Diffusion -----
def diffusion_helper(loader:DataLoader, type:str):
    torch.manual_seed(1)
    score_model = ScoreMatching(n_features, hidden_dim=64, time_emb_dim=64, n_resiblocks=2)
    score_trainer = ScoreTrainer(score_model, device)
    score_losses = score_trainer.train(n_epochs, train_loader=loader)
    ts = torch.linspace(0.1, 0.9, 20)
    sigma = 1.
    diffusion_sampler = DiffusionSampler(score_model, n_samples, ts=ts, sigma=sigma, simplex_aware=False)
    diffusion_recon = diffusion_sampler.simulate()
    if type == 'sza':
        return CustomActivation()(diffusion_recon)
    if type == 'clr':
        return torch.softmax(diffusion_recon, dim=-1)*100


# diffusion_recon_sza = diffusion_helper(loader_sza, "sza")
# diffusion_recon_clr = diffusion_helper(loader_clr, "clr")

# samples_dict = {
#     'Observed': x_obs,
#     'Reconstructed (SZA)': diffusion_recon_sza,
#     'Reconstructed (CLR)': diffusion_recon_clr
# }
# plot = Plot(samples_dict, "../../result/diffusion")
# plot.mean_variance()
# plot.beta_diversity()
# plot.stacked_bar(abundance, 5)









# ----- 6. Dirichlet Flow Matching (PMLR 2024) -----


