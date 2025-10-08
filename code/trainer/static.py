# static.py

from model import *
from data import *
from plot import *
import torch
import pandas as pd 
torch.manual_seed(123)


# ================================================================================ #
#   This script generates the plots in the webpage.                                #
#   Different parts correspond to different plots. Run with care.                  #
# ================================================================================ #



# ========================================================== #
#          Part 1: animation of flow and diffusion           #
# ========================================================== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tot_sample = 100000
n_features = 3
n_epochs = 10
fps = 10

# ---- True data ----
p_true = SymmetricDirichlet(dirich_param=torch.tensor([0.1,0.1,0.1]))
data_all = p_true.sample(tot_sample).numpy()
dataset = AbundanceDataset(pd.DataFrame(data_all.T))
loader_creator = AbundanceLoader(batch_size=10000, drop_last=False)
loader = loader_creator.create_loader(dataset)


n_sample = 2000
data_true = p_true.sample(n_sample).numpy()

# ---- Flow true ----
ts = torch.cat([torch.linspace(0, 0.5, 10), torch.linspace(0.5, 1, 20)])
p_init = SymmetricDirichlet(n_features)
path = LinearConditionalProbabilityPath(p_init, p_true)
z = path.sample_conditioning_variable(n_sample)

data = []
for i in range(len(ts)):
    t = ts[i].expand(n_sample)
    data.append(path.sample_conditional_path(z, t))
data = torch.stack(data, dim=0).numpy()


PlotSimplex(data, 
            data_prior = p_init.sample(n_sample).numpy(),
            data_true = data_true, 
            fps=fps, 
            title="Target probability path", 
            prior_tlt="Prior distribution",
            true_tlt="True distribution",
            plot_save_name="flow_true")




# ---- Flow learned ----
flow_model = FlowMatching(n_features, n_resiblocks=16, hidden_dim=16, time_emb_dim=128)
flow_trainer = FlowTrainer(flow_model, device)
flow_losses = flow_trainer.train(n_epochs, train_loader=loader, val_freq=50)
flow_sampler = FlowSampler(flow_model, n_sample, ts=ts, simplex_aware=False)
flow_recon = flow_sampler.simulate_with_trajectory()
flow_recon = (CustomActivation()(flow_recon)/100).detach().numpy()


PlotSimplex(flow_recon, flow_recon[0], flow_recon[-1], fps=fps, 
            title="Learned probability path", 
            prior_tlt="Prior distribution",
            true_tlt="Learned distribution",
            plot_save_name="flow_learned")



# ---- Diffusion true ----
ts = torch.cat([torch.linspace(0.2, 0.8, 5), torch.linspace(0.85, 1, 25)])
p_init = SymmetricDirichlet(n_features)
# p_init = StandardNormal(n_features)
path = GaussianConditionalProbabilityPath(
    p_init, p_true, alpha = LinearAlpha(), beta = SquareRootBeta(c=0.01)
)
z = path.sample_conditioning_variable(n_sample)

data = []
for i in range(len(ts)):
    t = ts[i].expand(n_sample)
    data.append(path.sample_conditional_path(z, t))
data = torch.stack(data, dim=0)
# data = CustomActivation()(data)/100
data = data.numpy()

PlotSimplex(data, 
            data_prior = p_init.sample(n_sample).numpy(),
            data_true = data_true, 
            fps=fps, 
            title="Target probability path", 
            prior_tlt="Prior distribution",
            true_tlt="True distribution",
            plot_save_name="diffusion_true")


# ---- Diffusion learned ----
score_model = ScoreMatching(n_features, n_resiblocks=16, hidden_dim=64, time_emb_dim=128)
score_trainer = ScoreTrainer(score_model, device)
score_losses = score_trainer.train(n_epochs, train_loader=loader, val_freq=50)

ts = torch.linspace(0.35, 1, 30)
sigma = torch.linspace(0.9, 0.3, len(ts))
diffusion_sampler = DiffusionSampler(score_model, p_init=p_init, sigma=sigma,
                                     n_samples=n_sample, ts=ts, simplex_aware=False)

diffusion_recon = diffusion_sampler.simulate_with_trajectory()
# diffusion_recon = CustomActivation()(diffusion_recon)/100
diffusion_recon[-1] = CustomActivation()(diffusion_recon[-1])/100
diffusion_recon = diffusion_recon.detach().numpy()


PlotSimplex(diffusion_recon, diffusion_recon[0], diffusion_recon[-1], fps=fps, 
            title="Target probability path", 
            prior_tlt="Prior distribution",
            true_tlt="Learned distribution",
            plot_save_name="diffusion_learned")





# ==================================================================== #
#       Part 2: Amplify cardiovascular disease diagnosed samples       #
# ==================================================================== #



# ----- 1. Load data -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
abundance, metadata = preprocess(tax_level="Genus", 
                                 agg_abundance_dir="../../data/aggregation",
                                 metadata_dir="../../data/matched")
metadata.set_index("#SampleID", inplace=True)
metadata = metadata[metadata['sample_type'] == 'Stool']


# ----- 2. Select sample ids -----
torch.manual_seed(123)
cat_var = 'cardiovascular_disease'
metadata.groupby(cat_var).size()
meta_diagnosed = metadata[metadata[cat_var].str.startswith("Diagnosed")][cat_var]
sample_ids_diagnosed = meta_diagnosed.index
abundance_diagnosed = abundance.T.loc[sample_ids_diagnosed].T


# ----- 3. Amplify diagnosed samples by GAN -----
dataset = AbundanceDataset(abundance_diagnosed)
loader_creator = AbundanceLoader(batch_size=256, drop_last=False)
loader = loader_creator.create_loader(dataset)

n_features = dataset.dim
n_samples = 1000 # number of generated samples
n_epochs = 5
latent_dim = 32

g_nn = Generator(latent_dim, n_features, use_batch_norm=False, n_layers=3)
d_nn = Discriminator(n_features, use_batch_norm=False, n_layers=3)

gan_trainer = GANTrainer(g_nn, d_nn, device)
gan_losses = gan_trainer.train(n_epochs, train_loader=loader)
gan_recon = g_nn.sample(n_samples)


# ----- 4. Combine abundance data -----
new_sample_ids = [f'new_sample_{i}' for i in range(n_samples)]
new_abundance = pd.DataFrame(data=gan_recon.numpy(), index=new_sample_ids, 
                             columns=abundance_diagnosed.T.columns)
abundance_diagnosed = pd.concat([abundance_diagnosed.T, new_abundance])

# ----- 5. Combine metadata -----
meta_diagnosed = pd.Series("Diagnosed", 
                           index=list(sample_ids_diagnosed) + list(new_sample_ids))

# ----- 6. Combine with healthy data -----
meta_healthy = metadata[metadata[cat_var].str.startswith("I do not")].sample(1500, random_state=42)
meta_healthy = meta_healthy[cat_var]
meta_healthy[:] = "Healthy"
meta = pd.concat([meta_diagnosed, meta_healthy])
meta.name = cat_var

abundance_healthy = abundance.T.loc[meta_healthy.index]
data = pd.concat([abundance_diagnosed, abundance_healthy])

del(abundance_diagnosed, abundance_healthy,
    meta_diagnosed, meta_healthy, sample_ids_diagnosed, 
    new_sample_ids, new_abundance, n_features, n_epochs, latent_dim, 
    d_nn, g_nn, gan_trainer, gan_losses, gan_recon, dataset, loader_creator, loader)


# ----- 7. Select relevant bacteria -----
genus = ['Kocuria', 'Staphylococcus', 'Faecalibacterium', 'Enhydrobacter'] 
data_genus = data.loc[:,genus]




# ==================================================================== #
#            Part 3: PCoA of cardiovascular disease of GAN             #
#                     (Run this after part 2)                          #
# ==================================================================== #
plot_cardio = PlotCardiovascular(meta, plot_save_dir='../../result/static')
plot_cardio.pcoa_plot(data_genus, 'cardio_pcoa')






# ==================================================================== #
#                Part 4: Latent dimension plots of VAE                 #
#                     (Run this after part 2)                          #
# ==================================================================== #

# ---- VAE on amplified data -----
torch.manual_seed(123)
dataset_sza = AbundanceDataset(data.T)
dataset_clr = AbundanceDataset(data.T, transformation='clr')

loader_creator = AbundanceLoader(batch_size=256, drop_last=False)
loader_sza = loader_creator.create_loader(dataset_sza)
loader_clr = loader_creator.create_loader(dataset_clr)



n_features = dataset_sza.dim
n_epochs = 10
latent_dim = 64

vae_sza = VAE(n_features, latent_dim)
vae_trainer_sza = VAETrainer(vae_sza, device)
vae_losses_sza = vae_trainer_sza.train(n_epochs, train_loader=loader_sza)
_, z_sza, _, _ = vae_sza(torch.tensor(data.values, dtype=torch.float32))

 
vae_clr = VAE(n_features, latent_dim)
vae_trainer_clr = VAETrainer(vae_clr, device)
vae_losses_clr = vae_trainer_clr.train(n_epochs, train_loader=loader_clr)
_, z_clr, _, _ = vae_clr(torch.tensor(data.values, dtype=torch.float32))


# ----- PCA plot -----
plot_cardio = PlotCardiovascular(meta, plot_save_dir='../../result/static')
plot_cardio.pca_plot(z_sza, 'cardio_pca_sza')
plot_cardio.pca_plot(z_clr, 'cardio_pca_clr')