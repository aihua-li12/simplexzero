from model import *
from data import *
import torch
from plot import PlotSimplex
import pandas as pd 
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tot_sample = 100000
n_features = 3
n_epochs = 10
fps = 10


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



