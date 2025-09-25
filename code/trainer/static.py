from model import *
import torch
from plot import PlotSimplex






n_sample = 2000
dir_final = SymmetricDirichlet(dirich_param=torch.tensor([0.2,0.2,0.2]))
data_true = dir_final.sample(n_sample).numpy()
fps = 12

# ---- Flow true ----
dir_init = SymmetricDirichlet(dirich_param=torch.tensor([1,1,1]))
data_prior = dir_init.sample(n_sample).numpy()
path = LinearConditionalProbabilityPath(dir_init, dir_final)
z = path.sample_conditioning_variable(n_sample)
ts = torch.cat(
    [torch.linspace(0, 0.6, 15), torch.linspace(0.6, 1, 35)]
)
data = []
for i in range(len(ts)):
    t = ts[i].expand(n_sample)
    data.append(path.sample_conditional_path(z, t))
data = torch.stack(data, dim=0).numpy()

PlotSimplex(data, data_prior, data_true, fps=fps, plot_save_name="flow_true")



# ---- Flow learned ----

# ---- Diffusion true ----
dir_init = StandardNormal(3)
data_prior = dir_init.sample(n_sample).numpy()
path = GaussianConditionalProbabilityPath(
    dir_init, dir_final, alpha = LinearAlpha(), beta = DiminishingBeta(c1=1)
)
z = path.sample_conditioning_variable(n_sample)
ts = torch.linspace(0, 1, 50)

data = []
for i in range(len(ts)):
    t = ts[i].expand(n_sample)
    data.append(path.sample_conditional_path(z, t))
data = torch.stack(data, dim=0).numpy()

PlotSimplex(data, data_prior, data_true, fps=fps, plot_save_name="diffusion_true")


# ---- Diffusion learned ----










