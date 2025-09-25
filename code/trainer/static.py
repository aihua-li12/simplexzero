from model import SymmetricDirichlet, LinearConditionalProbabilityPath
import torch
from plot import PlotSimplex





dir_init = SymmetricDirichlet(dirich_param=torch.tensor([1,1,1]))
dir_final = SymmetricDirichlet(dirich_param=torch.tensor([0.2,0.2,0.2]))
path = LinearConditionalProbabilityPath(dir_init, dir_final)
n_sample = 2000
z = path.sample_conditioning_variable(n_sample)
ts = torch.cat(
    [torch.linspace(0, 0.5, steps=10), torch.linspace(0.5, 1, steps=20)]
)


data = []
for i in range(len(ts)):
    t = ts[i].expand(n_sample)
    data.append(path.sample_conditional_path(z, t))
data = torch.stack(data, dim=0).numpy()
data_prior = dir_init.sample(n_sample).numpy()
data_true = dir_final.sample(n_sample).numpy()



PlotSimplex(data, data_prior, data_true, fps=15)
