from model import *
del(FlowTrainer)
import torch
from plot import PlotSimplex
from data import *
import pandas as pd 


class FlowTrainer(Trainer):
    def get_batch_loss(self, batch_data:torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss over the input batch using flow matching model
        Args:
            batch_data: (batch_size, dim)
        Returns:
            loss of this batch
        """
        batch_size = batch_data.size(0)
        p_init = SymmetricDirichlet(dirich_param=torch.tensor([1,1,1]))
        p_data = EmpiricalDistribution(batch_data)
        path = LinearConditionalProbabilityPath(p_init, p_data)
        z = path.sample_conditioning_variable(batch_size).to(torch.float32) # (batch_size, dim)
        t = torch.rand(batch_size, device=self.device) # (batch_size,)
        xt = path.sample_conditional_path(z, t) # (batch_size, dim)

        vec_field_learned = self.model(xt, t) # (batch_size, dim)
        vec_field_target = path.conditional_vector_field(xt, z, t) # (batch_size, dim)
        return F.mse_loss(vec_field_learned, vec_field_target, reduction='mean')

class ScoreTrainer(Trainer):
    def get_batch_loss(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss over the input batch using score matching model
        Args:
            batch_data: (batch_size, dim)
        Returns:
            loss of this batch
        """
        batch_size = batch_data.size(0)
        p_init = StandardNormal(3)
        p_data = EmpiricalDistribution(batch_data)
        path = GaussianConditionalProbabilityPath(
            p_init, p_data, alpha = LinearAlpha(), beta = DiminishingBeta(c1=1)
        ).to(self.device)

        z = path.sample_conditioning_variable(batch_size).to(torch.float32) # (batch_size, dim)
        t = torch.rand(batch_size, device=self.device) # (batch_size,)
        xt = path.sample_conditional_path(z, t) # (batch_size, dim)

        score_field_learned = self.model(xt, t) # (batch_size, dim)
        score_field_target = path.conditional_score(xt, z, t) # (batch_size, dim)
        return F.mse_loss(score_field_learned, score_field_target, reduction='mean')
        





device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_sample = 2000
n_features = 3
n_epochs = 10
fps = 12



dir_final = SymmetricDirichlet(dirich_param=torch.tensor([0.2,0.2,0.2]))
data_true = dir_final.sample(n_sample).numpy()

dataset = AbundanceDataset(pd.DataFrame(data_true.T))
loader_creator = AbundanceLoader(batch_size=2000, drop_last=False)
loader = loader_creator.create_loader(dataset)


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

PlotSimplex(data, data_prior, data_true, fps=fps, 
            plot_save_name="flow_true")




# ---- Flow learned ----
flow_model = FlowMatching(n_features, n_resiblocks=5)
flow_trainer = FlowTrainer(flow_model, device)
flow_losses = flow_trainer.train(n_epochs, train_loader=loader)
ts = torch.cat(
    [torch.linspace(0, 0.6, 15), torch.linspace(0.6, 1, 35)]
)
flow_recon = FlowSamplerTrajectory(flow_model, n_sample, ts=ts).detach().numpy()

PlotSimplex(flow_recon, flow_recon[0], flow_recon[-1], fps=fps, 
            plot_save_name="flow_learned")




# ---- Diffusion true ----
dir_init = StandardNormal(n_features)
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
score_model = ScoreMatching(n_features, n_resiblocks=5)
score_trainer = ScoreTrainer(score_model, device)
score_losses = score_trainer.train(n_epochs, train_loader=loader)
ts = torch.linspace(0, 1, 50)
diffusion_recon = DiffusionSamplerTrajectory(flow_model, score_model, sigma = 0, 
                                             n_samples=n_sample, ts=ts).detach().numpy()
PlotSimplex(diffusion_recon, diffusion_recon[0], diffusion_recon[-1], fps=fps, 
            plot_save_name="diffusion_learned")







