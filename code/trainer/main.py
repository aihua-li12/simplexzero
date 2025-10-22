# main.py

from data import *
from model import *
import torch

import os
os.getcwd()

from plot import *
from sklearn.preprocessing import StandardScaler
from skbio.stats.composition import ilr_inv, clr_inv

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
dataset_ilr = AbundanceDataset(abundance, transformation='ilr')

loader_creator = AbundanceLoader(batch_size=128, drop_last=False)
loader_sza = loader_creator.create_loader(dataset_sza)
loader_clr = loader_creator.create_loader(dataset_clr)
loader_ilr = loader_creator.create_loader(dataset_ilr)
# x = batch_data = next(iter(loader_ilr))


n_samples = 1000 # number of generated samples
n_epochs = 5
x_obs = torch.Tensor(abundance.T.sample(n_samples).values)





# ----- 2. VAE -----
def vae_helper(loader, type:str):
    torch.manual_seed(1)
    use_sza = type == 'sza'
    vae = VAE(input_dim=loader.dataset.dim, latent_dim=64, use_batch_norm=False, n_layers=3, use_sza=use_sza)
    vae_trainer = VAETrainer(vae, device)
    vae_losses = vae_trainer.train(n_epochs, train_loader=loader)
    vae_recon = vae.sample(n_samples)
    if type == 'sza':
        return vae_recon
    elif type == 'clr':
        return torch.Tensor(clr_inv(vae_recon))*100
    else:
        return torch.Tensor(ilr_inv(vae_recon))*100
    
vae_recon_sza = vae_helper(loader_sza, 'sza')
vae_recon_clr = vae_helper(loader_clr, 'clr')
vae_recon_ilr = vae_helper(loader_ilr, 'ilr')

samples_dict = {
    'Observed': x_obs,
    'Reconstructed (SZA)': vae_recon_sza,
    'Reconstructed (CLR)': vae_recon_clr,
    'Reconstructed (ILR)': vae_recon_ilr
}
plot = Plot(samples_dict, "../../result/vae")
plot.mean_variance()
plot.beta_diversity()
plot.stacked_bar(abundance)

print(calculate_jsd(x_obs, vae_recon_sza))
print(calculate_jsd(x_obs, vae_recon_clr))
print(calculate_jsd(x_obs, vae_recon_ilr))



# ----- 3. GAN -----
def gan_helper(loader, type:str):
    torch.manual_seed(1)
    use_sza = type == 'sza'
    g_nn = Generator(latent_dim=12, output_dim=loader.dataset.dim, n_layers=2, use_sza=use_sza, use_batch_norm=False)
    d_nn = Discriminator(input_dim=loader.dataset.dim, n_layers=2, use_batch_norm=False)
    gan_trainer = GANTrainer(g_nn, d_nn, device)
    gan_losses = gan_trainer.train(n_epochs, train_loader=loader)
    gan_recon = g_nn.sample(n_samples)
    if type == 'sza':
        return gan_recon
    elif type == 'clr':
        return torch.Tensor(clr_inv(gan_recon))*100
    else:
        return torch.Tensor(ilr_inv(gan_recon))*100


gan_recon_sza = gan_helper(loader_sza, 'sza')
gan_recon_clr = gan_helper(loader_clr, 'clr')
gan_recon_ilr = gan_helper(loader_ilr, 'ilr')


samples_dict = {
    'Observed': x_obs,
    'Reconstructed (SZA)': gan_recon_sza,
    'Reconstructed (CLR)': gan_recon_clr,
    'Reconstructed (ILR)': gan_recon_ilr
}
plot = Plot(samples_dict, "../../result/gan")
plot.mean_variance()
plot.beta_diversity()
plot.stacked_bar(abundance)


print(calculate_jsd(x_obs, gan_recon_sza))
print(calculate_jsd(x_obs, gan_recon_clr))
print(calculate_jsd(x_obs, gan_recon_ilr))




# ----- 4. Flow -----
def flow_helper(loader, type:str):
    torch.manual_seed(1)
    flow_model = FlowMatching(loader.dataset.dim, hidden_dim=64, time_emb_dim=64, n_resiblocks=4)
    flow_trainer = FlowTrainer(flow_model, device)
    flow_losses = flow_trainer.train(n_epochs, train_loader=loader)

    flow_sampler = FlowSampler(flow_model, n_samples, n_steps=100, simplex_aware=False)
    flow_recon = flow_sampler.simulate()
    if type == 'sza':
        return CustomActivation()(flow_recon)
    elif type == 'clr':
        return torch.Tensor(clr_inv(flow_recon))*100
    else:
        return torch.Tensor(ilr_inv(flow_recon))*100


flow_recon_sza = flow_helper(loader_sza, "sza")
flow_recon_clr = flow_helper(loader_clr, "clr")
flow_recon_ilr = flow_helper(loader_ilr, "ilr")

samples_dict = {
    'Observed': x_obs, 
    'Reconstructed (SZA)': flow_recon_sza,
    'Reconstructed (CLR)': flow_recon_clr,
    'Reconstructed (ILR)': flow_recon_ilr
}
plot = Plot(samples_dict, "../../result/flow")
plot.mean_variance()
plot.beta_diversity()
plot.stacked_bar(abundance)

print(calculate_jsd(x_obs, flow_recon_sza))
print(calculate_jsd(x_obs, flow_recon_clr))
print(calculate_jsd(x_obs, flow_recon_ilr))





# ----- 5. Diffusion -----
def diffusion_helper(loader, type:str):
    torch.manual_seed(1)
    score_model = ScoreMatching(loader.dataset.dim, hidden_dim=512, time_emb_dim=128, n_resiblocks=1)
    score_trainer = ScoreTrainer(score_model, device)
    score_losses = score_trainer.train(n_epochs, train_loader=loader, lr=1e-3)
    ts = torch.linspace(0.05, 0.95, 1000)
    sigma = None
    diffusion_sampler = DiffusionSampler(score_model, n_samples, ts=ts, sigma=sigma, simplex_aware=False)
    diffusion_recon = diffusion_sampler.simulate()
    if type == 'sza':
        return diffusion_recon
    if type == 'clr':
        return torch.softmax(diffusion_recon, dim=-1)


data = abundance.values.T
scaler = StandardScaler().fit(data)
data_scaled = pd.DataFrame(scaler.fit_transform(data)).T

dataset_sza = AbundanceDataset(data_scaled)
# dataset_clr = AbundanceDataset(data_scaled, transformation='clr')

loader_creator = AbundanceLoader(batch_size=128, drop_last=False)
loader_sza = loader_creator.create_loader(dataset_sza)
# loader_clr = loader_creator.create_loader(dataset_clr)
# x = batch_data = next(iter(loader_sza))


n_epochs = 10


diffusion_recon_sza = diffusion_helper(loader_sza, "sza").numpy()
diffusion_recon_sza = torch.Tensor(scaler.inverse_transform(diffusion_recon_sza))
diffusion_recon_sza = CustomActivation()(diffusion_recon_sza)


samples_dict = {
    'Observed': x_obs,
    'Reconstructed (SZA)': diffusion_recon_sza
}
plot = Plot(samples_dict, "../../result/diffusion")
plot.mean_variance()
# plot.beta_diversity()
plot.stacked_bar(abundance, 5)











samples_dict = {
    'Observed': x_obs,
    'Generated (VAE)': vae_recon_sza,
    'Generated (GAN)': gan_recon_sza,
    'Generated (Diffusion)': diffusion_recon_sza
}
plot = Plot(samples_dict, "../../result/comparison")
plot.mean_variance()
plot.stacked_bar(abundance, 5)




# ----- Table -----
results = {
    'VAE': {
        'SZA': [calculate_jsd(x_obs, vae_recon_sza),
                calculate_zeros(x_obs, vae_recon_sza, "sza"),
                calculate_alpha(x_obs, vae_recon_sza)],
        'CLR': [calculate_jsd(x_obs, vae_recon_clr),
                calculate_zeros(x_obs, vae_recon_clr, "clr"),
                calculate_alpha(x_obs, vae_recon_clr)],
        'ILR': [calculate_jsd(x_obs, vae_recon_ilr),
                calculate_zeros(x_obs, vae_recon_ilr, "ilr"),
                calculate_alpha(x_obs, vae_recon_ilr)]
    },
    'GAN': {
        'SZA': [calculate_jsd(x_obs, gan_recon_sza),
                calculate_zeros(x_obs, gan_recon_sza, "sza"),
                calculate_alpha(x_obs, gan_recon_sza)],
        'CLR': [calculate_jsd(x_obs, gan_recon_clr),
                calculate_zeros(x_obs, gan_recon_clr, "clr"),
                calculate_alpha(x_obs, gan_recon_clr)],
        'ILR': [calculate_jsd(x_obs, gan_recon_ilr),
                calculate_zeros(x_obs, gan_recon_ilr, "ilr"),
                calculate_alpha(x_obs, gan_recon_ilr)]
    },
    'Flow': {
        'SZA': [calculate_jsd(x_obs, flow_recon_sza),
                calculate_zeros(x_obs, flow_recon_sza, "sza"),
                calculate_alpha(x_obs, flow_recon_sza)],
        'CLR': [calculate_jsd(x_obs, flow_recon_clr),
                calculate_zeros(x_obs, flow_recon_clr, "clr"),
                calculate_alpha(x_obs, flow_recon_clr)],
        'ILR': [calculate_jsd(x_obs, flow_recon_ilr),
                calculate_zeros(x_obs, flow_recon_ilr, "ilr"),
                calculate_alpha(x_obs, flow_recon_ilr)]
    }
}


rows = []
for model, methods_data in results.items():
    jsd_row = {'Model': model, 'Metric': 'JSD'}
    zeros_row = {'Model': model, 'Metric': 'Zeros'}
    alpha_row = {'Model': model, 'Metric': 'Alpha'}
    for method, values in methods_data.items():
        jsd_row[method] = values[0]
        zeros_row[method] = values[1]
        alpha_row[method] = values[2]
    rows.extend([jsd_row, zeros_row, alpha_row])

df_results = pd.DataFrame(rows)
df_results.set_index(['Model', 'Metric'], inplace=True)

print("--- Multi-index DataFrame Results ---")
print("This structure will be converted to the LaTeX table.\n")
print(df_results.to_string(float_format="%.6f"))
print("\n" + "="*50 + "\n")


import re
df_latex = df_results.copy()

# Apply bold formatting to the numbers in the 'SZA' column
df_latex['SZA'] = df_latex['SZA'].apply(lambda x: f"\\textbf{{{x:.4f}}}")

# Rename the 'SZA' column header to be bold
df_latex.rename(columns={'SZA': r'\textbf{SZA}'}, inplace=True)

# To achieve single-line model names, we manipulate the index
df_latex.reset_index(inplace=True)
# Replace the model name with an empty string for the second metric row
df_latex.loc[df_latex['Model'].duplicated(), 'Model'] = ''
df_latex.set_index(['Model', 'Metric'], inplace=True)

# Generate the initial LaTeX code. Pandas might add a \cline after every row.
# We will manually clean these up using regex substitutions.
latex_code = df_latex.to_latex(
    float_format="%.4f",
    caption="Comparison of JSD and Zero Difference for generated samples.",
    label="tab:full_results",
    position="!htbp",
    # Do not use multirow as we manually handled the index
    escape=False  # Allow LaTeX commands
)

# --- Custom LaTeX Table Formatting ---
# 1. Remove the \cline within each model class (i.e., after the 'JSD' row)
#    This regex finds a line ending in 'JSD & ... \\', captures it, and removes
#    the following line if it is a \cline.
processed_latex = re.sub(r'(JSD\s*&.*\\\\\n)\\cline\{.+?\}\n', r'\1', latex_code)

# 2. Remove the final \cline that appears just before \bottomrule
processed_latex = re.sub(r'\\cline\{.+?\}\n(\\bottomrule)', r'\1', processed_latex)

# 3. Insert the \centering command to center the table
centered_latex_code = processed_latex.replace(
    r'\begin{tabular}', r'\centering' + '\n' + r'\begin{tabular}'
)

print("--- LaTeX Code for Publication ---")
print(centered_latex_code)