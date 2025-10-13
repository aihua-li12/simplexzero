# model.py

# ================================================================================ #
#   This script contains model architectures, training and sampling methods        #
# ================================================================================ #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict
import random
from torch.utils.data.dataloader import DataLoader
from logging import Logger
import torch.distributed as dist
from tqdm import tqdm
from torch.func import vmap, jacrev
import math 

# ----- Activation Function ----- #
class CustomActivation(nn.Module):
    """Customize the activation function: f(x)_i = max(x_i,0) / (sum_{j=1}^d max(x_j,0) + epsilon)
    Args:
        epsilon: to add on the denominator for numerical stability
    """
    def __init__(self, epsilon:float=0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Applies the activation function.
        Args:
            x: (batch_size, d)
        """
        relu_x = F.relu(x) # (batch_size, d)
        numerator = relu_x
        denominator = torch.sum(relu_x, dim=-1, keepdim=True) + self.epsilon # (batch_size, 1)
        return numerator / denominator * 100






# ----- Sinusoidal Time Embedding ----- #
class SinusoidalTimeEmbedding(nn.Module):
    """Embed time t from (batch_size,) to (batch_size, time_emb_dim)
    Args:
        time_emb_dim: time embedding dimension
    """
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,)
        Returns:
            full_embedding: (batch_size, time_emb_dim)
        """
        half_dim = self.time_emb_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.unsqueeze(1) * embeddings.unsqueeze(0)
        
        sin_embeddings = torch.sin(embeddings)
        cos_embeddings = torch.cos(embeddings)
        full_embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=-1)
        
        if self.time_emb_dim % 2 == 1:
            full_embeddings = nn.functional.pad(full_embeddings, (0, 1))    
        return full_embeddings






# ----- Sampleable Distributions ----- #

class Sampleable(ABC):
    """Abstract base class for sampleable distributions"""
    @property
    @abstractmethod
    def distribution(self) -> Any:
        """Get the distribution information of this distribution"""
        pass

    @abstractmethod
    def sample(self, n_samples:int) -> torch.Tensor:
        """Generate samples from this distribution"""
        pass
    

class SymmetricDirichlet(torch.nn.Module, Sampleable):
    """Symmetric Dirichlet distribution of d features.
    Args:
        dirich_dim: dimension of the Dirichlet distribution. 
        dirich_param: parameter of the Dirichlet distribution. 
    """
    def __init__(self, dirich_dim:int|None=None, dirich_param:torch.Tensor|None=None):
        super().__init__()

        if dirich_param is not None:
            self.register_buffer("dirich_param", dirich_param.to(dtype=torch.float32))

        if dirich_dim is not None:
            dirich_param = torch.ones(dirich_dim)
            self.register_buffer("dirich_param", dirich_param.to(dtype=torch.float32))

    @property
    def distribution(self) -> Any:
        return torch.distributions.Dirichlet(self.dirich_param)

    def sample(self, n_samples:int) -> torch.Tensor:
        """Generate samples from this distribution, (n_samples, dirich_dim)"""
        return self.distribution.sample([n_samples])



class EmpiricalDistribution(torch.nn.Module, Sampleable):
    """Empirical distribution of d features.
    Args:
        x_obs: observed data, (batch_size, dim)
    """
    def __init__(self, x_obs:torch.Tensor):
        super().__init__()
        self.x_obs = x_obs
    
    @property 
    def distribution(self) -> Any:
        return "Empirical distribution of x_obs"
    
    def sample(self, n_samples:int) -> torch.Tensor:
        """Generate samples from this distribution, (n_samples, dirich_dim)"""
        return torch.stack(random.choices(self.x_obs, k=n_samples), dim=0)


class StandardNormal(torch.nn.Module, Sampleable):
    """Standard multivariate normal distribution with dimension normal_dim"""
    def __init__(self, normal_dim:int):
        super().__init__()
        self.loc = torch.zeros(normal_dim).to(torch.float)
        self.covariance_matrix = torch.eye(normal_dim).to(torch.float)

    @property 
    def distribution(self) -> Any:
        return torch.distributions.MultivariateNormal(self.loc, self.covariance_matrix)
    
    def sample(self, n_samples:int) -> torch.Tensor:
        """Generate samples from this distribution, (n_samples, normal_dim)"""
        return self.distribution.sample([n_samples])






# ----- VAE Architecture ----- #
class VAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int=2, 
                 n_layers:int=3, use_batch_norm:bool=True, use_sza:bool=True):
        """VAE architecture
        Args:
            input_dim: dimension of input tensor
            latent_dim: dimension of latent space
            n_layers: number of hidden layers (from input to latent space)
            use_batch_norm: whether to use batch normaliztion
            use_sza: whether to use SZA function for the output layer
        """
        super().__init__()
        self.latent_dim = latent_dim

        hidden_dims = np.linspace(input_dim, latent_dim, n_layers+1, dtype=int)
        encoder_dims = hidden_dims[:-1]
        encoder_dim_out = encoder_dims[-1]
        decoder_dims = hidden_dims[::-1]
        
        # --- Encoder (n_layers-1) ---
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(nn.Linear(encoder_dims[i], encoder_dims[i+1]))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(encoder_dims[i+1])) # batch normalization
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
    
        # --- Latent space ---
        self.mu = nn.Linear(encoder_dim_out, latent_dim)
        self.log_var = nn.Linear(encoder_dim_out, latent_dim)
        
        # --- Decoder (n_layers) ---
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i+1]))
            if i < len(decoder_dims) - 2: # not imposing BN and ReLU on the final layer
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dims[i+1]))
                decoder_layers.append(nn.ReLU())
            else: # final output layer
                if use_sza:
                    decoder_layers.append(CustomActivation())
        self.decoder = nn.Sequential(*decoder_layers)


    def reparameterize(self, mu:torch.Tensor, log_var:torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*log_var) # standard deviation
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std # latent representation
        return z
    
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)

        return x_recon, z, mu, log_var
    
    def sample(self, n_samples:int) -> torch.Tensor:
        """Generate new samples.
        Args:
            n_samples: number of samples to be generated
        Returns:
            generated samples, (n_samples, input_dim)
        """
        z = StandardNormal(self.latent_dim).sample(n_samples) # (n_samples, latent_dim)
        return self.decoder(z).detach() # (n_samples, input_dim)
        





# ----- GAN Architecture ----- #
class Generator(nn.Module):
    """Generator of GAN
    Args:
        latent_dim: dimension of latent vectors
        output_dim: dimension of output vectors
        n_layers: number of layers in generator network
        use_batch_norm: whether to use batch normaliztion
        use_sza: whether to use SZA function for the output layer
    """
    def __init__(self, latent_dim:int, output_dim:int, 
                 n_layers:int=3, use_batch_norm:bool=True, use_sza:bool=True):
        super().__init__()

        hidden_dims = np.linspace(latent_dim, output_dim, n_layers+1, dtype=int)
    
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i < n_layers-1:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                layers.append(nn.ReLU())
            else: # final output layer
                if use_sza:
                    layers.append(CustomActivation())
        self.model = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        
    def forward(self, z:torch.Tensor) -> torch.Tensor:
        return self.model(z)
    
    def get_latent_dim(self) -> int:
        return self.latent_dim

    def sample(self, n_samples:int) -> torch.Tensor:
        """Generate new samples.
        Args:
            n_samples: number of samples to be generated
        Returns:
            generated samples, (n_samples, output_dim)
        """
        z = StandardNormal(self.latent_dim).sample(n_samples) # (n_samples, latent_dim)
        return self.model(z).detach() # (n_samples, output_dim)


class Discriminator(nn.Module):
    """Discriminator of GAN
    Args:
        intput_dim: dimension of input vectors
        n_layers: number of layers in discriminator network
        dropout_p: dropout probability
    """
    def __init__(self, input_dim:int, n_layers:int=3, 
                 dropout_p:float=0., use_batch_norm:bool=False):
        super().__init__()
        
        hidden_dims = np.linspace(input_dim, 1, n_layers+1, dtype=int)
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i < n_layers-1:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_p))
            else: # final output layer
                layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EmptyModule(nn.Module):
    def forward(self):
        pass






# ----- Flow Architecture ----- #
class ResidualBlock(nn.Module):
    """Residual block module that takes in layer (batch_size, hidden_dim) and 
    time embedding (batch_size, hidden_dim), and outputs layer (batch_size, hidden_dim).
    Args:
        hidden_dim: hidden dimension
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

        self.fc2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)


    def forward(self, x: torch.Tensor, t_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, hidden_dim)
            t_embedding: (batch_size, hidden_dim)
        Returns:
            h+residual: (batch_size, hidden_dim)
        """
        h = self.norm1(x)
        h = self.fc1(h)

        h = h + t_embedding

        h = self.norm2(h)
        h = self.fc2(h)

        return h + x
    


class FlowMatching(nn.Module):
    """Flow matching architecture. The forward pass inputs the current status x (batch_size, input_dim) 
    and time t (batch_size,), and outputs the learned vector field (batch_size, input_dim) at x and t.
    Args:
        input_dim: input dimension (n_features)
        hidden_dim: hidden dimension
        n_resiblocks: number of residual blocks
        time_emb_dim: time embedding dimension
    """
    def __init__(self, input_dim:int, hidden_dim:int=128, n_resiblocks:int=3, time_emb_dim:int=128):
        super().__init__()
        self.input_dim = input_dim
        # Time embedding: (batch_size,) -> (batch_size, time_emb_dim) -> (batch_size, hidden_dim)
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU()
        )
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_resiblocks)
        ])
        # Output layer
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Learn the vector field at status x and time t.
        Args:
            x: (batch_size, input_dim)
            t: (batch_size,)
        Returns:
            learned vector field, (batch_size, input_dim)
        """
        t_embedding = self.t_emb(t) # (batch_size, hidden_dim)
        h = self.input_layer(x) # (batch_size, hidden_dim)

        for block in self.residual_blocks:
            h = block(h, t_embedding)

        h = self.output_norm(h)
        learned_vec_field = self.output_layer(h) # (batch_size, input_dim)
        return learned_vec_field


### ----- ODE class -----

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Returns the drift coefficient of the ODE at state xt and time t"""
        pass

class LearnedODE(ODE):
    """ODE learned by some model
    Args:
        model: e.g., flow matching model    
    """
    def __init__(self, model:nn.Module):
        self.model = model
    
    def drift_coefficient(self, xt:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Return the drift coefficient of the learned ODE
        Args:
            xt: state at time t, (batch_size, dim)
            t: time t, (batch_size,)
        Returns:
            vec_field: learned vector field, (batch_size, dim)
        """
        return self.model(xt, t)



### ----- Simulator class -----

class Simulator(ABC):
    @abstractmethod
    def step(self, xt:torch.Tensor, t:torch.Tensor, dt:torch.Tensor, 
             sigma_t:torch.Tensor|None=None) -> torch.Tensor:
        """Take one simulation step
        Args:
            xt: current state at t, (batch_size, dim)
            t: current time, (batch_size,)
            dt: change in time, (batch_size,)
            sigma_t (optional): noise scaling the Brownian motion, (batch_size,)
        Returns:
            nxt: state at time t+dt, (batch_size, dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x:torch.Tensor, ts:torch.Tensor, 
                 simplex_aware:bool|None, sigmas:torch.Tensor|None=None) -> torch.Tensor:
        """Simulate the state x after ts timesteps
        Args:
            x: initial state at time ts[0], (batch_size, dim)
            ts: time steps, (n_timesteps, batch_size)
            simplex_aware: whether to simulate constrained within probability simplex
            sigmas (optional): noise scaling the Brownian motion, (n_timesteps, batch_size)
        Returns:
            iterated x: final state at time ts[-1], (batch_size, dim)
        """
        for t_idx in range(len(ts)-1):
            t = ts[t_idx,:] # (batch_size,)
            dt = ts[t_idx+1,:] - t # (batch_size,)
            if sigmas is not None:
                sigma_t = sigmas[t_idx,:]  # (batch_size,)
            else:
                sigma_t = None
            x = self.step(x, t, dt, sigma_t) # (batch_size, dim)
            if simplex_aware: # if simplex aware, project the sample onto the simplex
                x = CustomActivation()(x)/100
        return x
    
    @torch.no_grad()
    def simulate_with_trajectory(self, x:torch.Tensor, ts:torch.Tensor, 
                                 simplex_aware:bool|None, sigmas:torch.Tensor|None=None) -> torch.Tensor:
        """Simulate the trajectory of state x over ts timesteps
        Args:
            x: initial state at time ts[0], (batch_size, dim)
            ts: time steps, (n_timesteps, batch_size)
            simplex_aware: whether to simulate constrained within probability simplex
            sigmas (optional): noise scaling the Brownian motion, (n_timesteps, batch_size)
        Returns:
            xs: trajectory of state x over ts, (n_timestpes, batch_size, dim)
        """
        xs = [x.clone()]
        n_timesteps = ts.shape[0]
        for t_idx in tqdm(range(n_timesteps-1)):
            t = ts[t_idx,:]
            dt = ts[t_idx+1,:] - t
            if sigmas is not None:
                sigma_t = sigmas[t_idx,:]  # (batch_size,)
            else:
                sigma_t = None
            x = self.step(x, t, dt, sigma_t)
            if simplex_aware: # if simplex aware, project the sample onto the simplex
                x = CustomActivation()(x)/100
            xs.append(x.clone())
        return torch.stack(xs, dim=0)


class EulerSimulator(Simulator):
    def __init__(self, ode:ODE):
        self.ode = ode 

    def step(self, xt:torch.Tensor, t:torch.Tensor, dt:torch.Tensor,
             sigma_t:torch.Tensor|None=None) -> torch.Tensor:
        """One simulation step using Euler method
        Args:
            xt: current state at t, (batch_size, dim)
            t: current time, (batch_size,)
            dt: change in time, (batch_size,)
            sigma_t: keep in None
        Returns:
            nxt: state at time t+dt, (batch_size, dim)
        """
        assert sigma_t is None
        return xt + self.ode.drift_coefficient(xt, t) * (dt.view(-1, 1))



class FlowSampler():
    """Sampler of flow matching model
    Args:
        model: (trained) flow matching model
        n_samples: number of samples to be generated
        n_steps (optional): number of steps to be used in Euler's method
        ts (optional): time steps, (n_steps,)
        p_init (optional): initial distribution
        simplex_aware: whether to simulate constrained within probability simplex
    """
    def __init__(self, model:FlowMatching, n_samples:int,
                 n_steps:int|None=500,
                 p_init:Sampleable|None=None, 
                 ts:torch.Tensor|None=None,
                 simplex_aware:bool=True):
        
        if p_init is None:
            p_init = SymmetricDirichlet(model.input_dim)
            # p_init = StandardNormal(model.input_dim)

        if ts is None:
            ts = torch.linspace(0, 1, steps = n_steps) # (n_steps,)
        
        self.ts = ts.unsqueeze(-1).expand(-1, n_samples) # (n_steps, n_samples)
        
        self.x0 = p_init.sample(n_samples)
        self.simplex_aware = simplex_aware

        self.ode = LearnedODE(model)
        self.simulator = EulerSimulator(self.ode)

    def simulate(self) -> torch.Tensor:
        """Simulate the state x after ts timesteps"""
        x1 = self.simulator.simulate(self.x0, self.ts, simplex_aware=self.simplex_aware)
        return x1
    
    def simulate_with_trajectory(self) -> torch.Tensor:
        """Simulate the trajectory of state x over ts timesteps"""
        x1 = self.simulator.simulate_with_trajectory(self.x0, self.ts, simplex_aware=self.simplex_aware)
        return x1



### ----- Probability path class -----

class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """Abstract base class for conditional probability paths"""
    def __init__(self, p_init:Sampleable, p_true:Sampleable):
        super().__init__()
        self.p_init = p_init
        self.p_true = p_true
    
    @abstractmethod
    def sample_conditioning_variable(self, n_samples:int) -> torch.Tensor:
        pass 

    @abstractmethod
    def sample_conditional_path(self, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def conditional_vector_field(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        pass 

    @abstractmethod
    def conditional_score(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        pass

class LinearConditionalProbabilityPath(ConditionalProbabilityPath):
    """Linear conditional path from p_init to p_true"""
    def __init__(self, p_init:Sampleable, p_true:Sampleable):
        super().__init__(p_init, p_true)
    
    def sample_conditioning_variable(self, n_samples:int) -> torch.Tensor:
        """Generate samples of the conditioning variable: z ~ p_true(z)
        Args:
            n_samples: number of samples to be generated
        Returns:
            z: (batch_size, dim)
        """
        return self.p_true.sample(n_samples)

    def sample_conditional_path(self, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Generate samples of the conditional distribution p_t(x|z): xt = (1 - t) * x0 + t * z
        Args:
            z: (batch_size, dim)
            t: (batch_size,)
        Returns:
            xt: samples from p_t(x|z), (batch_size, dim)
        """
        x0 = self.p_init.sample(z.shape[0]) # (batch_size, dim)
        t = t.view(-1, 1) # (batch_size, 1)
        xt = (1 - t) * x0 + t * z
        return xt
    
    def conditional_vector_field(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Evaluates the conditional vector field at state xt and time t: u_t(xt|z) = (z - xt) / (1 - t).
        Note: t only defined in [0, 1).
        Args:
            xt: current state, (batch_size, dim)
            z: conditioning variable, same shape as xt
            t: current time, (batch_size,)
        Returns:
            conditional vector field u_t(xt|z), same shape as xt
        """
        t = t.view(-1, 1) # (batch_size, 1)
        cond_vec_field = (z - xt) / (1 - t)
        return cond_vec_field
    
    def conditional_score(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        raise Exception("Conditional score is not available for linear path!")
    





# ----- Diffusion Architecture ----- #
class ScoreMatching(nn.Module):
    """Score matching architecture. 
    The forward pass inputs the current status x (batch_size, input_dim) 
    and time t (batch_size,), and outputs the learned score field (batch_size, input_dim) at x and t.
    Args:
        input_dim: input dimension (n_features)
        hidden_dim: hidden dimension
        n_resiblocks: number of residual blocks
        time_emb_dim: time embedding dimension
    """
    def __init__(self, input_dim:int, hidden_dim:int=128, n_resiblocks:int=3, time_emb_dim:int=128):
        super().__init__()
        self.input_dim = input_dim
        # Time embedding: (batch_size,) -> (batch_size, time_emb_dim) -> (batch_size, hidden_dim)
        self.t_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU()
        )
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_resiblocks)
        ])
        # Output layer
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Learn the score field at status x and time t.
        Args:
            x: (batch_size, input_dim)
            t: (batch_size,)
        Returns:
            learned score field, (batch_size, input_dim)
        """
        t_embedding = self.t_emb(t) # (batch_size, hidden_dim)
        h = self.input_layer(x) # (batch_size, hidden_dim)

        for block in self.residual_blocks:
            h = block(h, t_embedding)

        h = self.output_norm(h)
        learned_score_field = self.output_layer(h) # (batch_size, input_dim)
        return learned_score_field
    

### ----- Coefficient class -----
class Alpha(ABC):
    def __init__(self):
        # Check alpha_0 = 0
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.zeros(1,1)
        ), "Must satisfy alpha_0 = 0"
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1)), torch.ones(1,1)
        ), "Must satisfy alpha_1 = 1"
    
    @abstractmethod
    def __call__(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate alpha_t. Should satisfy: 
        self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            t: (batch_size,)
        Returns:
            alpha_t: (batch_size,)
        """
        pass 

    def dt(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate d{alpha_t}/dt
        Args:
            t: (batch_size,)
        Returns:
            d{alpha_t}/dt: (batch_size,)
        """
        t = t.view(-1,1,1) # (batch_size, 1, 1)
        dt = vmap(jacrev(self))(t) # (batch_size, 1, 1, 1, 1)
        return dt.view(-1)

class Beta(ABC):
    def __init__(self, **kwargs):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.ones(1,1)
        ), "Must satisfy beta_0 = 1"
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1)), torch.zeros(1,1)
        ), "Must satisfy beta_1 = 0"
    
    @abstractmethod 
    def __call__(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate beta_t. Should satisfy:
        self(0.0) = 1.0, self(1.0) = 0.0
        Args:
            t: (batch_size,)
        Returns:
            beta_t: (batch_size,)
        """
        pass 

    def dt(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate d{beta_t}/dt
        Args:
            t: (batch_size,)
        Returns:
            d{beta_t}/dt: (batch_size,)
        """
        t = t.view(-1,1,1) # (batch_size, 1, 1)
        dt = vmap(jacrev(self))(t) # (batch_size, 1, 1, 1, 1)
        return dt.view(-1)

class LinearAlpha(Alpha):
    """alpha_t = t"""
    def __call__(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate alpha_t. Should satisfy: 
        self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            t: (batch_size,)
        Returns:
            alpha_t: (batch_size,)
        """
        return t

    def dt(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate d{alpha_t}/dt
        Args:
            t: (batch_size,)
        Returns:
            d{alpha_t}/dt: (batch_size,)
        """
        return torch.ones_like(t)

class SquareRootBeta(Beta):
    """beta_t = sqrt(1-t^c)"""
    def __call__(self, t:torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluate beta_t. Should satisfy:
        self(0.0) = 1.0, self(1.0) = 0.0
        Args:
            t: (batch_size,)
        Returns:
            beta_t: (batch_size,)
        """
        if len(kwargs) == 1:
            self.c = torch.tensor(kwargs.values())
        else:
            self.c = torch.tensor(1)

        return torch.sqrt(1-t**self.c) 
    def dt(self, t:torch.Tensor) -> torch.Tensor:
        """Evaluate d{beta_t}/dt
        Args:
            t: (batch_size,)
        Returns:
            d{beta_t}/dt: (batch_size,)
        """
        return -self.c * t**(self.c-1) / (2 * torch.sqrt(1-t**self.c) + 1e-3)

class DiminishingBeta(Beta):
    """beta_t = (exp(1-c1*t)-c2) / (exp(1) - c2) where c2 = exp(1-c1)"""
    def __call__(self, t:torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            t: (batch_size,)
            **kwargs: must be c=some float value
        Returns:
            beta_t: (batch_size,)
        """
        if len(kwargs) == 1:
            c1 = torch.tensor(kwargs.values())
        else:
            c1 = torch.tensor(4)
        c2 = torch.exp(1-c1)
        return (torch.exp(1-c1*t) - c2) /\
            (torch.exp(torch.tensor(1)) - c2)
    


### ----- Probability path class -----
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """Gaussian conditional probability path (37):
    p_t(x|z) = N(alpha_t * z, beta_t^2 * I_d)
    """
    def __init__(self, p_init:Sampleable, p_true:Sampleable,
                 alpha:Alpha, beta:Beta):
        super().__init__(p_init, p_true)
        self.alpha = alpha
        self.beta = beta
    
    def sample_conditioning_variable(self, n_samples:int) -> torch.Tensor:
        """Generate samples of the conditioning variable: z ~ p_true(z)
        Args:
            n_samples: number of samples to be generated
        Returns:
            z: (batch_size, dim)
        """
        return self.p_true.sample(n_samples)

    def sample_conditional_path(self, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Generate samples of the conditional distribution 
        p_t(x|z) = N(alpha_t * z, beta_t^2 * I_d)
        Args:
            z: (batch_size, dim)
            t: (batch_size,)
        Returns:
            xt: samples from p_t(x|z), (batch_size, dim)
        """
        return self.alpha(t).unsqueeze(-1)*z + self.beta(t).unsqueeze(-1)*torch.randn_like(z)
    
    def conditional_vector_field(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Evaluate the conditional vector field u_t(xt|z) at state xt and time t.
        Note: t only defined in [0, 1).
        Args:
            xt: current state, (batch_size, dim)
            z: conditioning variable, same shape as xt
            t: current time, (batch_size,)
        Returns:
            conditional vector field u_t(xt|z), same shape as xt
        """
        alpha_t = self.alpha(t).unsqueeze(-1) # (batch_size, 1)
        beta_t = self.beta(t).unsqueeze(-1) # (batch_size, 1)
        dt_alpha_t = self.alpha.dt(t).unsqueeze(-1) # (batch_size, 1)
        dt_beta_t = self.beta.dt(t).unsqueeze(-1) # (batch_size, 1)
        return (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + \
            dt_beta_t / beta_t * xt

    def conditional_score(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """Evaluate the conditional score at state xt and time t.
        Note: Only defined on t in [0, 1).
        Args:
            xt: current state, (batch_size, dim)
            z: conditioning variable, same shape as xt
            t: current time, (batch_size,)
        Returns:
            conditional score, same shape as xt
        """
        alpha_t = self.alpha(t).unsqueeze(-1) # (batch_size, 1)
        beta_t = self.beta(t).unsqueeze(-1) # (batch_size, 1)
        return (z * alpha_t - xt) / beta_t
        # return (z * alpha_t - xt)

### ----- SDE class -----
class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt:torch.Tensor, t:torch.Tensor, sigma_t:torch.Tensor) -> torch.Tensor:
        pass 
    
    @abstractmethod
    def diffusion_coefficient(self, xt:torch.Tensor, t:torch.Tensor, sigma_t:torch.Tensor) -> torch.Tensor:
        pass 

class GaussianConditionalSDE(SDE):
    """SDE of Gaussian conditional probability path according to formula (56)
    Args:
        score_model: score matching model
        alpha: coefficent alpha(t)
        beta: coefficient beta(t)
    """
    def __init__(self, score_model:ScoreMatching, alpha:Alpha, beta:Beta):
        super().__init__()
        self.score_model = score_model
        self.alpha = alpha
        self.beta = beta

    def drift_coefficient(self, xt:torch.Tensor, t:torch.Tensor, sigma_t:torch.Tensor) -> torch.Tensor:
        """Return the drift coefficient according to formula (56)
        Args:
            xt: state at time t, (batch_size, dim)
            t: time t, (batch_size,)
            sigma_t: noise scaling the Brownian motion, (batch_size,)
        Returns:
            drift coefficient, (batch_size, dim)
        """
        sigma_t = sigma_t.unsqueeze(-1) # (batch_size, 1)
        alpha_t = self.alpha(t).unsqueeze(-1) # (batch_size, 1)
        beta_t = self.beta(t).unsqueeze(-1) # (batch_size, 1)
        dt_alpha_t = self.alpha.dt(t).unsqueeze(-1) # (batch_size, 1)
        dt_beta_t = self.beta.dt(t).unsqueeze(-1) # (batch_size, 1)

        return (beta_t**2 * dt_alpha_t / alpha_t - \
                dt_beta_t * beta_t + 0.5 * sigma_t**2) * self.score_model(xt, t) + \
                    dt_alpha_t / alpha_t * xt



    def diffusion_coefficient(self, xt:torch.Tensor, t:torch.Tensor, sigma_t:torch.Tensor) -> torch.Tensor:
        """Return the diffusion coefficient according to formula (56)
        Args:
            xt: state at time t, (batch_size, dim)
            t: time t, (batch_size,)
            sigma_t: noise scaling the Brownian motion, (batch_size,)
        Returns:
            diffusion coefficient, (batch_size, dim)
        """
        sigma_t = sigma_t.unsqueeze(-1).expand(-1, xt.size(-1)) # (batch_size, dim)
        # return sigma_t * torch.randn_like(xt)
        return sigma_t



class LangevinFlowSDE(SDE):
    """SDE integrating a flow model and a score model:
    Args:
        flow_model: flow matching model 
        score_model: score matching model
    """
    def __init__(self, flow_model: FlowMatching, score_model: ScoreMatching):
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
    
    def drift_coefficient(self, xt:torch.Tensor, t:torch.Tensor, sigma_t:torch.Tensor) -> torch.Tensor:
        """Return the drift coefficient:
        vector field by flow model + 0.5 * sigma_t^2 * score field by score model
        Args:
            xt: state at time t, (batch_size, dim)
            t: time t, (batch_size,)
            sigma_t: noise scaling the Brownian motion, (batch_size,)
        Returns:
            drift coefficient, (batch_size, dim)
        """
        sigma_t = sigma_t.unsqueeze(-1) # (batch_size, 1)
        return self.flow_model(xt, t) + 0.5 * sigma_t**2 * self.score_model(xt, t)
    
    def diffusion_coefficient(self, xt:torch.Tensor, t:torch.Tensor, sigma_t:torch.Tensor) -> torch.Tensor:
        """Return the diffusion coefficient: sigma * dW_t
        Args:
            xt: state at time t, (batch_size, dim)
            t: time t, (batch_size,)
            sigma_t: noise scaling the Brownian motion, (batch_size,)
        Returns:
            diffusion coefficient, (batch_size, dim)
        """
        sigma_t = sigma_t.unsqueeze(-1).expand(-1, xt.size(-1)) # (batch_size, dim)
        # return sigma_t * torch.randn_like(xt)
        return sigma_t


### ----- Simulator class -----
class EulerMaruyamaSimulator(Simulator):
    """Euler-Maruyama method"""
    def __init__(self, sde:SDE):
        self.sde = sde 
    
    def step(self, xt:torch.Tensor, t:torch.Tensor, dt:torch.Tensor, 
             sigma_t:torch.Tensor|None=None) -> torch.Tensor:
        """One simulation step using Euler-Maruyama method
        Args:
            xt: current state at t, (batch_size, dim)
            t: current time, (batch_size,)
            dt: change in time, (batch_size,)
            sigma_t: noise scaling the Brownian motion, (batch_size,)
        Returns:
            nxt: state at time t+dt, (batch_size, dim)

        """
        assert sigma_t is not None
        dt = dt.unsqueeze(-1)
        return xt + self.sde.drift_coefficient(xt, t, sigma_t) * dt +\
        self.sde.diffusion_coefficient(xt, t, sigma_t) * torch.sqrt(dt) * torch.randn_like(xt)





class DiffusionSampler():
    """Sampler of diffusion model (flow + score)
    Args:
        score_model: score matching model
        n_samples: number of samples to be generated

        flow_model (optional): flow matching model. If provided, do LangevinFlowSDE().
        sigma (optional): noise scaling the Brownian motion
        n_steps (optional): number of steps to be used in Euler's method
        p_init (optional): initial distribution
        ts (optional): time steps, (n_steps,)
        simplex_aware: whether to simulate constrained within probability simplex
    """
    def __init__(self, score_model: ScoreMatching, n_samples:int, 
                 flow_model:FlowMatching|None=None,
                 sigma:float|torch.Tensor|None=None,
                 n_steps:int|None=None,
                 p_init:Sampleable|None=None, 
                 ts:torch.Tensor|None=None,
                 simplex_aware:bool=True):

        if p_init is None:
            # p_init = SymmetricDirichlet(score_model.input_dim)
            p_init = StandardNormal(score_model.input_dim)
        
        if ts is None:
            ts = torch.linspace(0, 1, steps = n_steps) # (n_steps,)
        
        self.ts = ts.unsqueeze(-1).expand(-1, n_samples) # (n_steps, n_samples)


        if isinstance(sigma, float):
            self.sigmas = torch.fill(torch.ones_like(self.ts), sigma) # (n_steps, n_samples)
        elif isinstance(sigma, torch.Tensor):
            assert len(sigma) == len(ts), "Sigma must have the same length as ts"
            self.sigmas = sigma.unsqueeze(-1).expand(-1, n_samples)
        else:
            self.sigmas = SquareRootBeta()(self.ts) # (n_steps, n_samples)


        self.x0 = p_init.sample(n_samples)
        self.simplex_aware = simplex_aware

        if flow_model is not None:
            self.sde = LangevinFlowSDE(flow_model, score_model)
        else:
            self.sde = GaussianConditionalSDE(score_model, alpha=LinearAlpha(), beta=SquareRootBeta())
        self.simulator = EulerMaruyamaSimulator(self.sde)

    def simulate(self) -> torch.Tensor:
        """Simulate the state x after ts timesteps"""
        x1 = self.simulator.simulate(self.x0, self.ts, self.simplex_aware, self.sigmas)
        return x1
    
    def simulate_with_trajectory(self) -> torch.Tensor:
        """Simulate the trajectory of state x over ts timesteps"""
        x1 = self.simulator.simulate_with_trajectory(self.x0, self.ts, self.simplex_aware, self.sigmas)
        return x1







# ----- Trainer ----- #
class Trainer(ABC):
    """This is the abstract trainer class for all generative models. 
    Each generative model just needs to specify its own get_batch_loss() method.
    """
    def __init__(self, model:nn.Module, device:torch.device, cfg:Dict[str, Any]|None=None,
                 logger:Logger|None=None, train_sampler=None, is_main_process:bool=True):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.train_sampler = train_sampler
        self.is_main_process = is_main_process
    
    def _log(self, message:str):
        """If logger exists, then records to log, otherwise print."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
        
    
    def get_optimizer(self, lr:float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    @abstractmethod
    def get_batch_loss(self, batch_data:torch.Tensor) -> torch.Tensor:
        """Compute loss over one batch"""
        pass

    def get_epoch_loss(self, loader:DataLoader, is_train:bool=True) -> torch.Tensor:
        """Compute loss over one epoch of the given data loader
        Args:
            loader: data loader
            is_train: True if this is a training epoch, False otherwise
        """
        epoch_loss = torch.tensor([0.], device=self.device)
        
        if is_train:
            for idx, batch_data in enumerate(loader):
                batch_data = batch_data.to(self.device)
                self.optimizer.zero_grad()
                loss = self.get_batch_loss(batch_data)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if self.batch_print_freq is not None and idx % self.batch_print_freq == 0:
                    self._log(f"({self.device}) Batch {idx+1} training loss {loss.item():<8.4f}")
        else:
            with torch.no_grad():
                for _, batch_data in enumerate(loader):
                    batch_data = batch_data.to(self.device)
                    loss = self.get_batch_loss(batch_data)
                    epoch_loss += loss.item()
        
        if dist.is_initialized():
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        avg_loss = epoch_loss / len(loader)
        return avg_loss

    def train(self, n_epochs:int,
              train_loader:DataLoader, 
              val_loader:DataLoader|None=None, 
              test_loader:DataLoader|None=None, 
              lr:float=1e-3, 
              val_freq:int=1,
              batch_print_freq:int|None=None) -> Dict[str, Any]:
        """Main training function
        Args:
            n_epochs: number of epochs
            train_loader: data loader of training set
            val_loader (optional): data loader of validation set
            test_loader (optional): data loader of test set
            lr (optional): learning rate
            val_freq (optional): frequency to evaluate the model on the validation set and print out the epoch loss
            batch_print_freq (optional): frequency to print out the batch loss
        Returns:
            a dictionary containing the training, validation, and test loss.
        """
        self.optimizer = self.get_optimizer(lr) # set optimizer attribute
        self.batch_print_freq = batch_print_freq
        self.model.train()
        train_losses, val_losses, test_loss = [], [], []

        for epoch in range(n_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            train_loss = self.get_epoch_loss(train_loader, is_train=True)
            train_losses.append(train_loss.cpu())

            log_message = f"Epoch {epoch+1:<3}/{n_epochs:<3} | " +\
                f"Training Loss: {float(train_loss):<8.4f}"

            if val_loader is not None and epoch % val_freq == 0: # validation set
                self.model.eval()
                val_loss = self.get_epoch_loss(val_loader, is_train=False)
                val_losses.append(val_loss.cpu())
                self.model.train()
                log_message = log_message + f" | Validation Loss: {float(val_loss):<8.4f}"

            if self.is_main_process and epoch % val_freq == 0: # print out epoch loss
                self._log(log_message)

        self.model.eval()

        if test_loader is not None: # test set
            test_loss = self.get_epoch_loss(test_loader, is_train=False).cpu()
            self._log(f'Final test loss: {float(test_loss):.2f}')

        train_losses = torch.cat(train_losses, dim=0) if len(train_losses) > 0 and isinstance(train_losses[0], torch.Tensor) else torch.tensor(train_losses)
        val_losses = torch.cat(val_losses, dim=0) if len(val_losses) > 0 and isinstance(val_losses[0], torch.Tensor) else torch.tensor(val_losses)
        return {'train': train_losses, 'validation': val_losses, 'test': test_loss}


class VAETrainer(Trainer):
    def get_batch_loss(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Compute the ELBO loss over one input batch using VAE model
        Args:
            batch_data: input batch data, (batch_size, d)
        Returns: 
            ELBO loss
        """
        batch_data = batch_data
        batch_data_recon, _, mu, log_var = self.model(batch_data)
        MSE = F.mse_loss(batch_data_recon, batch_data, reduction="sum")
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD


class GANTrainer(Trainer):
    """GAN training is different from other models because it involves 
    two neural networks (generator and discriminator). The initialization 
    __init__() and some methods are modified from the Trainer() class. 
    """
    def __init__(self, generator:Generator, discriminator:Discriminator,
                 device:torch.device, cfg:Dict[str, Any]|None=None,
                 logger:Logger|None=None, train_sampler=None, is_main_process:bool=True):
        Trainer.__init__(self, 
                         EmptyModule(), # this serves no purpose but for valid initialization
                         device, cfg, logger, train_sampler, is_main_process) # initialize these attributes
        # GAN specific neural networks
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = nn.BCELoss()

    def get_gan_optimizer(self, gan_model:nn.Module, lr:float):
        """Get optimizer.
        Args:
            gan_model: can be generator or discriminator
            lr: learning rate
        """
        return torch.optim.Adam(gan_model.parameters(), lr=lr)
    
    def get_batch_loss(self, batch_data:torch.Tensor) -> torch.Tensor:
        return torch.tensor([0])

    def get_gan_epoch_loss(self, loader:DataLoader, is_train:bool=True) -> tuple[torch.Tensor,torch.Tensor]:
        """Compute loss over one epoch of the given data loader
        Args:
            loader: data loader
            is_train: True if this is a training epoch, False otherwise
        """
        epoch_loss_g = torch.tensor([0.], device=self.device)
        epoch_loss_d = torch.tensor([0.], device=self.device)
        latent_dim = self.generator.get_latent_dim()

        if is_train:
            for idx, batch_data in enumerate(loader):
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)

                # label smoothing
                real_labels = torch.full((batch_size,), 0.95, device=self.device) # use 0.95 instead of 1.0
                fake_labels = torch.full((batch_size,), 0.05, device=self.device) # use 0.05 instead of 0.0
                z = torch.randn(batch_size, latent_dim, device=self.device) # (batch_size, latent_dim)
                fake_samples = self.generator(z) # (batch_size, d)


                # training discriminator
                self.opt_d.zero_grad()
                real_output = self.discriminator(batch_data).flatten() # (batch_size,)
                d_loss_real = self.criterion(real_output, real_labels)
                d_loss_real.backward()

                fake_output = self.discriminator(fake_samples.detach()).flatten() # (batch_size,)
                d_loss_fake = self.criterion(fake_output, fake_labels)
                d_loss_fake.backward()
                
                d_loss = d_loss_real + d_loss_fake
                self.opt_d.step()            

                # training generator
                self.opt_g.zero_grad()
                fake_output = self.discriminator(fake_samples).flatten()
                g_loss = self.criterion(fake_output, real_labels)
                g_loss.backward()
                self.opt_g.step()

                epoch_loss_g += g_loss.item()
                epoch_loss_d += d_loss.item()
                if self.batch_print_freq is not None and idx % self.batch_print_freq == 0:
                    self._log(f"({self.device}) Batch {idx+1} " +\
                              f"Generator loss: {g_loss.item():<8.4f}" +\
                              f"Discriminator loss: {d_loss.item():<8.4f}")
        else:
            with torch.no_grad():
                for _, batch_data in enumerate(loader):
                    batch_data = batch_data.to(self.device)
                    batch_size = batch_data.size(0)

                    real_labels = torch.ones(batch_size, device=self.device) # (batch_size,)
                    fake_labels = torch.zeros(batch_size, device=self.device) # (batch_size,)
                    z = torch.randn(batch_size, latent_dim, device=self.device) # (batch_size, latent_dim)
                    fake_samples = self.generator(z) # (batch_size, d)

                    # training discriminator
                    real_output = self.discriminator(batch_data).flatten() # (batch_size,)
                    d_loss_real = self.criterion(real_output, real_labels)

                    fake_output = self.discriminator(fake_samples).flatten() # (batch_size,)
                    d_loss_fake = self.criterion(fake_output, fake_labels)
                    
                    d_loss = d_loss_real + d_loss_fake

                    # training generator
                    g_loss = self.criterion(fake_output, real_labels)

                    epoch_loss_g += g_loss.item()
                    epoch_loss_d += d_loss.item()
                    
        if dist.is_initialized():
            dist.all_reduce(epoch_loss_g, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_loss_d, op=dist.ReduceOp.SUM)
        epoch_loss_g = epoch_loss_g / len(loader)
        epoch_loss_d = epoch_loss_d / len(loader)
        return epoch_loss_g, epoch_loss_d
    

    def train(self, n_epochs:int,
              train_loader:DataLoader, 
              val_loader:DataLoader|None=None, 
              test_loader:DataLoader|None=None, 
              lr:float=1e-4, 
              val_freq:int=1,
              batch_print_freq:int|None=None) -> Dict[str, Any]:
        """Main training function
        Args:
            n_epochs: number of epochs
            train_loader: data loader of training set
            val_loader (optional): data loader of validation set
            test_loader (optional): data loader of test set
            lr (optional): learning rate
            val_freq (optional): frequency to evaluate the model on the validation set and print out the epoch loss
            batch_print_freq (optional): frequency to print out the batch loss
        Returns:
            a dictionary containing the training, validation, and test loss.
        """
        self.opt_g = self.get_gan_optimizer(self.generator, lr=lr) # generator optimizer
        self.opt_d = self.get_gan_optimizer(self.discriminator, lr=lr) # discriminator optimizer
        self.batch_print_freq = batch_print_freq
        self.generator.train()
        self.discriminator.train()
        train_losses_g, val_losses_g, test_loss_g = [], [], []
        train_losses_d, val_losses_d, test_loss_d = [], [], []


        for epoch in range(n_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            train_loss_g, train_loss_d = self.get_gan_epoch_loss(train_loader, is_train=True)
            train_losses_g.append(train_loss_g.cpu())
            train_losses_d.append(train_loss_d.cpu())


            log_message = f"Epoch {epoch+1:<3}/{n_epochs:<3} | " +\
                f"Training generator loss: {float(train_loss_g):<.4f}, " +\
                f"discriminator loss: {float(train_loss_d):<.4f}"
            
            if val_loader is not None and epoch % val_freq == 0: # validation set
                self.generator.eval()
                self.discriminator.eval()
                val_loss_g, val_loss_d = self.get_gan_epoch_loss(val_loader, is_train=False)
                val_losses_g.append(val_loss_g.cpu())
                val_losses_d.append(val_loss_d.cpu())
                self.generator.train()
                self.discriminator.train()
                log_message = log_message + \
                    f"\n {" "*13}| Validation generator loss: {float(val_loss_g):<.4f}, " +\
                    f"discriminator loss: {float(val_loss_d):<.4f}"

            if self.is_main_process and epoch % val_freq == 0: # print out epoch loss
                self._log(log_message)

        self.generator.eval()
        self.discriminator.eval()

        if test_loader is not None: # test set
            test_loss_g, test_loss_d = self.get_gan_epoch_loss(test_loader, is_train=False)
            self._log(f'Final test generator loss: {float(test_loss_g):.4f}, discriminator loss {float(test_loss_d):.4f}')

        train_losses_g = torch.cat(train_losses_g, dim=0) if len(train_losses_g) > 0 and isinstance(train_losses_g[0], torch.Tensor) else torch.tensor(train_losses_g)
        train_losses_d = torch.cat(train_losses_d, dim=0) if len(train_losses_d) > 0 and isinstance(train_losses_d[0], torch.Tensor) else torch.tensor(train_losses_d)
        val_losses_g = torch.cat(val_losses_g, dim=0) if len(val_losses_g) > 0 and isinstance(val_losses_g[0], torch.Tensor) else torch.tensor(val_losses_g)
        val_losses_d = torch.cat(val_losses_d, dim=0) if len(val_losses_d) > 0 and isinstance(val_losses_d[0], torch.Tensor) else torch.tensor(val_losses_d)
        return {'train_g': train_losses_g,
                'train_d':  train_losses_d,
                'validation_g': val_losses_g,
                'validation_d': val_losses_d,
                'test_g': test_loss_g,
                'test_d': test_loss_d}


class FlowTrainer(Trainer):
    def get_batch_loss(self, batch_data:torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss over the input batch using flow matching model
        Args:
            batch_data: (batch_size, dim)
        Returns:
            loss of this batch
        """
        batch_size = batch_data.size(0)
        p_init = SymmetricDirichlet(batch_data.size(1))
        # p_init = StandardNormal(batch_data.size(1))
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
        # p_init = SymmetricDirichlet(batch_data.size(1))
        p_init = StandardNormal(batch_data.size(1))
        p_data = EmpiricalDistribution(batch_data)
        alpha, beta = LinearAlpha(), SquareRootBeta()
        path = GaussianConditionalProbabilityPath(
            p_init, p_data, alpha = alpha, beta = beta
        ).to(self.device)

        z = path.sample_conditioning_variable(batch_size).to(torch.float32) # (batch_size, dim)
        t = torch.rand(batch_size, device=self.device) # (batch_size,)
        xt = path.sample_conditional_path(z, t) # (batch_size, dim)

        score_field_learned = self.model(xt, t) # (batch_size, dim)
        score_field_target = path.conditional_score(xt, z, t) # (batch_size, dim)
        return F.mse_loss(score_field_learned, score_field_target, reduction='mean')
        












