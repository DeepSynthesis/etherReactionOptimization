import gpytorch
import torch
from tqdm import tqdm
from .constants import tkwargs
import random
import numpy as np 
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.constraints import GreaterThan

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                ard_num_dims=np.shape(train_x)[1],
                lengthscale_prior=GammaPrior(2.0, 0.2),
            ),
            outputscale_prior=GammaPrior(5.0, 0.5),
        )
        self.covar_module.base_kernel.lengthscale = 5.0

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def model_likelihood(train_x, train_y):
    print("build and optimize model for a variable.")

    seed = 1145141

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(GammaPrior(1.5, 0.1))
    likelihood.noise = 5.0
    likelihood.train()

    model = GPModel(train_x, train_y, likelihood).to(**tkwargs)
    model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
    model.train()

    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    num_epochs = 1000
    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.squeeze(-1).to(**tkwargs))
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood

if __name__ == "__main__":
    model_likelihood(np.array([[1, 1], [1,1]]), np.array([1, 1]))