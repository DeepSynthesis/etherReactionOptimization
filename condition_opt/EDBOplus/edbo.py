from summit import *
from scipy.spatial.distance import cdist
import copy
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from tqdm import tqdm
from botorch.models import SingleTaskGP, MixedSingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from sklearn.preprocessing import MinMaxScaler
from idaes.surrogate.pysmo.sampling import CVTSampling, LatinHypercubeSampling
import numpy as np
import random
import torch
import pandas as pd
import sys

from .constants import tkwargs
from .utils import model_likelihood
from .pareto import pareto_front_2_dim
from .scaler import StandardScaler

class newEDBO:
    def __init__(
        self,
        domain: Domain,
        sobol_num_samples: int = None,
        seed: int = 19260817,
        init_sampling_method: str = "CVT",
    ):
        self.domain = domain
        self.all_combos = self.domain.get_categorical_combinations()
        self.all_combos.index = range(len(self.all_combos))
        if sobol_num_samples is None:
            if len(self.all_combos) > 500000:
                self.sobol_num_samples = 128
            elif len(self.all_combos) > 100000:
                self.sobol_num_samples = 256
            elif len(self.all_combos) > 20000:
                self.sobol_num_samples = 512
            else:
                self.sobol_num_samples = 1024
        else:
            self.sobol_num_samples = sobol_num_samples
        self.seed = seed
        self.init_sampling_method = init_sampling_method
        assert init_sampling_method in ["CVT", "LHS"]

        l_all = len(self.all_combos)
        all_x = []
        for i in tqdm(range(l_all)):
            test_x_i = []
            for v in domain.input_variables:
                key_v = self.all_combos.loc[i, v.name].values[0]
                encode = list(v.ds.loc[key_v])
                test_x_i.extend(encode)
            all_x.append(test_x_i)
        self.all_x = np.array(all_x).astype(float)

    def suggest_experiments(self, prev_res: DataSet = None, batch_size: int = 5):
        return self._suggest_experiments_impl(prev_res, batch_size)

    def suggest_experiments_new(self, prev_res: DataSet = None, batch_size: int = 5, alkali_range=None):
        if alkali_range is not None:
            valid_combos = self.all_combos[self.all_combos["alkali"].isin(alkali_range)]
            valid_x = self.all_x[self.all_combos["alkali"].isin(alkali_range)].copy()
        else:
            valid_combos = self.all_combos.copy()
            valid_x = self.all_x.copy()
        
        return self._suggest_experiments_impl(prev_res, batch_size, valid_combos, valid_x)

    def _suggest_experiments_impl(self, prev_res=None, batch_size=5, valid_combos=None, valid_x=None):
        # Initialize valid_combos and valid_x if not provided
        if valid_combos is None:
            valid_combos = self.all_combos.copy()
        if valid_x is None:
            valid_x = self.all_x.copy()
        
        # Initial sampling if no previous results
        if prev_res is None or len(prev_res) == 0:
            return self._initial_sampling(batch_size, valid_combos, valid_x)
        
        # Prepare training data
        domain = self.domain
        col_x = [v.name for v in domain.input_variables]
        col_y = [v.name for v in domain.output_variables]
        num_obj = len(col_y)
        
        # Process training data
        train_y, train_x = self._prepare_training_data(prev_res, domain, col_y, col_x)
        
        # Scale data
        scaler_x, scaler_y = MinMaxScaler(), StandardScaler()
        train_x = scaler_x.fit_transform(train_x)
        train_y = scaler_y.fit_transform(train_y)
        
        # Get Pareto front
        pareto_y = pareto_front_2_dim(train_y)
        
        # Set reference point
        ref_mins = np.min(train_y, axis=0)
        ref_point = scaler_y.transform([ref_mins])[0]
        
        # Build models
        models = []
        for i in range(num_obj):
            train_x_i = torch.tensor(train_x).to(**tkwargs).double()
            train_y_i = train_y[:, i]
            train_y_i = np.atleast_2d(train_y_i).reshape(len(train_y_i), -1)
            train_y_i = torch.tensor(train_y_i.tolist()).to(**tkwargs).double()

            gp, likelihood = model_likelihood(train_x=train_x_i, train_y=train_y_i)
            model_i = SingleTaskGP(
                train_X=train_x_i,
                train_Y=train_y_i,
                covar_module=gp.covar_module,
                likelihood=likelihood,
            )
            models.append(model_i)
        
        bigmodel = ModelListGP(*models)
        
        # Prepare candidate points
        prev_comb = prev_res[col_x].copy()
        combos = pd.concat([valid_combos, prev_comb]).reset_index()
        combos.index = range(len(combos))
        combos.drop_duplicates(keep=False, inplace=True)
        combos_index = combos.index
        
        # Setup acquisition function
        sampler = SobolQMCNormalSampler(torch.Size([self.sobol_num_samples]), seed=1145141)
        partitioning = NondominatedPartitioning(
            ref_point=torch.tensor(ref_point).float(), 
            Y=torch.tensor(pareto_y).float()
        )
        
        EHVI = qExpectedHypervolumeImprovement(
            model=bigmodel,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )
        
        # Optimize acquisition function
        test_x = valid_x.copy()
        test_x = torch.tensor(scaler_x.transform(test_x)).double().to(**tkwargs)
        acq_result = optimize_acqf_discrete(
            acq_function=EHVI, 
            choices=test_x, 
            q=batch_size, 
            unique=True
        )
        
        # Get best samples
        best_samples = acq_result[0].detach().cpu().numpy()
        ans_indexs = []
        for sample in best_samples:
            d_i = cdist([sample], test_x, metric="cityblock")
            a = np.argmin(d_i)
            ans_indexs.append(a)
        
        return valid_combos.iloc[ans_indexs]

    def _initial_sampling(self, batch_size, valid_combos, valid_x):
        k = batch_size
        if self.init_sampling_method == "LHS":
            random_state = np.random.RandomState(self.seed)
            lhs = LHS(self.domain, random_state=random_state)
            return lhs.suggest_experiments(k)
        else:
            df_sampling = valid_x
            idaes = CVTSampling(df_sampling, number_of_samples=k, sampling_type="selection")
            samples = idaes.sample_points()
            
            init_indexs = []
            for sample in samples:
                d_i = cdist([sample], df_sampling, metric="cityblock")
                a = np.argmin(d_i)
                init_indexs.append(a)
            
            if len(init_indexs) < k:
                rand_choices = k - len(init_indexs)
                others = [i for i in range(len(valid_x)) if i not in init_indexs]
                np.random.seed(self.seed)
                random.seed(self.seed)
                init_indexs.extend(np.random.choice(others, rand_choices))
                init_indexs = sorted(init_indexs)
            
            return valid_combos.iloc[init_indexs].copy()

    def _prepare_training_data(self, prev_res, domain, col_y, col_x):
        train_y = prev_res[col_y].to_numpy().astype(float)
        
        # Adjust objectives based on maximize flag
        for i, v in enumerate(domain.output_variables):
            if not v.maximize:
                train_y[:, i] = -train_y[:, i]
        
        # Prepare input features
        train_x = []
        len_prev = len(prev_res)
        for i in range(len_prev):
            now_x = []
            for v in domain.input_variables:
                key_v = prev_res.loc[i, v.name].values[0]
                encode = list(v.ds.loc[key_v])
                now_x.extend(encode)
            train_x.append(now_x)
        
        return train_y, np.array(train_x).astype(float)