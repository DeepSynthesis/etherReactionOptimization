import numpy as np
import pandas as pd
import os
import copy
import sys
from summit.benchmarks.experimental_emulator import *
from summit.strategies import LHS
from summit import Runner
from summit.utils.dataset import DataSet
from summit.domain import *
from new_edbo import newEDBO
import random
import json

import warnings

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--seed", type=int, default=1145141)
args = parser.parse_args()
seed, batch_size = args.seed, args.batch_size

domain = Domain()
obj_list = ["yld", "ee"]
variable_list = [
    "amine_smiles",
    "cobalt_smiles",
    "oxidant_smiles",
    "solvent_smiles",
    "additive_smiles",
]
key_list = ["amine", "cobalt", "oxidant", "solvent", "additive"]

for x in key_list:
    data_v = pd.read_csv(f"data/106560_data/{x}_encoding.csv")
    lev_v = list(data_v[f"{x}_smiles"])

    data_v.drop(columns=["Unnamed: 0", f"{x}_smiles"], inplace=True)
    data_v.index = lev_v
    data_v = DataSet.from_df(data_v)

    domain += CategoricalVariable(
        name=x, description=x, levels=lev_v, descriptors=data_v
    )

for obj in obj_list:
    domain += ContinuousVariable(
        name=obj,
        description=obj,
        bounds=[0, 100],
        is_objective=True,
        maximize=True,
    )

prev_res = pd.read_csv("data/106560_data/prev_res.csv")
prev_res.drop(columns=["Unnamed: 0"], inplace=True)
prev_res = DataSet.from_df(prev_res)

import time

last_time = time.time()

strategy = newEDBO(domain)
next_points = strategy.suggest_experiments(prev_res=prev_res, batch_size=batch_size)
next_time = time.time()

print("time :", next_time - last_time)
print(next_points)
