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
from new_edbo import pareto_front_2_dim as pareto_front
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume

import warnings

warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--init_method", type=str, default="LHS")
parser.add_argument("--seed", type=int, default=1145141)
parser.add_argument("--encode", type=str, default="DFT")
args = parser.parse_args()
seed, init_method, batch_size = args.seed, args.init_method, args.batch_size


def run_experiments(data, cand):
    l_cand = len(cand)
    l_data = len(data)
    cand = cand.copy()
    for obj in obj_list:
        cand[(obj, "DATA")] = -1
    cand[("strategy", "METADATA")] = "EDBOplus"
    cand.index = range(l_cand)

    for i in range(l_cand):
        ok_find = False

        for j in range(l_data):
            flag = True
            for x in col_list:
                if data.loc[j, x] != cand.loc[i, x].values[0]:
                    flag = False
                    break
            if flag:
                for obj in obj_list:
                    cand.loc[i, obj] = data.loc[j, obj]
                ok_find = True
                break

        if not ok_find:
            print("error:run_experiments")
            sys.exit(0)

    return cand


if args.encode == "DFT":
    csv_path = "data/suzuki_352_data/dataset_B2_DFT_clean.csv"
elif args.encode == "mordred":
    csv_path = "data/suzuki_352_data/dataset_B3_Mordred_clean.csv"
elif args.encode == "OHE":
    csv_path = "data/suzuki_352_data/dataset_B1_OHE_clean.csv"


df_exp = pd.read_csv(csv_path)

if args.encode == "OHE":
    df_exp.drop(columns=["Unnamed: 0"], inplace=True)

domain = Domain()
obj_list = ["objective_conversion", "objective_selectivity"]
obj_maximize = [True, True]
key_list = ["ligand", "base", "solvent"]

cont_list = []

ref_point = []
all_points = df_exp[obj_list].copy()

for obj, is_max in zip(obj_list, obj_maximize):
    if is_max:
        ref_point_i = df_exp[obj].min()
    else:
        ref_point_i = -df_exp[obj].max()
        all_points[obj] = -all_points[obj]
    ref_point.append(ref_point_i)


pareto_all = pareto_front(all_points.to_numpy())


for i in key_list:
    df_i_col = [col for col in df_exp.columns if i in col]
    df_i = df_exp[df_i_col].copy()

    df_i.drop_duplicates(keep="first", inplace=True)
    col_i = df_i.columns
    smiles_list = list(df_i[i])
    df_i = df_i.select_dtypes(include=["number"])
    df_i.index = smiles_list
    df_i = DataSet.from_df(df_i)
    domain += CategoricalVariable(
        name=i, description=i, levels=smiles_list, descriptors=df_i
    )

for cont in cont_list:
    lis_i = list(set(df_exp[cont]))
    df_i = pd.DataFrame()
    df_i.index = lis_i
    df_i[cont] = lis_i
    df_i = DataSet.from_df(df_i)
    domain += CategoricalVariable(
        name=cont, description=cont, levels=lis_i, descriptors=df_i
    )

col_list = key_list + cont_list

for obj, is_max in zip(obj_list, obj_maximize):
    domain += ContinuousVariable(
        name=obj,
        description=obj,
        bounds=[-100, 100],
        is_objective=True,
        maximize=is_max,
    )

prev_res = None

hv = Hypervolume(ref_point=torch.Tensor(ref_point))
area_all = hv.compute(pareto_Y=torch.Tensor(pareto_all))

budget = 30

strategy = newEDBO(domain, seed=args.seed, init_sampling_method=init_method)
num_epochs = int(budget / batch_size)
hv_list = []
for epoch in range(num_epochs):
    next_points = strategy.suggest_experiments(prev_res=prev_res, batch_size=batch_size)
    next_res = run_experiments(df_exp, next_points)
    prev_res = DataSet(pd.concat([prev_res, next_res], axis=0))
    prev_res.index = range(len(prev_res))

    if (epoch + 1) * batch_size != len(prev_res):
        print("number error")
        sys.exit(0)

    now_points = prev_res[obj_list].copy().to_numpy()
    for i, is_max in enumerate(obj_maximize):
        if not is_max:
            now_points[:, i] = -now_points[:, i]
    now_pareto = pareto_front(now_points)
    hypervolume = hv.compute(pareto_Y=torch.Tensor(now_pareto))
    normalized_hv = hypervolume / area_all
    hv_list.append(normalized_hv)
    print("epoch :\n", epoch, "; hv :", normalized_hv)

print(hv_list)


def to_txt(res, file):
    g = open(file, "a+")
    rxn = np.array(res)
    out = ""
    for j in rxn:
        out = out + str(j) + " "
    print("out :", out)
    g.write(out)
    g.write("\n")
    g.flush()


# to_txt(hv_list, f"352_result/{init_method}_{batch_size}_{args.encode}.txt")
