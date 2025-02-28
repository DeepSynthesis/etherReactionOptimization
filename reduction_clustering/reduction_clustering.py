from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from tqdm import tqdm
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from .utils import generate_rdkit_descriptors, normalize_columns, reduction_clustering, get_best_n_clusters, batch_draw_mol





def get_cluster_and_reduction_results(
    input_csv, smiles_column, reduction_type, set_level, n_clusters, decomp_name, cluster_name, n_clusters_determined, recalc_desc
):
    if recalc_desc:
        generate_rdkit_descriptors(input_csv, smiles_column)

    input_df = pd.read_csv(input_csv)
    if reduction_type == "alcohols":
        if set_level == 0:
            input_df = input_df[input_df["level"] < 3].reset_index()
        else:
            input_df = input_df[input_df["level"] == set_level].reset_index()
    desc_file = pd.read_csv(Path(__file__).parent / Path(f"data/all_{reduction_type}_in_hspoc_desc.csv"), index_col="entry")
    # desc_file = pd.read_csv(
    #     Path(__file__).parent / Path(f"data/all_{reduction_type}_in_strict_mode_rdkit_descriptors.csv"), index_col="SMILES"
    # )
    # desc_file = desc_file.drop(columns=extreme_desc)
    desc_file = desc_file[~desc_file.index.duplicated(keep="first")]
    logger.info(f"Data not be calculated: {input_df[~input_df['SMILES'].isin(desc_file.index)]['SMILES'].tolist()}")
    input_df = input_df[input_df["SMILES"].isin(desc_file.index)].reset_index()
    data_X = desc_file.loc[input_df["SMILES"], :]
    data_X = np.array(data_X)
    data_X = np.nan_to_num(data_X)
    data_X = normalize_columns(data_X)

    # calculate mean and std_dev
    means = np.mean(data_X, axis=0)
    std_devs = np.std(data_X, axis=0)
    # calculatie Z-scores
    std_devs[std_devs == 0] = 1e-10
    z_scores = (data_X - means) / std_devs
    extreme_values = np.abs(z_scores) > 10
    extreme_values = np.sum(extreme_values, axis=0)
    extreme_data = pd.DataFrame(extreme_values, index=desc_file.columns, columns=["number"])
    # extreme_data.to_csv("extreme_data.csv")
    zero_extreme_list = extreme_data[extreme_data["number"] == 0].index.tolist()
    cleaned_desc = desc_file[zero_extreme_list]

    data_X = cleaned_desc.loc[input_df["SMILES"], :]
    data_X = np.array(data_X)
    data_X = np.nan_to_num(data_X)
    data_X = normalize_columns(data_X)
    data_X = data_X[:, np.any(data_X != 0, axis=0)]
    logger.info(data_X.shape)

    if n_clusters_determined != "pass":
        get_best_n_clusters(data_X=data_X, method=n_clusters_determined)
        return None
    else:
        decomp_data = reduction_clustering(data_X=data_X, decomp_name=decomp_name, cluster_name=cluster_name, n_clusters=n_clusters)
        decomp_data = pd.concat([input_df[["SMILES", "level", "Exp_State"]], decomp_data], axis=1)
        # decomp_data.to_csv(Path(__file__).parent / Path(f"{reduction_type}_results_structure/level_{set_level}_decomped.csv"))
        # all_smiles = decomp_data.groupby("label")

        # return all_smiles
        return decomp_data

if __name__ == "__main__":
    opts = DrawingOptions()
    opts.includeAtomNumbers = True
    opts.bondLineWidth = 2.8

    reduction_type = "alcohol"
    input_csv = Path(__file__).parent / Path(f"data/all_{reduction_type}_in_strict_mode.csv") 
    smiles_column = "SMILES" 
    recalc_desc = False
    set_level = 0
    draw_pictures = True
    n_clusters = 6
    decomp_name = "UMAP"
    cluster_name = "kmeans-with-agglomeration"
    n_clusters_determined = "pass"  # elbow, silhouette, pass
    opt = False
    decomp_data = get_cluster_and_reduction_results(
        input_csv=input_csv,
        smiles_column=smiles_column,
        set_level=set_level,
        n_clusters=n_clusters,
        decomp_name=decomp_name,
        cluster_name=cluster_name,
        n_clusters_determined=n_clusters_determined,
        recalc_desc=recalc_desc,
    )
    all_smiles = decomp_data.groupby("label")
    
    reacted_info = []
    if all_smiles is None:
        print("No data to process")
        exit()

    for i, gp in all_smiles:
        reacted_info.append(
            pd.DataFrame(
                {
                    "label": [i],
                    "Reacted_Num": [len(gp[gp["Exp_State"] == "React"])],
                    "No_Reacted_Num": [len(gp[gp["Exp_State"] == "No_React"])],
                    "Total": [len(gp)],
                }
            )
        )
        if len(gp[gp["Exp_State"] != "No_Exp"]) > 0:
            if draw_pictures:
                batch_draw_mol(
                    gp[gp["Exp_State"] != "No_Exp"],
                    "SMILES",
                    Path(__file__).parent / Path(f"{reduction_type}_results_structure/category_for_level_{set_level}_{i}_done"),
                    clean_origin=True,
                )
        print(f"All_data: {len(gp)}")
        if draw_pictures:
            batch_draw_mol(
                gp,
                "SMILES",
                Path(__file__).parent / Path(f"{reduction_type}_results_structure/category_for_level_{set_level}_{i}"),
                clean_origin=True,
            )
        pd.concat(reacted_info).to_csv(f"{reduction_type}_results_structure/reacted_info_for_level_{set_level}.csv", index=False)
