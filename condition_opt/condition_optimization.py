from datetime import datetime
import pandas as pd
from pathlib import Path
from rdkit import Chem

from summit.domain import CategoricalVariable, ContinuousVariable
from summit import Domain
from summit.utils.dataset import DataSet

from EDBOplus.edbo import newEDBO

desc_path = Path(__file__).parent / Path("../descriptors")


def canonicalize_smiles(smiles):
    if smiles == "blanck_cell":
        return smiles
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        return smiles


def get_reaction_space(domain, desc_class, reagent_types):
    """Get reaction space from descriptor csv file"""
    desc_dict = desc_class.get_desc_df()
    for tp in reagent_types:
        smiles_list = desc_dict[tp].index.tolist()
        descriptor_list = DataSet.from_df(desc_dict[tp])
        domain += CategoricalVariable(name=tp, description=tp, levels=smiles_list, descriptors=descriptor_list)

    return domain


def get_target_value(domain):
    """Get target value by index"""
    domain += ContinuousVariable(name="yld", description="yld", bounds=[0, 100], is_objective=True, maximize=True)
    domain += ContinuousVariable(name="ee", description="ee", bounds=[-100, 100], is_objective=True, maximize=True)

    return domain


# descriptor mapping
class descClass:
    def __init__(self):
        self.alkali_desc = pd.read_csv(desc_path / Path("alkali_desc_datadf.csv"), index_col=0)
        self.amine_desc = pd.read_csv(desc_path / Path("amine_total_desc_datadf.csv"), index_col=0)
        self.cobalt_desc = pd.read_csv(desc_path / Path("cobalt_cat_desc_datadf.csv"), index_col=0)
        self.oxidant_desc = pd.read_csv(desc_path / Path("oxidant_desc_datadf.csv"), index_col=0)
        self.solvent_desc = pd.read_csv(desc_path / Path("solvent_desc_datadf.csv"), index_col=0)

        self.alkali_desc.index = self.alkali_desc.index.map(canonicalize_smiles)
        self.amine_desc.index = self.amine_desc.index.map(canonicalize_smiles)
        self.oxidant_desc.index = self.oxidant_desc.index.map(canonicalize_smiles)
        self.solvent_desc.index = self.solvent_desc.index.map(canonicalize_smiles)

        space_size = 1
        space_size *= self.alkali_desc.shape[0]
        space_size *= self.amine_desc.shape[0]
        space_size *= self.cobalt_desc.shape[0]
        space_size *= self.oxidant_desc.shape[0]
        space_size *= self.solvent_desc.shape[0]

        print(f"Total Reaction Space Size: {space_size}")

    def map_desc(self, mol_type, data_df):
        if mol_type == "alkali":
            desc_df = self.alkali_desc
        elif mol_type == "amine":
            desc_df = self.amine_desc
        elif mol_type == "cobalt":
            desc_df = self.cobalt_desc
        elif mol_type == "oxidant":
            desc_df = self.oxidant_desc
        elif mol_type == "solvent":
            desc_df = self.solvent_desc
        else:
            raise ValueError("Invalid mol_type")

        desc_list = []
        for _, smiles in data_df.items():
            desc_list.append(desc_df.loc[smiles,])
        return pd.DataFrame(desc_list, index=data_df.index)

    def get_desc_df(self):
        return {
            "alkali": self.alkali_desc,
            "amine": self.amine_desc,
            "cobalt": self.cobalt_desc,
            "oxidant": self.oxidant_desc,
            "solvent": self.solvent_desc,
        }


if __name__ == "__main__":
    bayesian_opt_round = 3
    use_old_df = False

    data_path = Path(__file__).parent
    # Read data
    if use_old_df:
        data_df = pd.read_csv(data_path / Path("manual_conditions_cleaned.csv"))
        data_df = data_df[data_df["select_tag"] == True]

    data_df_new = pd.read_csv(data_path / Path(f"opt_round_{bayesian_opt_round}/manual_conditions_new.csv"))
    data_df_new = data_df_new[data_df_new["select_tag"] == True]

    data_df = pd.concat([data_df, data_df_new]) if use_old_df else data_df_new
    new_batch_id = data_df["batch_id"].max() + 1
    reagent_types = ["amine", "cobalt", "oxidant", "alkali", "solvent"]

    # generate reaction space
    desc_class = descClass()
    domain = Domain()
    domain = get_reaction_space(domain, desc_class, reagent_types=reagent_types)

    # generate target columns
    domain = get_target_value(domain)

    # remove ion of alkali
    print(data_df["alkali_smiles"])
    data_df["alkali_smiles"] = data_df["alkali_smiles"].apply(lambda x: x.split(".")[0])

    # canonicalize smiles in data_df except for cobalt.
    for tp in reagent_types:
        if tp != "cobalt":
            data_df[f"{tp}_smiles"] = data_df[f"{tp}_smiles"].apply(canonicalize_smiles)

    # process for done experments datas
    done_dataset = data_df.iloc[:, 1:]
    done_dataset.columns = reagent_types + ["yld", "ee", "select_tag"]

    # check if all batch molecules in reaction space
    desc_dict = desc_class.get_desc_df()

    print(done_dataset)
    done_dataset.reset_index(inplace=True, drop=True)
    for i in done_dataset.index:
        for tp in reagent_types:
            if done_dataset.loc[i, tp] not in desc_dict[tp].index:
                print(f"Invalid {tp} molecule: {done_dataset.loc[i, tp]}")
                done_dataset.loc[i, "select_tag"] = False
    # print(done_dataset)
    # done_dataset.to_csv("manual_conditions_matched.csv")
    print(done_dataset["select_tag"].value_counts())
    done_dataset = done_dataset[done_dataset["select_tag"]]
    done_dataset = done_dataset.drop(columns=["select_tag"])

    done_dataset.reset_index(inplace=True)
    done_dataset = DataSet.from_df(done_dataset)

    # proceed EDBO
    edbo = newEDBO(domain=domain, seed=42, init_sampling_method="LHS")
    alkali_range = [
        "O=C([O-])O",
        "O=C([O-])[O-]",
        "O=P([O-])(O)O",
        "O=P([O-])([O-])[O-]",
        "C1CN2CCN1CC2",
        "CN(C)c1ccncc1",
        "c1ccncc1",
        "C1CCC2=NCCCN2CC1",
        "O=C(O)c1ccccc1",
        "CC(=O)O"
    ]
    # edbo_restults = edbo.suggest_experiments_new(prev_res=done_dataset, batch_size=5, alkali_range=alkali_range)
    edbo_restults = edbo.suggest_experiments(prev_res=done_dataset, batch_size=5)

    # save results
    formatted_date = datetime.now().strftime("%Y%m%d")  # get data with format 'yyyymmdd'
    edbo_restults.to_csv(data_path / Path(f"opt_round_{bayesian_opt_round}/edbo-results_batch-{new_batch_id}_{formatted_date}.csv"))
