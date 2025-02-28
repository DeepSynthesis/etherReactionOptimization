import pandas as pd
from pathlib import Path
from rdkit import Chem

from summit.domain import CategoricalVariable, ContinuousVariable
from summit import Domain
from summit.utils.dataset import DataSet

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
    def __init__(self, desc_path):
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
