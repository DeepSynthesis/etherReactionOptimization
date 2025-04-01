from pathlib import Path
import subprocess
import pandas as pd
from wfnDescGet.wfn_desc_get import get_all_wfn_descriptors


mol_type = "additives"
folder_type = "additives"
data_df = pd.read_csv(f"index_info/{mol_type}.csv")
fchk_files = (Path(__file__).parent / Path(f"Gauss_Calcs/{folder_type}/output_fchk")).glob("*.fchk")
atoms_list = data_df["atoms_list"]
# bond_list = [[0, 1], [1, 2], [2, 3]]
# angle_list = [[0, 1, 2], [1, 2, 3]]
# dihedral_list = [[0, 1, 2, 3]]
prop = data_df["prop"].tolist()

df = get_all_wfn_descriptors(
    fchk_files=fchk_files,
    atoms_list=atoms_list,
    # bond_list=bond_list,
    # angle_list=angle_list,
    # dihedral_list=dihedral_list,
    prop=prop,
    kill=False,
    file_mode=1,
    fchk_files_sort=True,
)
df.to_csv(f"{mol_type}_desc_datadf.csv")
