import pandas as pd
from rdkit import Chem
from pathlib import Path


def check_smiles(data_df):
    for i in range(len(data_df)):
        try:
            mol = Chem.MolFromSmiles(data_df.loc[i, "Smiles"])
            assert mol != None
            data_df.loc[i, "Tag"] = 1
        except:
            print(f"Error smiles in line {i}")
            data_df.loc[i, "Tag"] = 0

    return data_df[data_df["Tag"] == 1]

def manual_delete(x):
    manual_delete_list = [
        "CCCCCCCCCCCCCCCCCCOC(=O)CCc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1",
    ]
    if x["Smiles"] in manual_delete_list:
        x["Tag"] = False

    return x


def deal_with_more_than_one_part_smiles(x):
    remove_acid_frags = ["Cl", "O", "O=S(=O)(O)O"]
    remove_ion_frags = ["[Na+]", "[K+]", "[NH4+]", "[Li+]"]
    error_parts = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "2H", "Sn"]
    if "." in x["Smiles"]:
        s_frags = x["Smiles"].split(".")
        # Case 1: Molecules with ligands, such as Cl or others, remove Cl, etc.
        s_frags = list(filter(lambda x: x not in remove_acid_frags, s_frags))
        # Case 2: Exists in the form of basic molecules, with [Na+], [K+], etc.
        s_frags = list(filter(lambda x: x not in remove_ion_frags, s_frags))
        x["Smiles"] = ".".join(s_frags)
    if "." in x["Smiles"]:
        x["Tag"] = False
    else:
        x["Smiles"].replace("[O-]", "O")
    for e in error_parts:
        if e in x["Smiles"]:
            x["Tag"] = False
            break
    return x


def canonical_smiles(x):
    try:
        x["Smiles"] = Chem.MolToSmiles(Chem.MolFromSmiles(x["Smiles"]), isomericSmiles=True)
        return x

    except:
        raise Exception(f"Error in smiles: {x['Smiles']}")


def has_alcohol(x):
    """Check if the structure has an OH group and exclude COOH and enol structures"""
    mol = Chem.MolFromSmiles(x["Smiles"])

    # Check if there is a hydroxyl group in the molecule
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1:
            x["Tag"] = True
            return x

    x["Tag"] = False
    return x


def get_oh_environment(mol, radius):
"""get alcohol chemical environment"""
    oh_environment = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1:  # Find hydroxyl atom
            env_atoms = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx())
            neighbors = set()
            for env_atom_idx in env_atoms:
                env_atom = mol.GetAtomWithIdx(env_atom_idx)
                neighbors.add((env_atom.GetSymbol(), env_atom.GetIdx()))  # Add neighboring atom's symbol and index to the set
            environment = (atom.GetSymbol(), tuple(sorted(neighbors)))  # Hydroxyl atom and its neighboring atoms' symbols and indices form a tuple
            oh_environment.append(environment)
    return oh_environment


def has_multiple_OH(x):
    mol = Chem.MolFromSmiles(x["Smiles"])
    environment_list = get_oh_environment(mol, radius=1)
    if len(environment_list) > 1:
        x["Tag"] = False
    return x


def pyridine_N_check(atom):
    return atom.GetIsAromatic()


def amide_N_check(mol, atom):
    neighbors = atom.GetNeighbors()
    for neighbor in neighbors:
        # Check if the neighboring atom is carbon and if it forms a double bond with the current atom
        if neighbor.GetSymbol() == "C" and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
            # Check if the carbon atom is connected to an oxygen atom
            c_neighbors = neighbor.GetNeighbors()
            for c_neighbor in c_neighbors:
                if c_neighbor.GetSymbol() == "O":
                    return True
    return False


def nitro_N_check(mol, atom):
    neighbors = atom.GetNeighbors()
    for neighbor in neighbors:
        if neighbor.GetSymbol() == "O" and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def cyanide_N_check(mol, atom):
    neighbors = atom.GetNeighbors()
    for neighbor in neighbors:
        if neighbor.GetSymbol() == "C" and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.TRIPLE:
            return True
    return False


def remove_nucleo_atoms(x, strict_mode=False):
    mol = Chem.MolFromSmiles(x["Smiles"])
    for atom in mol.GetAtoms():
        if strict_mode:
            if atom.GetSymbol() == "N" and atom.GetDegree() < 4:
                # Only include aliphatic amines, exclude pyridine N, amide N, nitro N, and C≡N
                if pyridine_N_check(atom):  # Check pyridine N
                    continue
                if amide_N_check(mol, atom):  # Check amide
                    continue
                if nitro_N_check(mol, atom):  # Check nitro N
                    continue
                if cyanide_N_check(mol, atom):  # Check cyanide
                    continue
                x["Tag"] = "N_nuc"  # Exclude nucleophilic reagents with N
            elif atom.GetSymbol() == "S" and atom.GetDegree() < 2:
                x["Tag"] = "S_nuc"  # Exclude nucleophilic reagents with SH, do not exclude thioether structures
            elif atom.GetSymbol() == "P" and atom.GetDegree() < 4:
                x["Tag"] = "P_nuc"  # Exclude nucleophilic reagents with P
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == "C":
                    for bond in neighbor.GetBonds():
                        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            x["Tag"] = "COOH_nuc"  # Exclude carboxylic acid types
    return x


def distinguish_alcohol_level(x):
    mol = Chem.MolFromSmiles(x["Smiles"])
    alcohols = [(atom, atom.GetIdx()) for atom in mol.GetAtoms() if atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0]
    try:
        assert len(alcohols) == 1
    except:
        print(x["Smiles"], alcohols)

    all_class_info = []
    for oxygen in alcohols:
        for neighbor in oxygen[0].GetNeighbors():
            if neighbor.GetSymbol() == "C":  # Check if the neighbor is a carbon atom
                num_bonds = len(neighbor.GetNeighbors())
                all_class_info.append(num_bonds - 1)

    x["OH_level"] = min(all_class_info)
    x["O_index"] = alcohols[0][1]
    return x


def remove_too_difficult_alcohol(x, threshold=100):
    mol = Chem.MolFromSmiles(x["Smiles"])
    heavy_mol_num = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    if heavy_mol_num > threshold:
        x["Tag"] = False
    return x


def check_reaction_state(row, data3):
    # Find the corresponding Smiles in data3
    match = data3[data3["Smiles"] == row["Smiles"]]

    # If not found, return 'No_Exp'
    if match.empty:
        row["Exp_State"] = "No_Exp"
    # If found, check the value of reaction_state
    elif match.iloc[0]["reaction_state"]:
        row["Exp_State"] = "React"
    else:
        row["Exp_State"] = "No_React"

    return row


def alcohol_cleaning(strict_mode=False):
    # generate alcohol smiles from xlsx
    data1 = pd.read_excel(Path(__file__).parent / "data/origin_data/alcohol_with_energy.xlsx")
    data2 = pd.read_excel(Path(__file__).parent / "data/origin_data/alcohol_with_bide.xlsx")
    data3 = pd.read_excel(Path(__file__).parent / "data/origin_data/alcohol_with_origin.xlsx")

    # Normalize data3

    print("checking data1...")
    data1 = check_smiles(data1)
    print("checking data2...")
    data2 = check_smiles(data2)
    print("checking data3...")
    data3 = check_smiles(data3)

    smiles_datas = pd.concat([data1["Smiles"], data2["Smiles"], data3["Smiles"]], axis=0).reset_index().drop(columns=["index"])
    smiles_datas = smiles_datas["Smiles"]

    # Convert all molecules to a unified form with OH (remove charges)
    if isinstance(smiles_datas, pd.Series):
        smiles_datas = pd.DataFrame({"Smiles": smiles_datas})
    smiles_datas["Tag"] = True

    print(f"Before basic data clean, len of smiles: {len(smiles_datas)}")
    smiles_datas = smiles_datas.apply(lambda x: deal_with_more_than_one_part_smiles(x), axis=1)
    smiles_datas = smiles_datas[smiles_datas["Tag"] == True]
    print(f"After basic data clean, len of smiles: {len(smiles_datas)}")

    # drop duplicates smiles and structures
    data3 = data3.apply(canonical_smiles, axis=1)
    smiles_datas = smiles_datas.apply(lambda x: canonical_smiles(x), axis=1)
    smiles_datas.drop_duplicates(inplace=True)
    print(f"After canonical, len of smiles: {len(smiles_datas)}")

    # Structure Screening 1
    # Filter molecules with alcohol structures (not COOH)
    smiles_datas = smiles_datas.apply(has_alcohol, axis=1)
    smiles_datas = smiles_datas[smiles_datas["Tag"]]
    print(f"After alcohol check, length of smiles: {len(smiles_datas)}.")
    smiles_datas

    # Structure Screening 2
    # If the structure has more than one OH (and these OH have different chemical environments), exclude it
    if strict_mode:
        smiles_datas = smiles_datas.apply(has_multiple_OH, axis=1)
        smiles_datas = smiles_datas[smiles_datas["Tag"] == 1]
        print(f"After multi OH check, length of smiles: {len(smiles_datas)}.")

    # Structure Screening 3
    # Exclude structures containing nucleophilic atoms such as N, S, P, and carboxylic acids,
    smiles_datas = smiles_datas.apply(lambda x: remove_nucleo_atoms(x, strict_mode=strict_mode), axis=1)
    smiles_datas = smiles_datas[smiles_datas["Tag"] == True]
    print(f"After removing nucleo atoms, length of smiles: {len(smiles_datas)}.")

    # Structure Screening 4
    # Exclude overly complex structures
    smiles_datas = smiles_datas.apply(remove_too_difficult_alcohol, axis=1)
    smiles_datas = smiles_datas.apply(manual_delete, axis=1)
    smiles_datas = smiles_datas[smiles_datas["Tag"] == True]
    print(f"After removing too difficult alcohol, length of smiles: {len(smiles_datas)}.")

    # Distinguish between primary, secondary, and tertiary alcohols
    smiles_datas = smiles_datas.apply(distinguish_alcohol_level, axis=1)

    # Record whether it has appeared in experiments and the experimental results
    smiles_datas = smiles_datas.apply(lambda x: check_reaction_state(x, data3), axis=1)

    atom_list = []
    for s in smiles_datas["Smiles"]:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            for atom in mol.GetAtoms():
                atom_list.append(atom.GetSymbol())

    return smiles_datas


if __name__ == "__main__":
    alcohol_cleaning(strict_mode=True).to_csv("data/all_alcohols_in_strict_mode.csv", index=False)
