from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path
import re
import pandas as pd


def smiles2filename(smiles: str):
    """convert smiles to valid file name."""
    filename = smiles.replace(".", "<d>").replace("/", "<ls>").replace("\\", "<rs>")
    return filename


def Mol2Gjf(
    file_name: str,
    mol_path: str,
    gjf_path: str,
    ini: str,
    command: str,
    other_info: str = "",
    charge: int = 0,
    spin: int = 1,
):
    """generate .gjf file from .mol file.

    Args:
        `file_name` (str): name of .mol file without suffix.\n
        `mol_path` (str): path of the .mol file.\n
        `gjf_path` (str): paht of .gjf file wanted to generate.\n
        `ini` (str): initial settings. Define CPU core and memory.\n
        `command` (str): command line of Gaussian. begin with `#`\n
        `other_info` (str, optional): some supplementary information that added behand the cordination parts. Defaults to None.\n
        `charge` (int, optional): charge of the molecular. Defaults to 0.\n
        `spin` (int, optional): spin of the molecular. Defaults to 1.\n

    """
    try:
        assert Path(mol_path / Path(file_name + ".mol")).exists() == True
    except:
        raise FileExistsError(f"there is no file {file_name}.mol")
    with open(mol_path / Path(file_name + ".mol")) as gjf:
        lines = gjf.readlines()
    # get cordinations from .mol file
    cords = []
    for l in lines:
        append_tag = True
        l = re.split(r"\s+", l)[1:5] if len(re.split(r"\s+", l)) >= 5 else ["0", "0", "0", "0"]
        for i in range(len(l)):
            if i < 3 and re.match(r"-?\d+\.\d+", l[i]) is None:
                append_tag = False
            elif re.match(r"[a-zA-Z]*", l[i]) is None:
                append_tag = False
        cords.append(l) if append_tag else None

    # write cordinations and other information to .gjf file
    with open(gjf_path / Path(file_name + ".gjf"), "w") as gjf:
        gjf.write(ini)
        gjf.write(f"%chk={str(Path(file_name))}.chk\n")
        gjf.write(command + "\n\n")
        gjf.write(f"{file_name}-descriptor_calculate\n\n")
        gjf.write(f"{charge} {spin}\n")
        for cord in cords:
            gjf.write(f" {cord[-1]}    {cord[0]}    {cord[1]}    {cord[2]}\n")
        gjf.write("\n")
        gjf.write(other_info)
        gjf.write("\n\n")


def Smiles2Mol(s: str, mol_path: str, index: int, file_name: str = "") -> None:
    """change smiles to .mol file. structrue optimization by MMFF94

    Args:
        `s` (str): input smiles.\n
        `mol_path` (str or PosixPath): path of the .mol file.\n
        `index` (int): the index of file.\n
    """
    mol = Chem.MolFromSmiles(s)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    file_name = str(index) + "-" + s if file_name == "" else file_name
    # file_name = file_name.replace(".", "---")
    file_name = smiles2filename(file_name)
    Chem.MolToMolFile(mol, str(mol_path / Path(f"{file_name}.mol")))
    charge = Chem.GetFormalCharge(mol)
    return file_name, charge


def SeriesSmiles2gjf(
    smiles_list: list,
    destination_path: str,
    command: str,
    other_info: str = "",
    spin: int = 1,
    core_num: int = 4,
    memory: int = 8,
    idx_start: int = 0,
) -> None:
    """from a smiles list generate a series of gjf files. structure optimization by MMFF94.\n
    index in filename start from 0.

    Args:
        `smiles_list` (list): a list of smiles or pd.Series\n
        `destination_path` (str, PixosPath): path where gjf files generated.\n
        `command` (str): command line of Gaussian. begin with `#`\n
        `other_info` (str, optional): some supplementary information that added behand the cordination parts. Defaults to None.\n
        `spin` (int, optional): spin of the molecular. Defaults to 1.\n
        `core_num` (int, optional): used CPU core number. Defaults to 4.\n
        `memory` (int, optional): used memory size (unit is GB). Defaults to 8GB.\n
        `idx_start` (int, optional): set the exact start idx of file. Defaults to 0.\n
    """
    ini = f"%mem={memory}GB\n%nprocshared={core_num}\n"
    if isinstance(smiles_list, pd.Series):
        smiles_list = smiles_list.tolist()
    elif type(smiles_list) != list:
        raise TypeError(f"Smiles list is {type(smiles_list)}, not list or pd.Series")
    destination_path = Path(destination_path)
    mol_path = destination_path / Path("input_mol")
    gjf_path = destination_path / Path("input_gjf")
    mol_path.mkdir(parents=True, exist_ok=True), gjf_path.mkdir(parents=True, exist_ok=True)
    for smiles in smiles_list:
        print("   " + smiles)
        if "blanck_cell" == smiles:
            continue
        file_name, charge = Smiles2Mol(s=smiles, mol_path=mol_path, index=idx_start + smiles_list.index(smiles))

        Mol2Gjf(
            file_name=file_name,
            mol_path=mol_path,
            gjf_path=gjf_path,
            ini=ini,
            command=command,
            other_info=other_info,
            charge=charge,
            spin=spin,
        )


if __name__ == "__main__":
    type_ = "HTE_new_mols"
    smiles_list = pd.read_csv(Path(__file__).parent / Path(f"new_add_smiles.csv"))["smiles"]
    SeriesSmiles2gjf(
        smiles_list,
        destination_path=Path(__file__).parent / Path(f"Gauss_calc/{type_}"),
        command="#p b3lyp def2svp opt freq",
        spin=1,
        core_num=24,
        memory=10,
        idx_start=0,
    )
