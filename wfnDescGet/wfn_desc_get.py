from copy import copy
from pathlib import Path
import subprocess
from subprocess import DEVNULL as null
from tqdm import tqdm
import pandas as pd
import shutil

from .wfn_utils import *


# Need to define personal Multiwfn path
# Better to add Gaussian excutable file path into setting.ini file of Multiwfn. option is `gaupath=/path/to/Gaussian`
# Current Multiwfn version is 3.8_dev_bin_Linux_noGUI (release date: 2023-Sep-9)
Multiwfn = str(Path("~/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI").expanduser())  # define path of Multiwfn


GlobalESPDescriptorName = [
    "min_ESP",
    "max_ESP",
    "aver_pESP",
    "aver_nESP",
    "aver_oESP",
    "vari_p_ESP",
    "vari_n_ESP",
    "vari_o_ESP",
    "char_balance",
    "MPI",
    "area_pSurface",
    "area_nSurface",
    "aver_ALIE",
    "vari_ALIE",
    "aver_LEAE",
    "vari_LEAE",
    "aver_LEA",
    "vari_LEA",
]
AtomESPDescriptorName = [
    "max_ESP_atom",
    "min_ESP_atom",
    "aver_ESP_atom",
    "vari_ESP_atom",
    "icp_atom",
    "nu_atom",
]


def get_ESP_descriptors(fchk_files: list, smiles_list: list = None, atoms_list: list = [], file_mode=1) -> pd.DataFrame:
    """Use Multiwfn to get ESP descriptors from fchk file. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `atoms_list (list, optional)`: atom number of atom properties that want to get. Defaults to None and not get atom properties
        Attention! Multiwfn atom number start from 1!\n
        `file_mode (int,optional)`: get descriptors from temp_files in ./Multiwfn_desc/tmp_ESP

    Returns:
        `pd.DataFrame`: ESPdescriptor dataframe
    """
    print("-----< Now generate ESP descriptors >-----")
    ESP_operator = (
        "12",  # Quantitative analysis of molecular surface
        "0",  # Start analysis now!
        "11",  # Output surface properties of each atom
        "n",  # no pdb file output
        "-1",  # Return to upper level menu
        "-1",  # Return to main menu
        "q",
    )  # Exit program gracefully

    ALIE_operator = ("12", "2", "2", "0", "-1", "-1", "q")
    LEAE_operator = ("12", "2", "4", "0", "-1", "-1", "q")
    LEA_operator = ("12", "2", "-4", "0", "-1", "-1", "q")
    ESP_operator = operator_combine(ESP_operator)
    ALIE_operator = operator_combine(ALIE_operator)
    LEAE_operator = operator_combine(LEAE_operator)
    LEA_operator = operator_combine(LEA_operator)

    fchk_files = list_convert(fchk_files)
    atoms_list, all_atom_name = atoms_desc_name_generate(atoms_list, AtomESPDescriptorName, len(fchk_files))

    AllESPDescriptorName = GlobalESPDescriptorName + all_atom_name
    ESPDescriptors = pd.DataFrame(columns=AllESPDescriptorName, dtype="float64")

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]

    assert len(fchk_files) == len(smiles_list) == len(atoms_list)  # check if these lists are same length

    (Path.cwd() / Path("Multiwfn_desc/tmp_ESP")).mkdir(parents=True, exist_ok=True)
    for s, atoms, fchk_file in tqdm(zip(smiles_list, atoms_list, fchk_files)):
        idx = fchk_files.index(fchk_file)
        ESP_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_ESP/{idx}-ESP_output.txt")
        ALIE_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_ESP/{idx}-ALIE_output.txt")
        LEAE_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_ESP/{idx}-LEAE_output.txt")
        LEA_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_ESP/{idx}-LEA_output.txt")

        use_temp_file = file_mode_deal(file_mode, [ESP_output, ALIE_output, LEAE_output, LEA_output])
        fchk_file = fchk_file_check(fchk_file)

        if not use_temp_file:
            with open(ESP_output, "w") as ESP_stdout, open(LEA_output, "w") as LEA_stdout:
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=ESP_stdout).communicate(input=ESP_operator)
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=LEA_stdout).communicate(input=LEA_operator)
            with open(ALIE_output, "w") as ALIE_stdout, open(LEAE_output, "w") as LEAE_stdout:
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=ALIE_stdout).communicate(input=ALIE_operator)
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=LEAE_stdout).communicate(input=LEAE_operator)

        # molecular surface ESP properties
        ESPDescriptors.loc[s, "min_ESP"] = get_data(ESP_output, "Global surface minimum:", "a.u.")
        ESPDescriptors.loc[s, "max_ESP"] = get_data(ESP_output, "Global surface maximum:", "a.u.")
        ESPDescriptors.loc[s, "aver_pESP"] = get_data(ESP_output, "Positive average value:", "a.u.")
        ESPDescriptors.loc[s, "aver_nESP"] = get_data(ESP_output, "Negative average value:", "a.u.")
        ESPDescriptors.loc[s, "aver_oESP"] = get_data(ESP_output, "Overall average value:", "a.u.")
        ESPDescriptors.loc[s, "vari_p_ESP"] = get_data(ESP_output, "Positive variance:", "a.u.^2")
        ESPDescriptors.loc[s, "vari_n_ESP"] = get_data(ESP_output, "Negative variance:", "a.u.^2")
        ESPDescriptors.loc[s, "vari_o_ESP"] = get_data(ESP_output, "Overall variance (sigma^2_tot):", "a.u.^2")
        ESPDescriptors.loc[s, "char_balance"] = get_data(ESP_output, "Balance of charges (nu):", "\n")
        ESPDescriptors.loc[s, "MPI"] = get_data(ESP_output, "Molecular polarity index (MPI):", "eV")
        ESPDescriptors.loc[s, "area_pSurface"] = get_data(ESP_output, "Positive surface area:", "Bohr^2")
        ESPDescriptors.loc[s, "area_nSurface"] = get_data(ESP_output, "Negative surface area:", "Bohr^2")

        # atom surface ESP properties
        max_min_atom_data = get_dataframe(
            ESP_output, "Atom#    All/Positive/Negative area (Ang^2)  Minimal value   Maximal value\n", "\n \n", reset_index=True
        )
        aver_vari_atom_data = get_dataframe(
            ESP_output, "Atom#    All/Positive/Negative average       All/Positive/Negative variance\n", "\n \n", reset_index=True
        )
        icp_nu_atom_data = get_dataframe(ESP_output, "  Atom#           Pi              nu         nu*sigma^2\n", "\n\n", reset_index=True)
        # Attension: some of atom ESP properties cannot get if this atom is not on surface.
        for i, atom in zip(range(len(atoms)), atoms):
            if atom in max_min_atom_data.index:
                ESPDescriptors.loc[s, f"max_ESP_atom{i}"] = max_min_atom_data.loc[atom, 4]
                ESPDescriptors.loc[s, f"min_ESP_atom{i}"] = max_min_atom_data.loc[atom, 5]
            if atom in aver_vari_atom_data.index:
                ESPDescriptors.loc[s, f"aver_ESP_atom{i}"] = aver_vari_atom_data.loc[atom, 1]
                ESPDescriptors.loc[s, f"vari_ESP_atom{i}"] = aver_vari_atom_data.loc[atom, 4]
            if atom in icp_nu_atom_data.index:
                ESPDescriptors.loc[s, f"icp_atom{i}"] = icp_nu_atom_data.loc[atom, 1]
                ESPDescriptors.loc[s, f"nu_atom{i}"] = icp_nu_atom_data.loc[atom, 2]

        # other molecular ESP properties
        ESPDescriptors.loc[s, "aver_ALIE"] = get_data(ALIE_output, "Average value:", "a.u.")
        ESPDescriptors.loc[s, "vari_ALIE"] = get_data(ALIE_output, "Variance:", "a.u.^2")
        ESPDescriptors.loc[s, "aver_LEAE"] = get_data(LEAE_output, "Overall average value:", "a.u.")
        ESPDescriptors.loc[s, "vari_LEAE"] = get_data(LEAE_output, "Overall variance:", "a.u.")
        ESPDescriptors.loc[s, "aver_LEA"] = get_data(LEA_output, "Overall average value:", "a.u.")
        ESPDescriptors.loc[s, "vari_LEA"] = get_data(LEA_output, "Overall variance:", "a.u.")

        ESPDescriptors = deal_with_dataframe(ESPDescriptors)
    return ESPDescriptors


GlobalCDFTDescriptorName = [
    "E_HOMO",
    "VIP",
    # "V2IP",
    "VEA",
    "Mulliken_Chi",
    "Chem_potential",
    "Elec_hardness",
    "Elec_softness",
    "Electro_idx",
    "Nucleo_idx",
    # "omega_cubic",  # to difficult to calculate....
]
AtomCDFTDescriptorName = [
    "Hirshfeld_atom",
    "Cond_Fukui+_atom",
    "Cond_Fukui-_atom",
    "Cond_Fukui0_atom",
    "Cond_dual_desc_atom",
    "Cond_softness+_atom",
    "Cond_softness-_atom",
    "Cond_softness0_atom",
    "Relative_electro_idx_atom",
    "Relative_nucleo_idx_atom",
    "Cond_electro_idx_atom",
    "Cond_nucleo_idx_atom",
    # "Cond_omega_cubic_idx_atom",
]

"""
• If the current system is a+1 charged radical cation, the N state is 1 2, the N+1 state is 0 1, and the N-1 state is 2 1.
• If the current system is a-1 charged radical anion, the N state is -1 2, the N+1 state is -2 1, and the N-1 state is 0 1.
• If the current system is a+1 charged closed-shell cation, the N state is 1 1, the N+1 state is 0 2, and the N-1 state is 2 2.
• If the current system is a-1 charged closed-shell anion, the N state is -1 1, the N+1 state is -2 2, and the N-1 state is 0 2.
"""


def get_CDFT_descriptors(
    fchk_files: list,
    smiles_list: list = None,
    atoms_list: list = [],
    prop=(0, 1),
    calc_method="",
    kill=False,
    file_mode=1,
) -> pd.DataFrame:
    """Use Multiwfn to get CDFT descriptors from fchk file. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `atoms_list (list, optional)`: atom number of atom properties that want to get. Defaults to None and not get atom properties
        Attention! Multiwfn atom number start from 1!\n
        `prop` (optional): set charge and spin. defaults to (0, 1) and results to `(0 1)\n(-1 2)\n(1 2)\n(2 1)`.
        if system is not closed shell, then you should define it by yourself.\n
        `calc_method` (str, optional): define gaussian calculation base/method. defaults use b3lyp/6-31g*.\n
        `kill (bool)`: Whether to continue calculation after Gaussian calculation not converged. Default False.\n
        `file_mode (int,optional)`: decide how to get descriptors from temp_files in ./Multiwfn_desc/tmp_CDFT.
        Attention! If change method, All `Nucleophilicity index` should be redefined. see http://sobereva.com/484

    Returns:
        `pd.DataFrame`: CDFTdescriptor dataframe
    """
    print("-----< Now generate CDFT descriptors >-----")
    fchk_files = list_convert(fchk_files)
    if type(prop) == tuple:
        prop = [prop] * len(fchk_files)
    elif type(prop) == list:
        assert len(prop) == len(fchk_files)
    else:
        print("make sure prop is prop is of type list or tuple.")
        # print(f"the length of prop is {len(prop)}, however, the length of fchk_files is {len(fchk_files)}.")
        raise TypeError("CDFT: prop set error.")

    atoms_list, all_atom_name = atoms_desc_name_generate(atoms_list, AtomCDFTDescriptorName, len(fchk_files))

    AllCDFTDescriptorName = GlobalCDFTDescriptorName + all_atom_name
    CDFTDescriptors = pd.DataFrame(columns=AllCDFTDescriptorName, dtype="float64")

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]
    assert len(fchk_files) == len(smiles_list) == len(atoms_list)  # check if these lists are same length
    (Path.cwd() / Path("Multiwfn_desc/tmp_CDFT")).mkdir(parents=True, exist_ok=True)
    for s, atoms, p, fchk_file in tqdm(zip(smiles_list, atoms_list, prop, fchk_files)):
        idx = fchk_files.index(fchk_file)
        use_temp_file = file_mode_deal(file_mode, Path.cwd() / Path(f"Multiwfn_desc/tmp_CDFT/{idx}-CDFT.txt"))

        spin_state = "({} {})\n({} {})\n({} {})".format(p[0], p[1], p[0] - 1, p[1] + 1, p[0] + 1, p[1] + 1)
        if all((Path.cwd() / Path(f"{f}.wfn")).exists() for f in ["N", "N-1", "N+1"]):
            CDFT_operator = (
                "22",  # Conceptual DFT (CDFT) analysis
                # "-1",  # Toggle calculating w_cubic electrophilicity index (Now not use cause to difficult to calculate...)
                "1",  # Generate .wfn files for N, N+1, N-1 electrons states (NO N-2)
                f"{calc_method}",  # Input Gaussian keywords used for single point task, e.g. PBE1PBE/def2SVP. Default B3LYP/6-31G*
                f"{spin_state}",  # Input the net charge and spin multiplicity. Default (0 1), (-1 2), (1 2) and (2 1)
                "n",  # use gaussian definded by setting.ini in Multiwfn to do calculations
                "2",  # Calculate various quantitative indices.
                "0",  # Return to main menu
                "q",  # Exit program gracefully
            )
        else:
            CDFT_operator = (
                "22",  # Conceptual DFT (CDFT) analysis
                # "-1",  # Toggle calculating w_cubic electrophilicity index (Now not use cause to difficult to calculate...)
                "1",  # Generate .wfn files for N, N+1, N-1 electrons states (NO N-2)
                f"{calc_method}",  # Input Gaussian keywords used for single point task, e.g. PBE1PBE/def2SVP. Default B3LYP/6-31G*
                f"{spin_state}",  # Input the net charge and spin multiplicity. Default (0 1), (-1 2), (1 2) and (2 1)
                "y",  # use gaussian definded by setting.ini in Multiwfn to do calculations
                "2",  # Calculate various quantitative indices.
                "0",  # Return to main menu
                "q",  # Exit program gracefully
            )
        CDFT_operator = operator_combine(CDFT_operator)
        fchk_file = fchk_file_check(fchk_file)

        if use_temp_file:
            CDFT_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_CDFT/{idx}-CDFT.txt")
        else:
            CDFT_output = Path.cwd() / Path("CDFT.txt")
            CDFT_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_CDFT/{idx}-CDFT.txt")

        if not use_temp_file:
            prog = subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null)
            prog.communicate(input=CDFT_operator)
            calc_res = prog.returncode
            try:
                assert calc_res == 0
            except:
                print(f"ERROR: Molecule {s} cannot be properly calculate. error code is {calc_res}")
                if kill:
                    raise Exception(
                        "Probobally gaussian calculation is not converged. Try `SCF(novaracc,noincfock)` or see http://sobereva.com/61."
                    )
                else:
                    continue

        # molecular CDFT properties
        CDFTDescriptors.loc[s, "E_HOMO"] = get_data(CDFT_output, "E_HOMO(N):", "Hartree,")
        CDFTDescriptors.loc[s, "VIP"] = get_data(CDFT_output, "Vertical IP:", "Hartree,")
        # CDFTDescriptors.loc[s, "V2IP"] = get_data(CDFT_output, "Vertical second IP:", "Hartree,")
        CDFTDescriptors.loc[s, "VEA"] = get_data(CDFT_output, "Vertical EA:", "Hartree,")
        CDFTDescriptors.loc[s, "Mulliken_Chi"] = get_data(CDFT_output, "Mulliken electronegativity:", "Hartree,")
        CDFTDescriptors.loc[s, "Chem_potential"] = get_data(CDFT_output, "Chemical potential:", "Hartree,")
        CDFTDescriptors.loc[s, "Elec_hardness"] = get_data(CDFT_output, "Hardness (=fundamental gap):", "Hartree,")
        CDFTDescriptors.loc[s, "Elec_softness"] = get_data(CDFT_output, "Softness:", "Hartree^-1")
        CDFTDescriptors.loc[s, "Electro_idx"] = get_data(CDFT_output, "Electrophilicity index:", "Hartree,")
        CDFTDescriptors.loc[s, "Nucleo_idx"] = get_data(CDFT_output, "Nucleophilicity index:", "Hartree,")
        # CDFTDescriptors.loc[s, "omega_cubic"] = get_data(CDFT_output, "Cubic electrophilicity index (w_cubic):", "Hartree,")

        # Atom CDFT properties
        hirshfeld_Fukui_atom_data = get_dataframe(
            CDFT_output, "Atom     q(N)    q(N+1)   q(N-1)     f-       f+       f0      CDD\n", "\n \n", remove=(10, None)
        )
        cond_E_N_atom_data = get_dataframe(
            CDFT_output, "Atom              Electrophilicity          Nucleophilicity\n", "\n \n", remove=(10, None)
        )
        # cond_cubic_atom_data = get_dataframe(CDFT_output, "Atom              Value\n", "\n \n", remove=(10, None))
        softness_rE_rN_atom_data = get_dataframe(
            CDFT_output, "Atom         s-          s+          s0        s+/s-       s-/s+\n", "\n \n", remove=(10, None)
        )

        for i, atom in zip(range(len(atoms)), atoms):
            atom -= 1  # to align it with dataframe
            CDFTDescriptors.loc[s, f"Hirshfeld_atom{i}"] = hirshfeld_Fukui_atom_data.loc[atom, 0]
            CDFTDescriptors.loc[s, f"Cond_Fukui-_atom{i}"] = hirshfeld_Fukui_atom_data.loc[atom, 3]
            CDFTDescriptors.loc[s, f"Cond_Fukui+_atom{i}"] = hirshfeld_Fukui_atom_data.loc[atom, 4]
            CDFTDescriptors.loc[s, f"Cond_Fukui0_atom{i}"] = hirshfeld_Fukui_atom_data.loc[atom, 5]
            CDFTDescriptors.loc[s, f"Cond_dual_desc_atom{i}"] = hirshfeld_Fukui_atom_data.loc[atom, 6]
            CDFTDescriptors.loc[s, f"Cond_softness-_atom{i}"] = softness_rE_rN_atom_data.loc[atom, 0]
            CDFTDescriptors.loc[s, f"Cond_softness+_atom{i}"] = softness_rE_rN_atom_data.loc[atom, 1]
            CDFTDescriptors.loc[s, f"Cond_softness0_atom{i}"] = softness_rE_rN_atom_data.loc[atom, 2]
            CDFTDescriptors.loc[s, f"Relative_electro_idx_atom{i}"] = softness_rE_rN_atom_data.loc[atom, 3]
            CDFTDescriptors.loc[s, f"Relative_nucleo_idx_atom{i}"] = softness_rE_rN_atom_data.loc[atom, 4]
            CDFTDescriptors.loc[s, f"Cond_electro_idx_atom{i}"] = cond_E_N_atom_data.loc[atom, 0]
            CDFTDescriptors.loc[s, f"Cond_nucleo_idx_atom{i}"] = cond_E_N_atom_data.loc[atom, 1]
            # CDFTDescriptors.loc[s, f"Cond_omega_cubic_idx_atom{i}"] = cond_cubic_atom_data.loc[atom, 0]
        # move CDFT.txt to stored
        if not use_temp_file:
            shutil.move(CDFT_output, CDFT_output_store)
            for wfn_file in list(Path.cwd().glob("*.wfn")):
                wfn_file.unlink() if wfn_file.exists() else None
    (Path.cwd() / Path("fort.7")).unlink() if (Path.cwd() / Path("fort.7")).exists() else None

    CDFTDescriptors = deal_with_dataframe(CDFTDescriptors)
    return CDFTDescriptors


def get_internal_cordination(
    fchk_files: list, atoms_list: list, smiles_list: list = None, bond_list: list = [], angle_list: list = [], dihedral_list: list = []
):
    """Use Multiwfn to get internal cordinations from fchk file. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `atoms_list (list)`: atom number of atom properties that want to get. Defaults to None and not get atom properties
        Attention! Multiwfn atom number start from 1!\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `bond_list (list, optional)`: used to represent a bond between bit `i` and bit `j` in a list of atoms. Defaults to [].
        `angle_list (list, optional)`: used to represent a angle in bit `i`, bit `j` and bit `k` in a list of atoms. Defaults to [].
        `dihedral_list (list, optional)`: used to represent a dihedral angle in bit `i`, bit `j`, bit `k` and bit `l` in a list of atoms. Defaults to [].

    Returns:
        `pd.DataFrame`: Internal cordination dataframe
    """
    print("-----< Now generate internal cordinations >-----")
    Internal_operator = [
        "300",  # Other functions (Part 3)
        "7",  # Geometry operation on the present system
        "-1",  # Output system to .xyz file
        "opt_struct.txt\n"  # Output file name
        "-10",  # Return
        "0",  # Return
        "q",  # Exit program gracefully
    ]

    Internal_operator = operator_combine(Internal_operator)
    fchk_files = list_convert(fchk_files)

    assert all(len(atoms) == len(atoms_list[0]) for atoms in atoms_list)
    assert all(len(b) == len(bond_list[0]) for b in bond_list)
    assert all(len(a) == len(angle_list[0]) for a in angle_list)
    assert all(len(d) == len(dihedral_list[0]) for d in dihedral_list)
    all_desc_name = internal_name_generate(len(bond_list), len(angle_list), len(dihedral_list))
    InternalDescriptors = pd.DataFrame(columns=all_desc_name)

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]

    assert len(fchk_files) == len(smiles_list) == len(atoms_list)  # check if these lists are same length

    for s, atoms, fchk_file in tqdm(zip(smiles_list, atoms_list, fchk_files)):
        fchk_file = fchk_file_check(fchk_file)
        atoms = [atom - 1 for atom in atoms]  # to change atom list from 1-start to 0-start

        subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=Internal_operator)
        cor_matrix = load_cordinations()

        # get bond length descriptors
        for b in bond_list:
            p1 = [float(i) for i in cor_matrix[atoms[b[0]]]]
            p2 = [float(i) for i in cor_matrix[atoms[b[1]]]]
            InternalDescriptors.loc[s, f"bond_{bond_list.index(b)}"] = bond_length_calc(p1, p2)

        for a in angle_list:
            p1 = [float(i) for i in cor_matrix[atoms[a[0]]]]
            p2 = [float(i) for i in cor_matrix[atoms[a[1]]]]
            p3 = [float(i) for i in cor_matrix[atoms[a[2]]]]
            InternalDescriptors.loc[s, f"angle_{angle_list.index(a)}"] = angle_calc(p1, p2, p3)

        for d in dihedral_list:
            p1 = [float(i) for i in cor_matrix[atoms[d[0]]]]
            p2 = [float(i) for i in cor_matrix[atoms[d[1]]]]
            p3 = [float(i) for i in cor_matrix[atoms[d[2]]]]
            p4 = [float(i) for i in cor_matrix[atoms[d[3]]]]
            InternalDescriptors.loc[s, f"dihedral_{dihedral_list.index(d)}"] = dihedral_angle_calc(p1, p2, p3, p4)

    InternalDescriptors = deal_with_dataframe(InternalDescriptors)
    return InternalDescriptors


GlobalShapeDescriptorName = [
    "Molecular_ES_radius_min",
    "Molecular_ES_radius_max",
    "Molecular_volumn",
    "Molecular_sphericity",
    "Molecular_all_surface_area",
    "Molecular_length",
    "Molecular_widht",
    "Molecular_height",
]


def get_molecular_shape(fchk_files: list, smiles_list: list = None, file_mode=1) -> pd.DataFrame:
    """Use Multiwfn to get some molecular descriptors from fchk file. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `file_mode (int,optional)`: decide how to get descriptors from temp_files in ./Multiwfn_desc/tmp_Shape.

    Returns:
        `pd.DataFrame`: Molecular shape descriptor dataframe
    """
    print("-----< Now generate molecular shape descriptors >-----")
    shape_operator = (  # for molecular radius, define as
        12,  # Quantitative analysis of molecular surface
        2,  # Select mapped function
        -1,  # Select non-function to reduce calculate time
        0,  # Start analysis now!
        10,  # Output the closest and farthest distance between the surface and a point
        "g",  # set geometry center of present system as the point
        -1,  # Return to upper level menu
        6,  # Start analysis without considering mapped function
        -1,  # Return to upper level menu
        -1,  # Return to main menu
        100,  #  Other functions (Part1)
        21,  # Calculate properties based on geometry information for specific atoms
        "size",  # Report size information of the whole system
        0,  # Return
        "q",  # Quit
        0,  # Return to main menu
        "q",  # Exit gracefully
    )
    shape_operator = operator_combine(shape_operator)

    fchk_files = list_convert(fchk_files)

    AllShapeDescriptorName = GlobalShapeDescriptorName
    ShapeDescriptors = pd.DataFrame(columns=AllShapeDescriptorName)

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]

    assert len(fchk_files) == len(smiles_list)  # check if these lists are same length
    (Path.cwd() / Path("Multiwfn_desc/tmp_shape")).mkdir(parents=True, exist_ok=True)
    for s, fchk_file in tqdm(zip(smiles_list, fchk_files)):
        idx = fchk_files.index(fchk_file)
        use_temp_file = file_mode_deal(file_mode, Path.cwd() / Path(f"Multiwfn_desc/tmp_shape/{idx}-shape.txt"))
        fchk_file = fchk_file_check(fchk_file)

        shape_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_shape/{idx}-shape.txt")

        if not use_temp_file:
            with open(shape_output, "w") as shape_stdout:
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=shape_stdout).communicate(input=shape_operator)

        # molecular CDFT properties
        ShapeDescriptors.loc[s, "Molecular_ES_radius_min"] = get_data(shape_output, "The closest distance to the point:", "Bohr")
        ShapeDescriptors.loc[s, "Molecular_ES_radius_max"] = get_data(shape_output, "The farthest distance to the point:", "Bohr")
        ShapeDescriptors.loc[s, "Molecular_volumn"] = get_data(shape_output, "Volume enclosed by the isosurface:", "Bohr^3")
        ShapeDescriptors.loc[s, "Molecular_sphericity"] = get_data(shape_output, "Sphericity:", "\n")
        ShapeDescriptors.loc[s, "Molecular_all_surface_area"] = get_data(shape_output, "Isosurface area:", "Bohr^2")
        ShapeDescriptors.loc[s, "Molecular_length"] = get_data(shape_output, "Length of the three sides:", "Angstrom")[0]
        ShapeDescriptors.loc[s, "Molecular_widht"] = get_data(shape_output, "Length of the three sides:", "Angstrom")[1]
        ShapeDescriptors.loc[s, "Molecular_height"] = get_data(shape_output, "Length of the three sides:", "Angstrom")[2]

    ShapeDescriptors = deal_with_dataframe(ShapeDescriptors)
    return ShapeDescriptors


AtomPropertiesDescriptorName = [
    "charge_Hirshfeld-atom",
    "charge_Mulliken-atom",
    "charge_SCPA-atom",
    "charge_Modified_Mulliken_by_Bickelhaupt-atom",
    "charge_Becke-atom",
    "charge_ADCH-atom",
    "charge_CHELPG-ESP-atom",
    "charge_MK-ESP-atom",
    "charge_CM5-atom",
    "charge_RESP-atom",
    "oxidation_state_LOBA-atom",
    "total_ESP-atom",
    "dipole-atom",
    "<r^2>-atom",
    "energy_index-atom",
]

GlobalAtomPropertiesDescrptorName = ["dipole", "<r^2>"]


def get_atom_descriptors(fchk_files: list, atoms_list: list, smiles_list: list = None, file_mode: int = 1) -> pd.DataFrame:
    """Use Multiwfn to get some atom properties descriptors from fchk file, like charge and dipole. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `atoms_list (list)`: atom number of atom properties that want to get.
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `use_temp_file (int,optional)`: get descriptors from temp_files in ./Multiwfn_desc/tmp_Atom

    Returns:
        `pd.DataFrame`: Molecular atom descriptor dataframe
    """
    print("-----< Now generate atom descriptors >-----")
    fchk_files = list_convert(fchk_files)
    assert all(len(atoms) == len(atoms_list[0]) for atoms in atoms_list)

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]

    atoms_list, all_atom_name = atoms_desc_name_generate(atoms_list, AtomPropertiesDescriptorName, len(fchk_files))
    assert len(fchk_files) == len(smiles_list) == len(atoms_list)  # check if these lists are same length

    Hirshfeld_operator = operator_combine((7, 1, 1, "y", 0, "q"))
    Mulliken_operator = operator_combine((7, 5, 1, "y", 0, 0, "q"))
    SCPA_operator = operator_combine((7, 7, "y", 0, "q"))
    Mod_Mulliken_operator = operator_combine((7, 9, "y", 0, "q"))
    Becke_operator = operator_combine((7, 10, 0, "y", 0, "q"))
    ADCH_operator = operator_combine((7, 11, 1, "y", 0, "q"))
    CHELPG_ESP_operator = operator_combine((7, 12, 1, "y", 0, 0, "q"))
    MK_ESP_operator = operator_combine((7, 13, 1, "y", 0, 0, "q"))
    CM5_operator = operator_combine((7, 16, 1, "y", 0, "q"))
    RESP_operator = operator_combine((7, 18, 1, "y", 0, 0, "q"))
    LOBA_operator = operator_combine((19, 1, 8, 100, 50, 0, -10, "q"))  # save the screen and delete new.fch
    atom_dipole_operator = operator_combine((15, 2, 2, 0, "q"))  # atom_moment.txt

    AllAtomDescriptorName = GlobalAtomPropertiesDescrptorName + all_atom_name
    AtomDescriptors = pd.DataFrame(columns=AllAtomDescriptorName)
    (Path.cwd() / Path("Multiwfn_desc/tmp_atom")).mkdir(parents=True, exist_ok=True)
    for s, atoms, fchk_file in tqdm(zip(smiles_list, atoms_list, fchk_files)):
        idx = fchk_files.index(fchk_file)

        charge_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-charge.txt")
        oxidation_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-oxidation.txt")
        dipole_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-multipole.txt")
        totalESP_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-AtomESP.txt")
        EI_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-energy_index.txt")

        use_temp_file = file_mode_deal(file_mode, [charge_output, oxidation_output, dipole_output, totalESP_output, EI_output])
        fchk_file = fchk_file_check(fchk_file)

        totalESP_operator = operator_combine((1,) + tuple(["a" + str(atom) for atom in atoms]) + ("q", "q"))  # save the screen
        energy_index_operator = operator_combine((200, 12) + tuple([str(atom) for atom in atoms]) + (0, 0, "q"))  # save the screen

        if not use_temp_file:
            charge_origin_output = Path.cwd() / Path(f"{fchk_file.stem}.chg")
            charge_output = Path.cwd() / Path(f"charge.txt")
            charge_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-charge.txt")
            charge_output.open("w")  # to clean the origin things
            oxidation_output = Path.cwd() / Path(f"oxidation.txt")
            oxidation_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-oxidation.txt")
            dipole_output = Path.cwd() / Path(f"multipole.txt")
            dipole_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-multipole.txt")
            totalESP_output = Path.cwd() / Path(f"AtomESP.txt")
            totalESP_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-AtomESP.txt")
            EI_output = Path.cwd() / Path(f"energy_index.txt")
            EI_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_atom/{idx}-energy_index.txt")
        if not use_temp_file:
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=Hirshfeld_operator)
            file_append(charge_output, charge_origin_output, "Hirshfeld:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=Mulliken_operator)
            file_append(charge_output, charge_origin_output, "Mulliken:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=SCPA_operator)
            file_append(charge_output, charge_origin_output, "SCPA:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=Mod_Mulliken_operator)
            file_append(charge_output, charge_origin_output, "Mod Mulliken:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=Becke_operator)
            file_append(charge_output, charge_origin_output, "Becke:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=ADCH_operator)
            file_append(charge_output, charge_origin_output, "ADCH:")
            # subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=CHELPG_ESP_operator)
            # file_append(charge_output, charge_origin_output, "CHELPG:")
            # subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=MK_ESP_operator)
            # file_append(charge_output, charge_origin_output, "MK_ESP:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=CM5_operator)
            file_append(charge_output, charge_origin_output, "CM5:")
            # subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=RESP_operator)
            # file_append(charge_output, charge_origin_output, "RESP:")

            with open(oxidation_output, "w") as oxidation_outstd:
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=oxidation_outstd).communicate(
                    input=LOBA_operator
                )

            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=atom_dipole_operator)

            with open(totalESP_output, "w") as totalESP_outstd:
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=totalESP_outstd).communicate(
                    input=totalESP_operator
                )

            with open(EI_output, "w") as EI_outstd:
                subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=EI_outstd).communicate(
                    input=energy_index_operator
                )

        # Charge DataFrame get
        Hirshfeld_data = get_dataframe(charge_output, "Hirshfeld:\n", "\n-------------", remove=(38, None))
        Mulliken_data = get_dataframe(charge_output, "Mulliken:\n", "\n-------------", remove=(38, None))
        SCPA_data = get_dataframe(charge_output, "SCPA:\n", "\n-------------", remove=(38, None))
        Modified_Mulliken_data = get_dataframe(charge_output, "Mod Mulliken:\n", "\n-------------", remove=(38, None))
        Becke_data = get_dataframe(charge_output, "Becke:\n", "\n-------------", remove=(38, None))
        ADCH_data = get_dataframe(charge_output, "ADCH:\n", "\n-------------", remove=(38, None))
        CHELPG_ESP_data = get_dataframe(charge_output, "CHELPG:\n", "\n-------------", remove=(38, None))
        MK_ESP_data = get_dataframe(charge_output, "MK_ESP:\n", "\n-------------", remove=(38, None))
        CM5_data = get_dataframe(charge_output, "CM5:\n", "\n-------------", remove=(38, None))
        RESP_data = get_dataframe(charge_output, "RESP:\n", "\n-------------", remove=(38, None))

        oxidation_data = get_dataframe(oxidation_output, " Input 0 can exit\n", " The sum of oxidation states", remove=(34, None))
        dipole_data = get_newly_construct_dataframe(
            dipole_output,
            ("*****  Atom", "\n \n"),
            (
                [" Atomic dipole moments:\n X=.*Y=.*Z=.*Norm=", "Atomic electronic spatial extent \<r\^2\>:"],
                ["\n", "\n"],
            ),
            ["atom_dipole", "<r^2>"],
            re_escape_on_data=False,
        )
        totalESP_data = get_newly_construct_dataframe(
            totalESP_output,
            ("------------ Calculate properties at a point ------------", " \n           "),
            ("Total ESP:", "a.u."),
            "total_ESP",
        )
        EI_data = get_newly_construct_dataframe(
            EI_output, (" Calculate EI index for which atom? e.g. 5", "\n\n"), ("The EI index:", "a.u."), "energy_index"
        )

        AtomDescriptors.loc[s, "dipole"] = get_data(dipole_output, "dipole moment (a.u.&Debye):", "\n")[0]
        AtomDescriptors.loc[s, "<r^2>"] = get_data(dipole_output, "Molecular electronic spatial extent <r^2>:", "\n")

        for i, atom in zip(range(len(atoms)), atoms):
            atom -= 1
            AtomDescriptors.loc[s, f"charge_Hirshfeld-atom{i}"] = Hirshfeld_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_Mulliken-atom{i}"] = Mulliken_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_SCPA-atom{i}"] = SCPA_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_Modified_Mulliken_by_Bickelhaupt-atom{i}"] = Modified_Mulliken_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_Becke-atom{i}"] = Becke_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_ADCH-atom{i}"] = ADCH_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_CHELPG-ESP-atom{i}"] = CHELPG_ESP_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_MK-ESP-atom{i}"] = MK_ESP_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_CM5-atom{i}"] = CM5_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"charge_RESP-atom{i}"] = RESP_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"oxidation_state_LOBA-atom{i}"] = oxidation_data.loc[atom, 0]
            AtomDescriptors.loc[s, f"total_ESP-atom{i}"] = totalESP_data.loc[i, "total_ESP"]
            AtomDescriptors.loc[s, f"dipole-atom{i}"] = dipole_data.loc[atom, "atom_dipole"]
            AtomDescriptors.loc[s, f"<r^2>-atom{i}"] = dipole_data.loc[atom, "<r^2>"]
            AtomDescriptors.loc[s, f"energy_index-atom{i}"] = EI_data.loc[i, "energy_index"]

        if not use_temp_file:
            shutil.move(charge_output, charge_output_store)
            shutil.move(dipole_output, dipole_output_store)
            shutil.move(oxidation_output, oxidation_output_store)
            shutil.move(totalESP_output, totalESP_output_store)
            shutil.move(EI_output, EI_output_store)

    AtomDescriptors = deal_with_dataframe(AtomDescriptors)

    for f in Path.cwd().glob("*.chg"):
        f.unlink() if f.exists() else None
    (Path.cwd() / Path("new.fch")).unlink() if (Path.cwd() / Path("new.fch")).exists() else None

    return AtomDescriptors


BondPropertiesDescriptorName = [
    "Mayer_bond_order_bond",
    "Wiberg_bond_order_bond",
    "Mulliken_bond_order_bond",
    "Fuzzy_bond_order_bond",
    "Laplacian_bond_order_bond",
    "Bond_dipole_bond",
]


def get_bond_descriptors(fchk_files: list, atoms_list: list, bond_list: list, smiles_list: list = None, file_mode: int = 1) -> pd.DataFrame:
    """Use Multiwfn to get some bond properties descriptors from fchk file, like bond order. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `atoms_list (list)`: atom number of atom properties that want to get.\n
        `bond_list (list)`: used to represent a bond between bit `i` and bit `j` in a list of atoms. Defaults to [].\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `file_mode (int,optional)`: get descriptors from temp_files in ./Multiwfn_desc/tmp_bond
    Returns:
        `pd.DataFrame`: bond properties descriptos dataframe
    """
    print("-----< Now generate bond descriptors >-----")
    fchk_files = list_convert(fchk_files)
    assert all(len(atoms) == len(atoms_list[0]) for atoms in atoms_list)
    assert all(len(b) == len(bond_list[0]) for b in bond_list)

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]

    all_bond_name = generate_desc_names(BondPropertiesDescriptorName, len(bond_list))
    assert len(fchk_files) == len(smiles_list) == len(atoms_list)  # check if these lists are same length

    mayer_order_operator = operator_combine((9, 1, "y", 0, "q"))
    wiberg_order_operator = operator_combine((9, 3, "y", 0, "q"))
    mulliken_order_operator = operator_combine((9, 4, "y", 0, "q"))
    fuzzy_order_operator = operator_combine((9, 7, "y", 0, "q"))
    laplacian_order_operator = operator_combine((9, 8, "y", 0, "q"))
    mayer_order_str = "mayer bond order:\n Note: The diagonal elements are the sum of corresponding row elements\n ***************************** Bond order matrix *****************************"
    wiberg_order_str = "wiberg bond order:\n Note: The diagonal elements are the sum of corresponding row elements\n ***************************** Bond order matrix *****************************"
    mulliken_order_str = "mulliken bond order:\n Note:The diagonal elements are the sum of corresponding row elements\n ************************ Mulliken bond order matrix ************************"
    fuzzy_order_str = "fuzzy bond order:\n ********************* Total delocalization index matrix *********************"
    laplacian_order_str = "laplacian bond order:\n ************************ Laplacian bond order matrix ************************"

    AllBondDescriptorName = all_bond_name
    BondDescriptors = pd.DataFrame(columns=AllBondDescriptorName)
    (Path.cwd() / Path("Multiwfn_desc/tmp_bond")).mkdir(parents=True, exist_ok=True)
    for s, atoms, fchk_file in tqdm(zip(smiles_list, atoms_list, fchk_files)):
        idx = fchk_files.index(fchk_file)
        use_temp_file = file_mode_deal(file_mode, Path.cwd() / Path(f"Multiwfn_desc/tmp_bond/{idx}-bond_order.txt"))

        fchk_file = fchk_file_check(fchk_file)
        dipole_operator = (200, 2, 2) + tuple(["{},{}".format(atoms[b[0]], atoms[b[1]]) for b in bond_list]) + ("q", 0, 0, "q")
        dipole_operator = operator_combine(dipole_operator)

        if use_temp_file:
            order_origin_output = Path.cwd() / Path(f"bndmat.txt")
            order_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_bond/{idx}-bond_order.txt")
        else:
            order_origin_output = Path.cwd() / Path(f"bndmat.txt")
            order_output = Path.cwd() / Path(f"bond_order.txt")
            order_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_bond/{idx}-bond_order.txt")
            order_output.open("w")  # to clean the origin things
        if not use_temp_file:
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=mayer_order_operator)
            file_append(order_output, order_origin_output, "mayer bond order:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=wiberg_order_operator)
            file_append(order_output, order_origin_output, "wiberg bond order:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=mulliken_order_operator)
            file_append(order_output, order_origin_output, "mulliken bond order:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=fuzzy_order_operator)
            file_append(order_output, order_origin_output, "fuzzy bond order:")
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=null).communicate(input=laplacian_order_operator)
            file_append(order_output, order_origin_output, "laplacian bond order:")

        dipole_output = Path.cwd() / Path(f"Multiwfn_desc/tmp_bond/{idx}-bond_dipole.txt")
        with open(dipole_output, "w") as dipole_outstd:
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=dipole_outstd).communicate(input=dipole_operator)

        mayer_order_data = deal_with_seperated_table(order_output, (mayer_order_str, "--------------"), "\n        ")
        wiberg_order_data = deal_with_seperated_table(order_output, (wiberg_order_str, "--------------"), "\n        ")
        mulliken_order_data = deal_with_seperated_table(order_output, (mulliken_order_str, "--------------"), "\n        ")
        fuzzy_order_data = deal_with_seperated_table(order_output, (fuzzy_order_str, "--------------"), "\n        ")
        laplacian_order_data = deal_with_seperated_table(order_output, (laplacian_order_str, "--------------"), "\n        ")
        dipole_data = get_newly_construct_dataframe(
            dipole_output,
            (" Input index of two atoms, e.g.", "Contribution to system dipole moment"),
            (" Bond dipole moment \(a\.u\.\):\n  X=.*Y=.*Z=.*Norm=", "\n"),
            "bond_dipole",
            re_escape_on_data=False,
        )
        for j, b in zip(range(len(bond_list)), bond_list):
            BondDescriptors.loc[s, f"Mayer_bond_order_bond{j}"] = mayer_order_data.loc[atoms[b[0]], atoms[b[1]]]
            BondDescriptors.loc[s, f"Wiberg_bond_order_bond{j}"] = wiberg_order_data.loc[atoms[b[0]], atoms[b[1]]]
            BondDescriptors.loc[s, f"Mulliken_bond_order_bond{j}"] = mulliken_order_data.loc[atoms[b[0]], atoms[b[1]]]
            BondDescriptors.loc[s, f"Fuzzy_bond_order_bond{j}"] = fuzzy_order_data.loc[atoms[b[0]], atoms[b[1]]]
            BondDescriptors.loc[s, f"Laplacian_bond_order_bond{j}"] = laplacian_order_data.loc[atoms[b[0]], atoms[b[1]]]
            BondDescriptors.loc[s, f"Bond_dipole_bond{j}"] = dipole_data.loc[j, "bond_dipole"]

        if not use_temp_file:
            shutil.move(order_output, order_output_store)

    BondDescriptors = deal_with_dataframe(BondDescriptors)

    return BondDescriptors


GlobalOrbitDescriptorName = [
    "HOMO_energy",
    "LUMO_energy",
    "HOMO_LUMO_energy_gap",
    "HOMO_ODI",
    "LUMO_ODI",
]

AtomOrbitDescriptorName = [
    "HOMO_ODI-atom",
    "LUMO_ODI-atom",
]


def get_orbit_descriptors(fchk_files: list, smiles_list: list = None, atoms_list: list = []) -> pd.DataFrame:
    """Use Multiwfn to get some molecular orbit descriptors from fchk file. see detail descriptor name in markdown file.
       default use the filename that drops '-' as smiles_list

    Args:
        `fchk_files (list)`: Multiwfn input files.\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `atoms_list (list, optional)`: atom number of atom properties that want to get. Defaults to None and not get atom properties
        Attention! Multiwfn atom number start from 1!\n

    Returns:
        `pd.DataFrame`: Molecular orbit descriptor dataframe
    """
    print("-----< Now generate orbit descriptors >-----")
    orbit_operator = (  # for molecular radius, define as
        0,  # Show molecular structure and view orbitals
        8,  # Orbital composition analysis
        3,  # Orbital composition analysis with Ros-Schuit (SCPA) partition
        "h",  # "h" for HOMO
        "l",  # "l" for LUMO
        0,  # input 0 to return
        -10,  # Return to main menu
        "q",  # Exit gracefully
    )
    atoms_list, all_atom_name = atoms_desc_name_generate(atoms_list, AtomOrbitDescriptorName, len(fchk_files))

    orbit_operator = operator_combine(orbit_operator)
    fchk_files = list_convert(fchk_files)

    AllOrbitDescriptorName = GlobalOrbitDescriptorName + all_atom_name
    OrbitDescriptors = pd.DataFrame(columns=AllOrbitDescriptorName)

    if smiles_list is None:
        smiles_list = smiles_list = ["".join(Path(f).stem.split("-")[1:]) for f in fchk_files]

    assert len(fchk_files) == len(smiles_list) == len(atoms_list)  # check if these lists are same length
    (Path.cwd() / Path("Multiwfn_desc/tmp_orbit")).mkdir(parents=True, exist_ok=True)
    for s, atoms, fchk_file in tqdm(zip(smiles_list, atoms_list, fchk_files)):
        idx = fchk_files.index(fchk_file)
        fchk_file = fchk_file_check(fchk_file)

        orbit_output = Path.cwd() / Path("orbit.txt")
        orbit_output_store = Path.cwd() / Path(f"Multiwfn_desc/tmp_orbit/{idx}-orbit.txt")

        with open(orbit_output, "w") as orbit_stdout:
            subprocess.Popen([Multiwfn, str(fchk_file)], stdin=subprocess.PIPE, stdout=orbit_stdout).communicate(input=orbit_operator)

        ODI_data = get_newly_construct_dataframe(
            orbit_output,
            ("Orbital delocalization index:", "\n"),
            ("Orbital delocalization index:", "\n"),
            desc_list="ODI",
            re_escape_on_paragraph=False,
        )
        with orbit_output.open("r") as f:
            orbit_text = f.read()
        context = re.findall("Composition of each atom:\n(.*?)\n\n", orbit_text, re.DOTALL)
        HOMO_ODI_atom_data = str2dataframe(context[0], remove=(17, 30))
        LUMO_ODI_atom_data = str2dataframe(context[-1], remove=(17, 30))

        # molecular CDFT properties
        if "alpha" in orbit_output.read_text():
            OrbitDescriptors.loc[s, "HOMO_energy"] = (
                get_data(orbit_output, "alpha-HOMO, energy:", "a.u.") + get_data(orbit_output, "beta-HOMO, energy:", "a.u.")
            ) * 0.5
            OrbitDescriptors.loc[s, "LUMO_energy"] = (
                get_data(orbit_output, "alpha-LUMO, energy:", "a.u.") + get_data(orbit_output, "beta-LUMO, energy:", "a.u.")
            ) * 0.5
            OrbitDescriptors.loc[s, "HOMO_LUMO_energy_gap"] = (
                get_data(orbit_output, "HOMO-LUMO gap of alpha orbitals:", "a.u.")
                + get_data(orbit_output, "HOMO-LUMO gap of beta orbitals:", "a.u.")
            ) * 0.5
        else:
            OrbitDescriptors.loc[s, "HOMO_energy"] = get_data(orbit_output, "HOMO, energy:", "a.u.")
            OrbitDescriptors.loc[s, "LUMO_energy"] = get_data(orbit_output, "LUMO, energy:", "a.u.")
            OrbitDescriptors.loc[s, "HOMO_LUMO_energy_gap"] = get_data(orbit_output, "HOMO-LUMO gap:", "a.u.")

        OrbitDescriptors.loc[s, "HOMO_ODI"] = ODI_data.loc[0, "ODI"]
        OrbitDescriptors.loc[s, "LUMO_ODI"] = ODI_data.loc[1, "ODI"]
        for i, atom in zip(range(len(atoms)), atoms):
            atom -= 1
            OrbitDescriptors.loc[s, f"HOMO_ODI-atom{i}"] = HOMO_ODI_atom_data.loc[atom, 0]
            OrbitDescriptors.loc[s, f"LUMO_ODI-atom{i}"] = LUMO_ODI_atom_data.loc[atom, 0]
        # move orbit.txt to stored
        shutil.move(orbit_output, orbit_output_store)

    OrbitDescriptors = deal_with_dataframe(OrbitDescriptors)
    return OrbitDescriptors


def get_all_wfn_descriptors(
    fchk_files: list,
    atoms_list: list = [],
    smiles_list: list = None,
    bond_list: list = [],
    angle_list: list = [],
    dihedral_list: list = [],
    calc_method: str = "",
    prop=(0, 1),
    kill=False,
    file_mode: int = 1,
    fchk_files_sort: bool = False,
) -> pd.DataFrame:
    """Use Multiwfn to generate quantum descriptors.

    Args:
        `fchk_files (list)`: Multiwfn input files. Attention! .glob() cannot give right file sort!\n
        `atoms_list (list, optional)`: atom number of atom properties that want to get. Defaults to None and not get atom properties
        Attention! Multiwfn atom number start from 1!\n
        `smiles_list (list, optional)`: smiles list for descriptor dataframe index. Defaults use fchk file name with '-' removed\n
        `bond_list (list, optional)`: used to represent a bond between bit `i` and bit `j` in a list of atoms. Defaults to [].\n
        `angle_list (list, optional)`: used to represent a angle in bit `i`, bit `j` and bit `k` in a list of atoms. Defaults to [].\n
        `dihedral_list (list, optional)`: used to represent a dihedral angle in bit `i`, bit `j`, bit `k` and bit `l` in a list of atoms. Defaults to [].\n
        `calc_method` (str, optional): define gaussian calculation base/method. defaults use b3lyp/6-31g*.\n
        `prop` (optional): set charge and spin. defaults to (0, 1) and results to `(0 1)\n(-1 2)\n(1 2)\n(2 1)`.\n
        `kill (bool)`: Whether to continue calculation after Gaussian calculation not converged. Default False.\n
        `file_mode (int,optional)`: get descriptors from temp_files in ./Multiwfn_desc/tmp_xxx
        `0` means do not read temp_files. `1` means read exist temp_files. `2` means only read temp_files.\n
        `fchk_files_sort (bool,optional)`: whether to sort fchk_files by number like 'x-'. default not sort.
        if no smiles_list, this option will be true!

    Returns:
        pd.DataFrame: all descriptors from multiwfn.
    """
    All_desc_df = []
    fchk_files = list_convert(fchk_files)
    try:
        atoms_list = atoms_list if type(atoms_list) == list else atoms_list.tolist()
        prop = prop if type(prop) == list or type(prop) == tuple else prop.tolist()
        if len(atoms_list) != 0:
            if type(atoms_list[0]) != list:
                atoms_list = [eval(atoms) for atoms in atoms_list]
        if type(prop) == list and type(prop[0]) == str:
            prop = [eval(p) for p in prop]
    except:
        print("Cannot Transfer atoms_list or prop to list properly. Check input infomations.")
        raise Exception

    if smiles_list == None or fchk_files_sort:
        fchk_files = sorted(fchk_files, key=lambda x: int(x.name.split("-")[0]))
    All_desc_df.append(
        get_CDFT_descriptors(
            fchk_files=fchk_files,
            atoms_list=copy(atoms_list),
            smiles_list=smiles_list,
            calc_method=calc_method,
            prop=prop,
            kill=kill,
            file_mode=file_mode,
        )
    )
    All_desc_df.append(
        get_ESP_descriptors(fchk_files=fchk_files, atoms_list=copy(atoms_list), smiles_list=smiles_list, file_mode=file_mode)
    )
    All_desc_df.append(get_molecular_shape(fchk_files=fchk_files, smiles_list=smiles_list, file_mode=file_mode))
    All_desc_df.append(get_orbit_descriptors(fchk_files=fchk_files, atoms_list=copy(atoms_list), smiles_list=smiles_list))
    if atoms_list != []:
        All_desc_df.append(
            get_internal_cordination(
                fchk_files=fchk_files,
                atoms_list=copy(atoms_list),
                smiles_list=smiles_list,
                bond_list=bond_list,
                angle_list=angle_list,
                dihedral_list=dihedral_list,
            )
        )
        All_desc_df.append(
            get_atom_descriptors(fchk_files=fchk_files, atoms_list=copy(atoms_list), smiles_list=smiles_list, file_mode=file_mode)
        )
        if bond_list != []:
            All_desc_df.append(
                get_bond_descriptors(
                    fchk_files=fchk_files, atoms_list=copy(atoms_list), bond_list=bond_list, smiles_list=smiles_list, file_mode=file_mode
                )
            )
    All_desc_df = pd.concat(All_desc_df, axis=1)

    # clean generated temp files.

    trash_files = [
        "opt_struct.txt",
        "orbit.txt",
        "multipole.txt",
        "bndmat.txt",
        "atom_moment.txt",
        "AtomESP.txt",
        "CDFT.txt",
        "bond_order.txt",
        "charge.txt",
        "energy_index.txt",
        "oxidation.txt",
    ]
    for trash_file in trash_files:
        (Path.cwd() / Path(trash_file)).unlink() if (Path.cwd() / Path(trash_file)).exists() else None

    return All_desc_df
