from genericpath import isfile
from pathlib import Path
from collections.abc import Iterable
import pandas as pd
import numpy as np
import re


def operator_combine(operator_set: tuple) -> str:
    """transfer a operator set into a operator string."""
    op_str = ""
    for op in operator_set:
        op_str += str(op) + "\n"
    return op_str.encode("utf-8")


def get_data(
    context: str, keywords_above: str, keywords_below: str, is_file: bool = True, re_escape_on: bool = True, out_type: str = "first"
) -> str:
    """get descriptor data from given keywords."""
    if is_file:
        with Path(context).open("r") as f:
            text = f.read()
    else:
        text = context
    if re_escape_on:
        pat = re.escape(keywords_above) + "(.*?)" + re.escape(keywords_below)
    else:
        pat = keywords_above + "(.*?)" + keywords_below

    if out_type == "first":
        data = re.findall(pat, text)[0].strip()
    elif out_type == "last":
        data = re.findall(pat, text)[-1].strip()
    elif out_type == "all":
        data = re.findall(pat, text)
        return [x.strip() for x in data]
    else:
        raise NameError(f"no out_type: {out_type}")
    if type(data) == str:
        if data == "":
            data = 0.0
        elif len(data.split()) > 1:
            data = data.split()
        else:
            try:
                data = float(data)
            except:
                raise TypeError("wrong in data.")
    return data


def str2dataframe(context: str, remove: tuple = (0, None), reset_index: bool = False, reset_column: bool = False) -> pd.DataFrame:
    if reset_column:
        new_column, context = [0] + context.split("\n")[0].split(), context.split("\n")[1:]
        new_column = [int(c) for c in new_column]
    else:
        new_column, context = None, context.split("\n")
    data_df = pd.DataFrame([x[remove[0] : remove[1]].split() for x in context], dtype="float64", columns=new_column)
    if reset_index:
        data_df.set_index(0, inplace=True)
        data_df.dropna(how="all", inplace=True)
        data_df.index = data_df.index.astype(int)
    return data_df


def get_dataframe(
    context: str, keywords_above: str, keywords_below: str, remove: tuple = (0, None), reset_index: bool = False, is_file: bool = True
) -> str:
    """get descriptor data from given keywords."""
    if is_file:
        with Path(context).open("r") as f:
            text = f.read()
    else:
        text = context
    start = text.find(keywords_above) + len(keywords_above)
    end = start + text[start:].find(keywords_below)
    content = text[start:end]

    return str2dataframe(content, remove=remove, reset_index=reset_index)


def deal_with_seperated_table(file_name: str, df_keywords: (str, str), df_split_keywords: str) -> pd.DataFrame:
    with open(file_name, "r") as f:
        text = f.read()
    pat = re.escape(df_keywords[0]) + r"(.*?)" + re.escape(df_keywords[1])
    origin_dataframes = re.findall(pat, text, re.DOTALL)[0]
    dataframe_list = []
    origin_dataframes = origin_dataframes.split(df_split_keywords)
    for df_text in origin_dataframes:
        if df_text == "":
            continue
        df_tmp = str2dataframe(df_text, reset_index=True, reset_column=True)
        dataframe_list.append(df_tmp)

    return pd.concat(dataframe_list, axis=1)


def list_convert(obj) -> list:
    """convert non-list object to list."""
    if isinstance(obj, Iterable) and type(obj) != str:
        return [o for o in obj]
    else:
        return [obj]


def generate_desc_names(desc_name: list, item_num: int) -> list:
    """generate a descriptor name list"""
    all_names = []
    for x in range(item_num):
        all_names += [i + str(x) for i in desc_name]
    return all_names


def atoms_desc_name_generate(atoms_list: list, desc_name: list, len_list: str) -> list:
    """from atom_list and desc_name generate a all atom descriptor nams."""
    if atoms_list != []:
        for x in atoms_list:
            assert len(x) == len(atoms_list[0])
        # assert all(len(x) == len(atoms_list[0]) for x in atoms_list)  # to check if all atom lists is with the same length.
        all_atom_name = generate_desc_names(desc_name=desc_name, item_num=(len(atoms_list[0])))
    else:
        for _ in range(len_list):
            atoms_list.append([])
        all_atom_name = []
    return atoms_list, all_atom_name


def fchk_file_check(fchk_file):
    """check if fchk file exists."""
    fchk_file = Path(fchk_file)
    if not fchk_file.exists():
        raise FileExistsError(f"there is no file called '{str(fchk_file)}'")
    elif fchk_file.suffix != ".fchk":
        raise FileExistsError(f"file '{str(fchk_file)}' is not with right suffix 'fchk'")
    return fchk_file


def internal_name_generate(bond_num: int, angle_num: int, dihedral_num: int):
    """generate internal descriptors names."""
    all_desc_name = []
    for i in range(bond_num):
        all_desc_name.append(f"bond_{i}")
    for j in range(angle_num):
        all_desc_name.append(f"angle_{j}")
    for k in range(dihedral_num):
        all_desc_name.append(f"dihedral_{k}")
    return all_desc_name


def load_cordinations(file_name="opt_struct.txt") -> list[list]:
    """load cordinations from .xyz file."""
    with open(Path.cwd() / file_name, "r") as cor_file:
        all_cors = cor_file.readlines()[2:]  # Remove the leading header line
        all_cors = [atom_line.split()[1:4] for atom_line in all_cors]

    return all_cors


def bond_length_calc(p1: list, p2: list) -> float:
    """calculate distance of 2 points."""
    assert len(p1) == len(p2) == 3
    v = np.array(p2) - np.array(p1)
    return np.linalg.norm(v)


def angle_calc(p1: list, p2: list, p3: list) -> float:
    """calculate angle of 3 points."""
    assert len(p1) == len(p2) == len(p3) == 3
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_angle) / np.pi * 180.0


def dihedral_angle_calc(p1: list, p2: list, p3: list, p4: list) -> float:
    """calculate dihedral angle of 4 points."""
    assert len(p1) == len(p2) == len(p3) == len(p4) == 3
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p4) - np.array(p3)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_angle) / np.pi * 180.0


def file_append(appended_file: str, append_file: str, info: str = "") -> None:
    """append one file on anther file."""

    with open(appended_file, "a") as f_appended, open(append_file, "r") as f_append:
        f_appended.write("\n\n" + info + "\n")
        f_appended.write(f_append.read())
        f_appended.write("----------------------------\n")


def get_newly_construct_dataframe(
    file_name: str,
    paragraph_keywords: (str, str),
    data_keywords: (list, list),
    desc_list: list,
    re_escape_on_paragraph: bool = True,
    re_escape_on_data: bool = True,
) -> pd.DataFrame:
    data_keywords = (list_convert(data_keywords[0]), list_convert(data_keywords[1]))
    desc_list = list_convert(desc_list)
    assert len(data_keywords[0]) == len(data_keywords[1]) == len(desc_list)

    with open(file_name, "r") as f:
        text = f.read()
    if re_escape_on_paragraph:
        pat = re.escape(paragraph_keywords[0]) + r".*?" + re.escape(paragraph_keywords[1])
    else:
        pat = paragraph_keywords[0] + r".*?" + paragraph_keywords[1]
    paragraphs = re.findall(pat, text, re.DOTALL)
    # from IPython import embed; embed()
    output_df = pd.DataFrame(columns=desc_list)

    idx = 0
    for p in paragraphs:
        for desc, above, below in zip(desc_list, data_keywords[0], data_keywords[1]):
            output_df.loc[idx, desc] = get_data(p, above, below, is_file=False, re_escape_on=re_escape_on_data)
        idx += 1

    return output_df


def deal_with_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """change `NaN` and empty unit to 0.0 and change str to number."""
    df.replace("NaN", 0.0, inplace=True)
    df.fillna(0.0, inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def file_mode_deal(file_mode, temp_file_path):
    """trasfer file_mode into whether to use temp file."""
    if file_mode == 0:
        return False
    elif file_mode == 1:
        if type(temp_file_path) == list:
            return all(p.exists() for p in temp_file_path)
        else:
            return temp_file_path.exists()
    elif file_mode == 2:
        return True
    else:
        raise Exception("ERROR file_mode input...")
