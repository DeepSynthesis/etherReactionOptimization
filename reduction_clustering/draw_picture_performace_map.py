import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as mcolors
import seaborn as sns
from rdkit import Chem

level = 0
decomp_method = "UMAP"
reduction_type = "alcohol"
input_df = pd.read_csv(f"{reduction_type}_results_structure/level_{level}_decomped.csv", index_col=0)
exp_df = pd.read_csv(f"data/experimental_data_results_of_{reduction_type}.csv") 
c_id = 1

unique_labels = input_df["label"].unique()
num_labels = len(unique_labels)

palette = sns.color_palette("plasma", num_labels)
plt.figure(figsize=(8, 6))

unique_labels = sorted(input_df["label"].unique())
for label, scatter_color in zip(unique_labels, palette):
    label_data = input_df[input_df["label"] == label]
    sns.kdeplot(x=label_data["component1"], y=label_data["component2"], fill=True, alpha=0.3, color=scatter_color)

scatter_plot = sns.scatterplot(data=input_df, x="component1", y="component2", hue="label", palette=palette, s=25, alpha=0.1)

exp_df["react_tag"] = (exp_df["condition1_yield"] != 0) | (exp_df["condition2_yield"] != 0)
exp_df["SMILES"] = exp_df["SMILES"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True))

merged_df = exp_df.merge(input_df[["alcohol_SMILES", "component1", "component2"]], how="left", left_on="SMILES", right_on="alcohol_SMILES")

exp_df["component1"] = merged_df["component1"]
exp_df["component2"] = merged_df["component2"]

unmatched_smiles = exp_df[exp_df["component1"].isna() | exp_df["component2"].isna()]["SMILES"]

if not unmatched_smiles.empty:
    print("The following smiles were not found in input_df:")
    print(unmatched_smiles.to_list())

norm = mcolors.Normalize(vmin=exp_df[f"condition{c_id}_ee"].min(), vmax=exp_df[f"condition{c_id}_ee"].max())

colorcmap = sns.color_palette("flare", as_cmap=True)
norm = mcolors.Normalize(vmin=exp_df[f"condition{c_id}_ee"].min(), vmax=exp_df[f"condition{c_id}_ee"].max())
exp_df["color"] = exp_df.apply(lambda row: mcolors.to_hex(colorcmap(norm(row[f"condition{c_id}_ee"]))), axis=1)
exp_df["size"] = exp_df.apply(lambda row: 75 if not row["react_tag"] else row[f"condition{c_id}_yield"] * 2 + 100, axis=1)
exp_df = exp_df.iloc[::-1].reset_index(drop=True)
plt.scatter(
    exp_df["component1"],
    exp_df["component2"],
    c=exp_df["color"],
    s=exp_df["size"],
    alpha=1,
    edgecolors="white",
    linewidths=0.5,
)

# Add color bar for condition1_ee
sm = plt.cm.ScalarMappable(cmap=colorcmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
cbar.ax.tick_params(labelsize=12)
cbar.set_label("ee value scale", rotation=270, labelpad=30, fontsize=14)  # Rotate label and adjust padding

plt.xlabel("UMAP1", size=14)
plt.ylabel("UMAP2", size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend().set_visible(False)
plt.title(" ")
exp_df.to_csv('temp.csv')

plt.savefig(f"{reduction_type}_results_structure/final_figure_{decomp_method}_for_{level}OH_for_condition{c_id}.png", dpi=300) 
