from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def draw_radar_chart(result_type, substrate_type):
    exp_df = pd.read_csv(f"data/experimental_data_results_of_{substrate_type}.csv") 
    exp_df = exp_df[(exp_df[f"condition1_{result_type}"] != 0) & (exp_df[f"condition2_{result_type}"] != 0)]
    exp_df.fillna(True, inplace=True)
    exp_df = exp_df[exp_df["select_tag"] != "no_use"]
    
    # categories = [f"a{i+1}" for i in range(len(exp_df))]
    if substrate_type == "alcohol":
        categories = [
            "3a",
            "3b",
            "3c",
            "3e",
            "3f",
            "3g",
            "3i",
            "3j",
            "3k",
            "3m",
            "3o",
            "3p",
            "3q",
            "3r",
            "3s",
            "3t",
            "3u",
            "3v",
            "3x",
            "3y",
        ]
    elif substrate_type == "aldehyde":
        categories = ["4a", "4b", "4c", "4d", "4e", "4f", "4g", "4h", "4i", "4j", "4k", "4l", "4m", "4n", "4o", "4p", "4q", "4r"]
    num_vars = len(categories)

    condition1_res = exp_df[f"condition1_{result_type}"].tolist()
    condition2_res = exp_df[f"condition2_{result_type}"].tolist()

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    condition1_res += condition1_res[:1]  
    condition2_res += condition2_res[:1]  

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    palette = ["#5aa497", "#e5833b"]
    ax.plot(angles, condition1_res, label=f"HO", color=palette[0], linewidth=2)
    ax.fill(angles, condition1_res, alpha=0.3, color=palette[0])
    ax.plot(angles, condition2_res, label=f"BO", color=palette[1], linewidth=2)
    ax.fill(angles, condition2_res, alpha=0.3, color=palette[1])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")

    ax.yaxis.grid(True, color="grey", linestyle="dotted", linewidth=0.8)
    if result_type == "yield":
        ax.set_ylim(0, 80) 
    elif result_type == "ee":
        ax.set_ylim(30, 100)

    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"result_pictures/radar_chart_{result_type}_for_{substrate_type}.png", dpi=300)

sns.set_theme(style="whitegrid")

level = 0
decomp_method = "UMAP"
input_df = pd.read_csv(f"results_structure/level_{level}_alcohol_decomped.csv", index_col=0)

for r in ["yield", "ee"]:
    for s in ["alcohol", "aldehyde"]:
        draw_radar_chart(r, s)
