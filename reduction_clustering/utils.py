from pathlib import Path
from loguru import logger
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from tqdm import tqdm
from umap import UMAP
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering as AggCluster, SpectralClustering as SpecCluster
import matplotlib.pyplot as plt
import plotly.express as px


figure_path = Path(__file__).parent


def generate_viridis_color_sequence(n):
    def rgba_to_hex(rgba):
        rgba_int = [round(val * 255) for val in rgba[:3]]
        hex_values = [hex(val)[2:].zfill(2) for val in rgba_int]
        hex_color = "#" + "".join(hex_values)
        return hex_color

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    hex_colors = [rgba_to_hex(color) for color in colors]
    return hex_colors


def get_best_n_clusters(data_X, method, c_range=(2, 30), do_reduction=False):
    import numpy as np
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    n_clusters_range = range(c_range[0], c_range[1])

    if do_reduction:
        data_X = dimension_reduction(data_X, decomp_name="UMAP")

    result = []
    logger.info(f"Starting {method} analysis...")
    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, n_init="auto")
        fit_res = kmeans.fit(data_X)
        if method == "elbow":
            result.append(sum(np.min(cdist(data_X, kmeans.cluster_centers_, "euclidean"), axis=1)) / data_X.shape[0])
        elif method == "silhouette":
            result.append(silhouette_score(data_X, kmeans.labels_, metric="euclidean"))
        else:
            raise Exception(f"no method called: {method}.")
    if method == "elbow":
        plt.plot(n_clusters_range, result, "gx-")
    elif method == "silhouette":
        plt.plot(n_clusters_range, result, "r*-")
    plt.xlabel("k")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(f"best K value by {method} rule")
    plt.savefig(figure_path / Path(f"cluster_{method}_with_kmeans.png"), dpi=300)


def tSNE_perplexity_optimize(data_X, perplexity_range, predict_type=""):
    divergence = []
    for i in tqdm(perplexity_range):
        model = TSNE(n_components=2, init="pca", perplexity=i, verbose=0)
        reduced = model.fit_transform(data_X)
        divergence.append(model.kl_divergence_)
    print(divergence)
    fig = px.line(x=perplexity_range, y=divergence, markers=True)
    fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
    fig.update_traces(line_color="orange", line_width=1)
    fig.write_image(Path(__file__).parent / Path(f"tSNE_perpelxity_optimize.png"))


def dimension_reduction(data_X, decomp_name, rd=42):
    if decomp_name == "t-SNE":
        decomp_model = TSNE(n_components=2, verbose=2)
    elif decomp_name == "UMAP":
        decomp_model = UMAP(n_components=2, n_jobs=1, n_neighbors=20, min_dist=1, random_state=rd)
    elif decomp_name == "PCA":
        decomp_model = PCA(n_components=2)
    elif decomp_name == "ICA":
        decomp_model = FastICA(n_components=2, random_state=rd)
    elif decomp_name == "MDS":
        decomp_model = MDS(n_components=2, normalized_stress=False, verbose=2, n_jobs=24)
    elif decomp_name == "pass":
        decomp_model = None
    else:
        logger.error(f"No decomposation method called {decomp_name}.")
        raise Exception

    if decomp_name != "pass":
        logger.info(f"Start {decomp_name} decomposition...")
        decomp_data = decomp_model.fit_transform(data_X)
        logger.info(f"{decomp_name} decomposition done!")
        decomp_data = pd.DataFrame(decomp_data, columns=["component1", "component2"])
    else:
        decomp_data = pd.DataFrame()

    return decomp_data


def data_clustering(data_X, n_clusters, cluster_name, decomp_data, label_info, rd=42):
    if cluster_name == "by_atom" or cluster_name == "by_category":
        decomp_data["label"] = label_info
        random_point = [0 * len(set(label_info))]
    else:
        data_X = decomp_data.to_numpy()
        logger.info(f"Start {cluster_name} cluster...")
        if cluster_name == "kmeans":
            cluster_model = KMeans(n_clusters=n_clusters, random_state=rd, n_init="auto")
            # closest_point = cluster_model.transform(condition_X).argmin(axis=0)
            # closest_point_data = decomp_data.loc[closest_point]

        elif cluster_name == "BDSCAN":
            cluster_model = DBSCAN()
        elif cluster_name == "AggCluster":
            cluster_model = AggCluster(n_clusters=n_clusters)
        elif cluster_name == "SpecCluster":
            cluster_model = SpecCluster(n_clusters=n_clusters, affinity="nearest_neighbors")
        elif cluster_name == "kmeans-with-agglomeration":
            hierarchical = AggCluster(n_clusters=n_clusters)
            hierarchical_labels = hierarchical.fit_predict(data_X)
            initial_centroids = np.zeros((n_clusters, data_X.shape[1]))
            for i in range(n_clusters):
                initial_centroids[i, :] = np.mean(data_X[hierarchical_labels == i, :], axis=0)

            cluster_model = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1, random_state=42)
        else:
            raise Exception(f"No cluster method called {cluster_name}.")

        cluster_model = cluster_model.fit(data_X)
        decomp_data["label"] = [int(x) for x in cluster_model.labels_]
        logger.info(f"{cluster_name} cluster done!")

    return decomp_data


def reduction_clustering(data_X, decomp_name, cluster_name, n_clusters, label_info=None, rd=42):
    decomp_data = dimension_reduction(data_X, decomp_name, rd)
    decomp_data = data_clustering(data_X, n_clusters, cluster_name, decomp_data, label_info)
    # samples = decomp_data.groupby("label").apply(lambda x: x.sample(cluster_in_num))
    return decomp_data


def generate_rdkit_descriptors(input_csv, smiles_column):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import AllChem
    from rdkit.ML.Descriptors import MoleculeDescriptors

    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv)

    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in input CSV file.")
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptor_df = pd.DataFrame(columns=descriptor_names)
    for i, smiles in tqdm(enumerate(df[smiles_column]), total=len(df[smiles_column])):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            descriptors = calc.CalcDescriptors(mol)
            descriptor_df.loc[i] = descriptors
        else:
            descriptor_df.loc[i] = [None] * len(descriptor_names)

    print("Shape of the descriptor DataFrame:", descriptor_df.shape)
    
    descriptor_df = pd.concat([df[[smiles_column, "level"]], descriptor_df], axis=1)
    output_csv = Path(str(input_csv.stem) + "_rdkit_descriptors.csv")
    descriptor_df.to_csv(output_csv, index=False)


def batch_draw_mol(data_df, column_name, save_path, max_length=30, clean_origin=False, show_atom_numbers=False):
    from rdkit import Chem
    from rdkit.Chem import Draw
    from pathlib import Path

    picture_path = Path(save_path)
    if not picture_path.exists():
        picture_path.mkdir()
    if clean_origin:
        [p.unlink() for p in picture_path.glob("*.jpg")]

    smiles = data_df.loc[:, column_name].drop_duplicates().tolist()
    smiles = [s for s in smiles if s != "blanck_cell"]
    smiles_lists = [smiles[i : i + max_length] for i in range(0, len(smiles), max_length)]

    for i, smiles_list in enumerate(smiles_lists):
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

        if show_atom_numbers:
            for mol in mol_list:
                if mol is not None:
                    for atom in mol.GetAtoms():
                        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

        img = Draw.MolsToGridImage(mol_list, molsPerRow=3, subImgSize=(500, 300), legends=None, returnPNG=False)
        img.save(picture_path / f"{i}.jpg")


def normalize_columns(arr):
    normalized_arr = np.zeros_like(arr, dtype=float)
    # print(np.where(arr == "blanck_cell"))
    for col_index in range(arr.shape[1]):
        col = arr[:, col_index]
        col_norm = np.linalg.norm(col)
        if col_norm == 0:
            normalized_arr[:, col_index] = col
        else:
            normalized_arr[:, col_index] = col / col_norm
    return normalized_arr
