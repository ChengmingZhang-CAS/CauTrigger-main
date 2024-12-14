import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import torch
import random
import anndata
import warnings
import sys
from typing import Iterable
import os

os.environ["OMP_NUM_THREADS"] = '1'
#
# from rich.console import Console
# from rich.progress import track as track_base
from tqdm import tqdm as tqdm_base
from scipy.linalg import norm
from math import ceil, floor
from collections.abc import Iterable as IterableClass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from anndata._core.sparse_dataset import SparseDataset
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
import math
import scanpy as sc
from scipy import sparse
from scipy.stats import norm as normal
from sklearn.neighbors import NearestNeighbors
from velocyto.estimation import colDeltaCorpartial

Number = Union[int, float]


# ======================================== data.utils ========================================
def _check_nonnegative_integers(
        data: Union[pd.DataFrame, np.ndarray, sp_sparse.spmatrix, h5py.Dataset]
):
    """Approximately checks values of data to ensure it is count data."""

    # for backed anndata
    if isinstance(data, h5py.Dataset) or isinstance(data, SparseDataset):
        data = data[:100]

    if isinstance(data, np.ndarray):
        data = data
    elif issubclass(type(data), sp_sparse.spmatrix):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    else:
        raise TypeError("data type not understood")

    n = len(data)
    inds = np.random.permutation(n)[:20]
    check = data.flat[inds]
    return ~np.any(_is_not_count(check))


def _is_not_count(d):
    return d < 0 or d % 1 != 0


def read_txt(filename, dtypefloat=True):
    data = {}
    f = open(filename, "r")
    line = f.readlines()
    flag = 0
    for l in line:
        flag += 1
        t = l.split()
        if flag == 1:
            title = [eval(t[k]) for k in range(len(t))]
        else:
            if dtypefloat:
                data[eval(t[0])] = [float(t[k]) for k in range(1, len(t))]
            else:
                data[eval(t[0])] = [eval(t[k]) for k in range(1, len(t))]
    f.close()
    df = pd.DataFrame(data, index=title)
    return df.T


# ======================================== module.utils ========================================
def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c) + (1.0 - lmbd) / 2.0 * torch.pow(c, 2)


def regularizer_l12(c, lmbd=1.0):
    return torch.norm(torch.norm(c, p=1, dim=1), p=2)


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def get_anchor_index(data, n_pcs=50, n_clusters=1000):
    """get the anchor sample index."""
    n_pcs = min(data.shape[1], n_pcs)
    pca = PCA(n_components=n_pcs, svd_solver="arpack", random_state=0)
    z = pca.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters).fit(z)
    dist_mat = cdist(z, kmeans.cluster_centers_)
    index = dist_mat.argmin(axis=0)
    return index


def get_sample_index(data, train_size=0.9, validation_size=0.1, n_pcs=50, n_sample_ref=1000):
    """get the splited sample index."""
    n_samples = data.shape[0]
    n_train = ceil(train_size * n_samples)
    n_val = floor(n_samples * validation_size)
    random_state = np.random.RandomState(seed=42)
    permutation = random_state.permutation(data.shape[0])
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val: (n_val + n_train)]
    test_idx = permutation[(n_val + n_train):]
    train_data = data[train_idx]
    n_pcs = min(train_data.shape[1], n_pcs)
    pca = PCA(n_components=n_pcs, svd_solver="arpack", random_state=0)
    z = pca.fit_transform(train_data)
    kmeans = KMeans(n_clusters=n_sample_ref).fit(z)
    dist_mat = cdist(z, kmeans.cluster_centers_)
    anchor_idx = dist_mat.argmin(axis=0)
    return train_idx, val_idx, test_idx, anchor_idx


def get_top_coeff(coeff, non_zeros=1000, ord=1):
    coeff_top = torch.tensor(coeff)
    N, M = coeff_top.shape
    non_zeros = min(M, non_zeros)
    values, indices = torch.topk(torch.abs(coeff_top), dim=1, k=non_zeros)
    coeff_top[coeff_top < values[:, -1].reshape(-1, 1)] = 0
    coeff_top = coeff_top.data.numpy()
    if ord is not None:
        coeff_top = coeff_top / norm(coeff_top, ord=ord, axis=1, keepdims=True)
    return coeff_top


def get_sparse_rep(coeff, non_zeros=1000):
    N, M = coeff.shape
    non_zeros = min(M, non_zeros)
    _, index = torch.topk(torch.abs(coeff), dim=1, k=non_zeros)

    val = coeff.gather(1, index).reshape([-1]).cpu().data.numpy()
    indicies = index.reshape([-1]).cpu().data.numpy()
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sp_sparse.csr_matrix((val, indicies, indptr), shape=coeff.shape)
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


# ======================================== model.utils ========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_control_scores(probs, control_scores_inc, control_scores_dec, metric='score', sample_size=None, alpha=0.5):
    """
    Plot control scores and probabilities.
    """
    if isinstance(control_scores_inc, pd.DataFrame):
        control_scores_inc = control_scores_inc[metric].to_numpy()
    if isinstance(control_scores_dec, pd.DataFrame):
        control_scores_dec = control_scores_dec[metric].to_numpy()

    # Downsample if sample_size is specified and less than the total number of samples
    if sample_size is not None and sample_size < len(probs):
        indices = np.random.choice(len(probs), sample_size, replace=False)
        control_scores_inc = control_scores_inc[indices]
        control_scores_dec = control_scores_dec[indices]
        probs = probs[indices]

    # Find the maximum control score for scaling
    max_control_score = max(np.max(np.abs(control_scores_inc)), np.max(np.abs(control_scores_dec)))

    # Plotting
    plt.figure(figsize=(15, 10))

    # Scatter plot for probabilities on y-axis
    plt.scatter(np.zeros(len(probs)), probs, color='blue', alpha=alpha, label='Probability')

    for i, prob in enumerate(probs):
        # Scale control scores to the maximum control score
        scaled_inc_score = (control_scores_inc[i] / max_control_score) * 0.1
        scaled_dec_score = (control_scores_dec[i] / max_control_score) * 0.1

        # Bar plot for increase scores on the right (red)
        plt.barh(prob, scaled_inc_score, height=0.01, color='red', alpha=alpha, left=0)
        # Bar plot for decrease scores on the left (blue)
        plt.barh(prob, -scaled_dec_score, height=0.01, color='blue', alpha=alpha, left=0)

    plt.xlabel(f'Control {metric.capitalize()}')
    plt.ylabel(f'Probability')
    plt.title(f'Control {metric.capitalize()} and Probability for Each Sample')

    # Create custom legend
    red_patch = mpatches.Patch(color='red', label='Increase Control')
    blue_patch = mpatches.Patch(color='blue', label='Decrease Control')
    plt.legend(handles=[red_patch, blue_patch])

    plt.xlim(-0.15, 0.15)  # Adjust the x-axis limits for better visualization
    plt.show()


def plot_control_scores_by_category(adata, control_details_inc, control_details_dec, metric='score'):
    """
    Plot control scores by category (label) for increase and decrease conditions with specific color schemes.
    """
    # Extract the specific metric
    scores_inc = control_details_inc[metric].to_numpy()
    scores_dec = control_details_dec[metric].to_numpy()

    # Extract sample categories
    labels = adata.obs['labels']
    labels.index = pd.RangeIndex(start=0, stop=len(labels))

    # Create DataFrame
    df_scores = pd.DataFrame({
        'Increase': scores_inc,
        'Decrease': scores_dec,
        'Label': labels
    })

    # Convert data to long format
    df_long = pd.melt(df_scores, id_vars='Label', value_vars=['Increase', 'Decrease'], var_name='Variable', value_name='Score')

    # Plot boxplot with specific color scheme
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_long, x='Label', y='Score', hue='Variable',
                palette={'Increase': sns.color_palette("Reds")[2], 'Decrease': sns.color_palette("Blues")[2]},
                hue_order=['Decrease', 'Increase'])  # Adjusting the order of categories
    plt.title(f'Control {metric.capitalize()} for Increase vs Decrease Probabilities by Label')
    plt.xlabel('Sample Label')
    plt.ylabel(f'Control {metric.capitalize()}')
    plt.show()

    # Plot scatter plot with specific color scheme
    plt.figure(figsize=(12, 6))
    plt.scatter(df_scores.index, df_scores['Increase'], label='Increase', color='red', alpha=0.6)
    plt.scatter(df_scores.index, df_scores['Decrease'], label='Decrease', color='blue', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel(f'Control {metric.capitalize()}')
    plt.title(f'Control {metric.capitalize()} for Each Sample by Label')
    plt.legend()
    plt.show()

    # Plot line plot with specific color scheme
    plt.figure(figsize=(12, 6))
    plt.plot(df_scores['Increase'], label='Increase', color='red', alpha=0.7)
    plt.plot(df_scores['Decrease'], label='Decrease', color='blue', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel(f'Control {metric.capitalize()}')
    plt.title(f'Control {metric.capitalize()} for Each Sample by Label')
    plt.legend()
    plt.show()


def plot_3d_state_transition(adata, sample_indices=None, use_pca=True, concat_pca=True, feature1=None, feature2=None,
                             n_components=2, max_samples=10, smooth=2, draw_contours=True):
    """
    Plot the state transition for specified samples in the dataset.
    The transition can be visualized using PCA or specified causal features.
    Limits the number of samples plotted to 'max_samples'.
    """
    # Create a custom color map
    cmap = plt.cm.viridis  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

    # If no specific samples are provided, plot all
    if sample_indices is None:
        sample_indices = list(adata.uns['causal_update'].keys())

    # Limit the number of samples to plot
    if len(sample_indices) > max_samples:
        print(f"Too many samples to plot. Only plotting the first {max_samples} samples.")
        sample_indices = sample_indices[:max_samples]

    # Iterate over specified samples for plotting
    for sample_idx in sample_indices:
        update_data = adata.uns['causal_update'][sample_idx]
        sampling_data = adata.uns['causal_sampling'][sample_idx]
        control_score = adata.uns["control_details"]['score'][sample_idx]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if use_pca:
            pca = PCA(n_components=n_components)
            combined_data = np.vstack((sampling_data.iloc[:, :-1], update_data[update_data.columns[3:]]))
            combined_df = pd.DataFrame(data=combined_data, columns=sampling_data.columns[:-1])
            if concat_pca:
                pca.fit(combined_df)
            else:
                pca.fit(sampling_data.iloc[:, :-1])
                # pca.fit(update_data[update_data.columns[3:]])
            pca_result = pca.transform(sampling_data.iloc[:, :-1])  # Exclude 'prob' column
            x_surf, y_surf, z_surf = pca_result[:, 0], pca_result[:, 1], sampling_data['prob'].values
            pca_path = pca.transform(update_data[update_data.columns[3:]])
            x_path, y_path, z_path = pca_path[:, 0], pca_path[:, 1], update_data['prob'].values
        else:
            if feature1 is None or feature2 is None:
                raise ValueError("Please specify both feature1 and feature2 for non-PCA plotting.")
            x_surf, y_surf, z_surf = sampling_data[feature1], sampling_data[feature2], sampling_data['prob']
            x_path, y_path, z_path = update_data[feature1].values, update_data[feature2].values, update_data['prob'].values
        # Interpolation for smoother surface
        x_combined = np.concatenate((x_surf, x_path))
        y_combined = np.concatenate((y_surf, y_path))
        z_combined = np.concatenate((z_surf, z_path))
        x_min = x_combined.min()
        x_max = x_combined.max()
        y_min = y_combined.min()
        y_max = y_combined.max()
        x_range = np.linspace(x_min, x_max, 200)
        y_range = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x_range, y_range)

        rbf = Rbf(x_combined, y_combined, z_combined, function='linear', smooth=smooth)
        Z = rbf(X, Y)
        # Z = griddata((x_combined, y_combined), z_combined, (X, Y), method='linear')  # "cubic" or "nearest"
        z_path = np.clip(z_path, Z.min(), Z.max())

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
        ax.plot_wireframe(X, Y, Z, color='black', alpha=0.5, linewidth=0.3)
        # scatter = ax.scatter(x_path, y_path, z_path, c=z_path, cmap=cmap, vmin=0, vmax=1, s=1, alpha=0.5)
        ax.plot(x_path, y_path, z_path, color='red', linewidth=5)  # Plot the path
        ax.view_init(elev=30, azim=30)  # Adjust the elevation and azimuth angles as needed
        if draw_contours:
            ax.contour(X, Y, Z, zdir='z', offset=ax.get_zlim()[0], cmap=cmap)
        # Annotate axes
        ax.text(x_path[0], y_path[0], z_path[0], 'Start', color='black', fontsize=12)
        ax.text(x_path[-1], y_path[-1], z_path[-1], 'End', color='black', fontsize=12)
        ax.set_zlim(0, 1)
        ax.set_xlabel(feature1 if not use_pca else 'PC1')
        ax.set_ylabel(feature2 if not use_pca else 'PC2')
        ax.set_zlabel('Probability')
        plt.title(f'State Transition Plot for Sample {sample_idx} (Control Score: {control_score:.2f})')
        plt.colorbar(surf, label='Probability', shrink=0.5, pad=0.1)
        plt.show()


def plot_causal_feature_transitions(adata, sample_indices=None, features=None, max_features=10):
    """
    Plot the transitions of multiple features against probability for specified samples.
    Adjusted to handle overlapping x-axis labels.
    """
    # If no specific sample indices are provided, plot all
    if sample_indices is None:
        sample_indices = adata.uns['causal_update'].keys()

    # If no specific features are provided, use all causal features
    if features is None:
        features = adata.uns['causal_update'][list(sample_indices)[0]].columns[3:]

    # Check if the number of features exceeds the maximum limit
    if len(features) > max_features:
        print(f"Warning: Too many features to plot. Only plotting the first {max_features} features.")
        features = features[:max_features]

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis')

    # Determine the layout of subplots
    n_features = len(features)
    n_cols = 3  # Adjust the number of columns
    n_rows = (n_features + n_cols - 1) // n_cols  # Calculate the required number of rows

    # Adjust figure size to avoid crowding
    plt.figure(figsize=(n_cols * 6, n_rows * 5))  # Increase figure size

    # Iterate over specified sample indices for plotting
    for sample_idx in sample_indices:
        # Extract the update data for the sample
        update_data = adata.uns['causal_update'][sample_idx]
        control_score = adata.uns["control_details"]['score'][sample_idx]

        # Normalize probability values for color mapping
        norm = plt.Normalize(vmin=0, vmax=1)

        for i, feature in enumerate(features):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            colors = cmap(norm(update_data['prob']))
            ax.scatter(update_data[feature], update_data['prob'], c=colors, marker='o', s=10)  # Adjust point size
            # Add start and end annotations
            ax.text(update_data[feature].iloc[0], update_data['prob'].iloc[0], 'Start', color='black')
            ax.text(update_data[feature].iloc[-1], update_data['prob'].iloc[-1], 'End', color='black')

            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Probability')
            # ax.set_xlabel(f'Feature Value ({feature})', fontsize=9)  # Adjust font size
            # Adjust font size and label positions
            if (i % n_cols) == n_cols - 1 or i == len(features) - 1:  # Only label the bottom row of each column
                ax.set_xlabel('Feature Value', fontsize=9)  # Generic label for x-axis
            else:
                ax.set_xlabel('')  # No label for other rows 还是有问题
            ax.set_ylabel('Probability', fontsize=9)  # Adjust font size
            ax.set_title(f'Feature {feature}', fontsize=10)  # Adjust font size
            ax.set_ylabel('Probability', fontsize=9)  # Adjust font size
            ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1
            ax.set_title(f'Feature {feature}', fontsize=10)  # Adjust font size

            # Format x-axis to reduce decimal places
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.suptitle(f'Transitions of Multiple Features for Sample {sample_idx} (Control Score: {control_score:.2f})', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to avoid overlap
        plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust horizontal and vertical spacing
        plt.show()


def select_features(df, threshold=None, topk=None, elbow=False):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    selected_features = []
    for column in df.columns:
        sorted_column = df[column].sort_values(ascending=False)

        if elbow:
            # 计算每个特征的权重
            weights = sorted_column.values
            # 计算累加的权重
            cumulative_weights = weights.cumsum()
            # 寻找权重累加变化较大的拐点
            diff = pd.Series(cumulative_weights).diff()
            elbow_point = diff.idxmax() + 1 if diff.idxmax() is not None else 1
            selected = pd.Series(0, index=df.index)
            selected[sorted_column.nlargest(elbow_point).index] = 1

        else:
            if threshold and not topk:
                cum_sum = sorted_column.cumsum()
                selected = (cum_sum <= threshold).astype(int)
                if selected.sum() == 0:
                    selected[sorted_column.index[0]] = 1
            elif topk:
                top_k_features = sorted_column.nlargest(topk).index
                selected = pd.Series(0, index=df.index)
                selected[top_k_features] = 1
            else:
                raise ValueError('Please pass valid argument!')

        selected = pd.Series(selected, name=column)
        selected_features.append(selected)
    selected_df = pd.concat(selected_features, axis=1)
    selected_df.columns = df.columns
    return selected_df.reindex(df.index)


def plot_vector_field(adata,state_pair=None,KO_Gene=None,state_obs=None,embedding_name=None,method=None,
                      sampled_fraction=1, min_mass=0.008,scale=0.1, save_dir=None,smooth=0.8,n_grid=40,
                     draw_single=None,n_suggestion=12,show=True,dot_size=None,run_suggest_mass_thresholds=False):
    
    def replace_with_CauTrigger(adata, corrcoef, state_pair=state_pair, state_obs=state_obs, method=method):
        corrcoef_ = corrcoef.copy()
        if method == 'prob':
            probs = adata.obs['probs'].copy()
            probs_pert = adata.obs['probs_pert'].copy()
        if method == 'logits':
            probs = adata.obs['logits'].copy()
            probs_pert = adata.obs['logits_pert'].copy()
        neigh_ixs = adata.uns['neigh_ixs'].copy()
        target_index = np.where((adata.obs[state_obs] == state_pair[0]) | (adata.obs[state_obs] == state_pair[1]))[0]  # 484.得到的是真实的索引而不是adata.obs的index
        nontarget_neighbor_index = np.setdiff1d(np.unique(neigh_ixs[target_index, :].flatten()), target_index)  # 451
        nonneighbor_index = np.setdiff1d(np.setdiff1d(np.arange(probs.shape[0]), target_index), nontarget_neighbor_index)  # 1484
        for i in target_index:
            cols = np.setdiff1d(np.union1d(target_index, nontarget_neighbor_index), i)  # 934
            cor =  - abs(probs_pert[i] - probs[cols])  # cor越大，代表向这个细胞转移的概率越大
            corrcoef_[i, cols] = cor
        # 其他两类的行全部变成0就行？
        corrcoef_[np.union1d(nontarget_neighbor_index, nonneighbor_index), :] = 0
        
        return corrcoef_

    def estimate_transition_prob(adata, state_pair=state_pair,state_obs=state_obs, embedding_name=embedding_name, n_neighbors=None,
                                 sampled_fraction=sampled_fraction, sigma_corr=0.05, replace_prob=None,draw_single=None):
        sampling_probs = (0.5, 0.1)
        # X = _adata_to_matrix(self.adata, "imputed_count")
        X = adata.layers["imputed_count"].transpose().copy()
        # delta_X = _adata_to_matrix(self.adata, "delta_X")
        delta_X = adata.layers["delta_X"].transpose().copy()
    
        embedding = adata.obsm[embedding_name].copy()
    
        if n_neighbors is None:
            n_neighbors = int(adata.shape[0] / 5)
    
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=4)
        nn.fit(embedding)  # NOTE should support knn in high dimensions
        embedding_knn = nn.kneighbors_graph(mode="connectivity")
    
        # Pick random neighbours and prune the rest
        neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
        p = np.linspace(sampling_probs[0], sampling_probs[1], neigh_ixs.shape[1])
        p = p / p.sum()
    
        sampling_ixs = np.stack([np.random.choice(neigh_ixs.shape[1],
                                                  size=(int(sampled_fraction * (n_neighbors + 1)),),
                                                  replace=False,
                                                  p=p) for i in range(neigh_ixs.shape[0])], 0)
        # adata.uns['sampling_ixs'] = sampling_ixs
        neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]
        nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
        embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                           neigh_ixs.ravel(),
                                           np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                          shape=(neigh_ixs.shape[0],
                                                 neigh_ixs.shape[0]))
        adata.uns['neigh_ixs'] = neigh_ixs.copy()
        
        corrcoef = colDeltaCorpartial(X, delta_X, neigh_ixs)  # velocyto
        # corrcoef = _colDeltaCorpartial(X, delta_X, neigh_ixs)  # my
        if replace_prob == 'prob':
            corrcoef = replace_with_CauTrigger(adata, corrcoef, state_pair=state_pair,state_obs=state_obs, method='prob')
        if replace_prob == 'logits':
            corrcoef = replace_with_CauTrigger(adata, corrcoef,state_pair=state_pair,state_obs=state_obs, method='logits')
    
        if np.any(np.isnan(corrcoef)):
            corrcoef[np.isnan(corrcoef)] = 1
            # logging.debug(
            #     "Nans encountered in corrcoef and corrected to 1s. If not identical cells were present it is probably a small isolated cluster converging after imputation.")
        transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.A  # naive
        transition_prob /= transition_prob.sum(1)[:, None]
    
        adata.obsm['embedding_knn'] = embedding_knn.copy()
        adata.obsp['transition_prob'] = transition_prob.copy()

    def calculate_embedding_shift(adata, embedding_name=embedding_name):
        transition_prob = adata.obsp['transition_prob'].copy()
        embedding = adata.obsm[embedding_name].copy()
        embedding_knn = adata.obsm['embedding_knn'].copy()
    
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]  # shape (2,ncells,ncells)
        with np.errstate(divide='ignore', invalid='ignore'):  # 这个的作用是忽略除以0以及nan警告
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)  # divide by L2
            np.fill_diagonal(unitary_vectors[0, ...], 0)  # fix nans
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        delta_embedding = (transition_prob * unitary_vectors).sum(2)
        delta_embedding -= (embedding_knn.A * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T
        delta_embedding = delta_embedding.T
        adata.obsm['delta_embedding'] = delta_embedding.copy()

    def calculate_p_mass(adata, embedding_name=embedding_name, smooth=smooth, n_grid=n_grid, n_neighbors=None,draw_single=None):
        steps = (n_grid, n_grid)
        embedding = adata.obsm[embedding_name].copy()
        # delta_embedding = adata.obsm['delta_embedding'].copy()
        if draw_single:
            adata_tmp = adata.copy()
            adata_tmp.obsm['delta_embedding'][adata_tmp.obs[state_obs] != draw_single] = 0
            delta_embedding = adata_tmp.obsm['delta_embedding'].copy()
        else:
            delta_embedding = adata.obsm['delta_embedding'].copy()
        # Prepare the grid
        grs = []
        for dim_i in range(embedding.shape[1]):
            m, M = np.min(embedding[:, dim_i]), np.max(embedding[:, dim_i])
    
            # if xylim[dim_i][0] is not None:
            #     m = xylim[dim_i][0]
            # if xylim[dim_i][1] is not None:
            #     M = xylim[dim_i][1]
    
            m = m - 0.025 * np.abs(M - m)
            M = M + 0.025 * np.abs(M - m)
            gr = np.linspace(m, M, steps[dim_i])
            grs.append(gr)
    
        meshes_tuple = np.meshgrid(*grs)
        gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

        if n_neighbors is None:
            n_neighbors = int(adata.shape[0] / 5)

        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embedding)
        dists, neighs = nn.kneighbors(gridpoints_coordinates)
    
        std = np.mean([(g[1] - g[0]) for g in grs])
        # isotropic gaussian kernel
        gaussian_w = normal.pdf(loc=0, scale=smooth * std, x=dists)
        total_p_mass = gaussian_w.sum(1)
    
        UZ = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:,None]  # weighed average
        magnitude = np.linalg.norm(UZ, axis=1)
        # Assign attributes
        flow_embedding = embedding
        flow_grid = gridpoints_coordinates
        flow = UZ
        flow_norm = UZ / np.percentile(magnitude, 99.5)
        flow_norm_magnitude = np.linalg.norm(flow_norm, axis=1)
        adata.uns['total_p_mass'] = total_p_mass.copy()
        adata.uns['flow_grid'] = flow_grid.copy()
        adata.uns['flow'] = flow.copy()

    def suggest_mass_thresholds(adata, embedding_name=embedding_name, n_suggestion=n_suggestion, s=1, n_col=4):
        embedding = adata.obsm[embedding_name].copy()
        total_p_mass = adata.uns['total_p_mass'].copy()
        flow_grid = adata.uns['flow_grid'].copy()
    
        min_ = total_p_mass.min()
        max_ = total_p_mass.max()
        suggestions = np.linspace(min_, max_ / 2, n_suggestion)
        import math
        n_rows = math.ceil(n_suggestion / n_col)
    
        fig, ax = plt.subplots(n_rows, n_col, figsize=[5 * n_col, 5 * n_rows])
        if n_rows == 1:
            ax = ax.reshape(1, -1)
    
        row = 0
        col = 0
        for i in range(n_suggestion):
    
            ax_ = ax[row, col]
    
            col += 1
            if col == n_col:
                col = 0
                row += 1
    
            idx = total_p_mass > suggestions[i]
    
            # ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
            ax_.scatter(embedding[:, 0], embedding[:, 1], c="lightgray", s=s)
            ax_.scatter(flow_grid[idx, 0],
                        flow_grid[idx, 1],
                        c="black", s=s)
            ax_.set_title(f"min_mass: {suggestions[i]: .2g}")
            ax_.axis("off")
        plt.show()

    def calculate_mass_filter(adata, embedding_name=embedding_name, min_mass=min_mass, plot=False):
        embedding = adata.obsm[embedding_name].copy()
        total_p_mass = adata.uns['total_p_mass'].copy()
        flow_grid = adata.uns['flow_grid'].copy()
    
        # adata.uns['min_mass'] = min_mass
        mass_filter = (total_p_mass < min_mass)
        adata.uns['mass_filter'] = mass_filter.copy()
    
        if plot:
            fig, ax = plt.subplots(figsize=[5, 5])
    
            # ax_.scatter(gridpoints_coordinates[mass_filter, 0], gridpoints_coordinates[mass_filter, 1], s=0)
            ax.scatter(embedding[:, 0], embedding[:, 1], c="lightgray", s=10)
            ax.scatter(flow_grid[~mass_filter, 0],
                       flow_grid[~mass_filter, 1],
                       c="black", s=0.5)
            ax.set_title("Grid points selected")
            ax.axis("off")
            plt.show()

    def plot_flow(adata, state_obs=state_obs, embedding_name=embedding_name, dot_size=dot_size, scale=scale, KO_Gene=KO_Gene, save_dir=save_dir, show=show):
        fig, ax = plt.subplots()
        sc.pl.embedding(adata, basis=embedding_name, color=state_obs, ax=ax, show=False,size=dot_size)
        # fig = plt.gcf()
        ax.set_title("")
        ax.get_legend().set_visible(False)
        # mass filter selection
        mass_filter = adata.uns['mass_filter'].copy()
        # Gridpoint cordinate selection
        gridpoints_coordinates = adata.uns['flow_grid'].copy()
        flow = adata.uns['flow'].copy()
        ax.quiver(gridpoints_coordinates[~mass_filter, 0],
                           gridpoints_coordinates[~mass_filter, 1],
                           flow[~mass_filter, 0],
                           flow[~mass_filter, 1],
                           scale=scale)
        ax.axis("off")
        ax.set_title(f"{' and '.join(KO_Gene)} Knock Out")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{' and '.join(KO_Gene)} Knock Out.png", bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    estimate_transition_prob(adata,state_pair=state_pair,state_obs=state_obs,embedding_name=embedding_name, replace_prob=method, n_neighbors=None, sampled_fraction=1)
    calculate_embedding_shift(adata,embedding_name=embedding_name)
    calculate_p_mass(adata,embedding_name=embedding_name,draw_single=draw_single)
    if run_suggest_mass_thresholds:
        suggest_mass_thresholds(adata, embedding_name=embedding_name,n_suggestion=n_suggestion)  # 确定了min_mass，这步就非必要
        return adata
    calculate_mass_filter(adata,embedding_name=embedding_name, min_mass=min_mass, plot=False)
    plot_flow(adata,state_obs=state_obs, embedding_name=embedding_name,dot_size=dot_size, scale=scale,show=show)
    return adata