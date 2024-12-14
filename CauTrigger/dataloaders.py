from math import ceil, floor
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import anndata
from anndata import AnnData
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def generate_synthetic(
        n_samples=300,  # 200
        oversampling_factor=5,
        n_up_features=100,  # 50
        n_down_features=300,  # 150
        n_up_causal=50,  # 10
        n_down_causal=100,  # 50
        n_hidden=0,
        n_latent=5,
        n_targets=1,
        bias=0.0,
        effective_rank=None,  # low rank parameter
        tail_strength=0.5,  # low rank parameter
        causal_strength=5.0,
        noise=0.001,
        shuffle_feature=False,
        shuffle_sample=True,
        is_linear=True,
        activation="relu",
        seed=42,
):
    """
    Generate synthetic data with two pathways, and pack them into an AnnData object.

    Parameters
    ----------
    n_samples : int, default=200
        The number of samples.
    oversampling_factor : int, default=5
        The oversampling factor.
    n_up_features : int, default=50
        The number of upstream features.
    n_down_features : int, default=150
        The number of downstream features.
    n_up_causal : int, default=10
        The number of causal upstream features.
    n_down_causal : int, default=50
        The number of causal downstream features.
    n_hidden : int, default=0
        The number of hidden units in the non-linear model.
    n_latent : int, default=10
        The number of latent units in the non-linear model.
    n_targets : int, default=1
        The number of target variables.
    bias : float, default=0.0
        The bias in the linear model.
    effective_rank : int, default=None
        The effective rank of the input matrix.
    tail_strength : float, default=0.5
        The strength of the tail of the input matrix.
    causal_strength : float, default=5.0
        The strength of the causal weights.
    noise : float, default=0.0
        The noise added to the input data.
    shuffle_feature : bool, default=False
        Whether to shuffle the features.
    shuffle_sample : bool, default=True
        Whether to shuffle the samples.
    is_linear : bool, default=True
        Whether the transformation from features to targets is linear.
    activation : str, default='relu'
        The activation function in the non-linear model.
    seed : int, default=42
        The seed for the random number generator.

    Returns
    -------
    adata : anndata.AnnData
        The generated data in an AnnData object.
    """
    np.random.seed(seed)
    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    n_up_spurious = n_up_features - n_up_causal
    n_down_spurious = n_down_features - n_down_causal
    n_features = n_up_features + n_down_features

    if effective_rank is None:
        # Randomly generate a well conditioned input set
        X_up = np.random.standard_normal(size=(total_samples, n_up_features))
    else:
        # Randomly generate a low rank, fat tail input set
        X_up = make_low_rank_matrix(
            n_samples=total_samples,
            n_features=n_up_features,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
        )

    if is_linear:
        # Generate causal weights from upstream features to downstream causal features
        cau_weights, spu_weights = generate_causal_weights(
            n_up_features,
            n_up_causal,
            n_down_causal,
            n_down_spurious,
            causal_strength=causal_strength,
        )

        # Generate weights from downstream causal features to targets
        d2t_causal_weights = np.random.normal(size=(n_down_causal, n_targets))

        # Calculate the downstream causal features and targets
        down_cau_feats = np.dot(X_up, cau_weights)
        down_spu_feats = np.dot(X_up, spu_weights)
        y = np.dot(down_cau_feats, d2t_causal_weights) + bias
        X_down = np.concatenate([down_cau_feats, down_spu_feats], axis=1)
        X_down = X_down / X_down.std()

    else:
        # Determine the weights and layers to use based on n_hidden
        if n_hidden > 0:
            # n_features -> n_hidden -> n_latent
            first_weights, first_weights_spu = generate_causal_weights(
                n_up_features,
                n_up_causal,
                n_hidden,
                n_hidden,
                causal_strength=causal_strength,
            )
            second_weights = np.random.normal(size=(n_hidden, n_latent))
            second_weights_spu = np.random.normal(size=(n_hidden, n_latent))
        else:
            # n_features -> n_latent
            first_weights, first_weights_spu = generate_causal_weights(
                n_up_features,
                n_up_causal,
                n_latent,
                n_latent,
                causal_strength=causal_strength,
            )
            second_weights = None
            second_weights_spu = None

        # Apply the weights to the input for causal features
        first_layer = np.dot(X_up, first_weights)
        first_layer = apply_activation(first_layer, activation)

        if second_weights is not None:
            second_layer = np.dot(first_layer, second_weights)
            second_layer = apply_activation(second_layer, activation)
        else:
            second_layer = first_layer

        # n_latent -> n_down_causal
        output_weights = np.random.normal(size=(n_latent, n_down_causal))
        down_cau_feats = np.dot(second_layer, output_weights)

        # n_down_causal -> n_targets
        d2t_causal_weights = np.random.normal(size=(n_down_causal, n_targets))
        y = np.dot(down_cau_feats, d2t_causal_weights) + bias

        # Apply the weights to the input for spurious features
        first_layer_spu = np.dot(X_up, first_weights_spu)
        first_layer_spu = apply_activation(first_layer_spu, activation)

        if second_weights_spu is not None:
            second_layer_spu = np.dot(first_layer_spu, second_weights_spu)
            second_layer_spu = apply_activation(second_layer_spu, activation)
        else:
            second_layer_spu = first_layer_spu

        # n_latent -> n_down_spurious
        output_weights_spu = np.random.normal(size=(n_latent, n_down_spurious))
        down_spu_feats = np.dot(second_layer_spu, output_weights_spu)

        # Concatenate causal and spurious downstream features
        X_down = np.concatenate([down_cau_feats, down_spu_feats], axis=1)
        X_down = X_down / X_down.std()

    # Generate indices for positive and negative samples
    sorted_indices = np.argsort(y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])
    # Generate state variable
    # print(y.mean())
    labels = np.where(y > y.mean(), 1, 0)

    # feature types
    up_types = [
        "causal" if i < n_up_causal else "spurious" for i in range(n_up_features)
    ]
    down_types = [
        "causal" if i < n_down_causal else "spurious" for i in range(n_down_features)
    ]

    n_X_confusing = 10
    n_y_confusing = 10
    if n_X_confusing:
        X_up = generate_X_confusing_variables(
            X_up, n_causal=np.arange(10), confusing_number=n_X_confusing
        )
        X_confusing_feature_types = np.full(n_X_confusing, "X_confusing")
        up_types = np.hstack((up_types, X_confusing_feature_types))
    if n_y_confusing:
        X_up = generate_y_confusing_variables(X_up, y, confusing_number=n_y_confusing)
        y_confusing_feature_types = np.full(n_y_confusing, "y_confusing")
        up_types = np.hstack((up_types, y_confusing_feature_types))

    # X_up = (X_up - X_up.mean(axis=0)) / X_up.std(axis=0)

    var_df_up = pd.DataFrame(
        {"feature_types": up_types},
        index=[f"up_f{i + 1}" for i in range(n_up_features + 20)],
    )
    var_df_down = pd.DataFrame(
        {"feature_types": down_types},
        index=[f"down_f{i + 1}" for i in range(n_down_features)],
    )

    # Add noise
    if noise > 0.0:
        X_up += np.random.normal(scale=noise * X_up.std(), size=X_up.shape)
        X_down += np.random.normal(scale=noise * X_down.std(), size=X_down.shape)

    # shuffle
    if shuffle_sample:
        row_indices = np.random.permutation(indices)
    else:
        row_indices = indices
    if shuffle_feature:
        col_indices_up = np.random.permutation(n_up_features)
        col_indices_down = np.random.permutation(n_down_features)
    else:
        col_indices_up = np.arange(n_up_features)
        col_indices_down = np.arange(n_down_features)

    # generate AnnData
    adata = AnnData(X_up[row_indices, :][:, col_indices_up], dtype=X_up.dtype)
    adata.obs["labels"] = labels[row_indices]
    adata.var = var_df_up.iloc[col_indices_up, :]

    # add X_down into adata.obsm, and its feature info into adata.uns
    adata.obsm["X_down"] = X_down[row_indices, :][:, col_indices_down]
    adata.uns["X_down_feature"] = var_df_down.iloc[col_indices_down, :]

    return adata


def generate_causal_weights(
        n_up_feats, n_up_cau, n_down_cau, n_down_spu, causal_strength=6.0
):
    # weights for downstream causal features
    n_up_spu = n_up_feats - n_up_cau
    weights_causal = np.zeros((n_up_feats, n_down_cau))
    values = np.random.normal(loc=causal_strength, scale=1, size=(n_up_cau, n_down_cau))
    signs = np.random.normal(loc=0.0, scale=1.0, size=(n_up_cau, n_down_cau))
    cau_values = np.where(signs < 0, -values, values)
    weights_causal[:n_up_cau, :] = cau_values
    weights_causal[n_up_cau:, :] = np.clip(
        np.random.normal(loc=0, scale=1, size=(n_up_spu, n_down_cau)), -1, 1
    )

    # weights for downstream spurious features
    weights_spurious = np.zeros((n_up_feats, n_down_spu))
    values = np.random.normal(loc=causal_strength, scale=1, size=(n_up_spu, n_down_spu))
    signs = np.random.normal(loc=0.0, scale=1.0, size=(n_up_spu, n_down_spu))
    spu_values = np.where(signs < 0, -values, values)
    weights_spurious[n_up_cau:, :] = spu_values
    weights_spurious[:n_up_cau, :] = np.clip(
        np.random.normal(loc=0, scale=1, size=(n_up_cau, n_down_spu)), -1, 1
    )

    return weights_causal, weights_spurious


def apply_activation(x, activation):
    if activation == "relu":
        return np.maximum(x, 0)
    elif activation == "tanh":
        return np.tanh(x)
    return x


def make_low_rank_matrix(
        n_samples=100,
        n_features=100,
        effective_rank=10,
        tail_strength=0.5,
):
    """Generate a mostly low rank matrix with bell-shaped singular values.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The matrix.
    """
    n = min(n_samples, n_features)

    # Random (ortho normal) vectors
    u, _ = np.linalg.qr(np.random.standard_normal(size=(n_samples, n)))
    v, _ = np.linalg.qr(np.random.standard_normal(size=(n_features, n)))
    # u, _ = linalg.qr(
    #     np.random.standard_normal(size=(n_samples, n)),
    #     mode="economic",
    #     check_finite=False,
    # )
    #
    # v, _ = linalg.qr(
    #     np.random.standard_normal(size=(n_features, n)),
    #     mode="economic",
    #     check_finite=False,
    # )

    # Index of the singular values
    singular_ind = np.arange(n, dtype=np.float64)

    # Build the singular profile by assembling signal and noise components
    low_rank = (1 - tail_strength) * np.exp(-1.0 * (singular_ind / effective_rank) ** 2)
    tail = tail_strength * np.exp(-0.1 * singular_ind / effective_rank)
    s = np.identity(n) * (low_rank + tail)

    return np.dot(np.dot(u, s), v.T) * 100


def generate_sparse_matrix(n_rows, n_cols, k=2, replace=False):
    # Initialize sparse matrix
    M = np.zeros((n_rows, n_cols))

    # Generate random indices
    indices = np.random.choice(n_cols, size=k * n_rows, replace=replace)
    # indices = np.sort(indices)

    # Assign random values to matrix
    for i in range(n_rows):
        row_indices = indices[i * k: (i + 1) * k]
        row_values = np.random.randn(k)
        M[i, row_indices] = row_values

    return M


def generate_block_matrix(p, k=10, sparsity=0.1):
    """
    Generates a dim x p matrix consisting of k blocks.
    Each block is a randomly generated p/k x p/k matrix.
    """
    matrix = np.zeros((p, p))
    block_size = p // k

    for i in range(k):
        block = np.random.randn(block_size, block_size)
        matrix[
        i * block_size: (i + 1) * block_size, i * block_size: (i + 1) * block_size
        ] = block

    if sparsity > 0:
        matrix[np.random.rand(*matrix.shape) < sparsity] = 0
    return matrix


def generate_synthetic_v1(
        n_samples,
        n_feats=1000,
        n_up_feats=10,
        n_down_feats=990,
        n_latent=5,
        noise_scale=1.0,
        seed=42,
):
    """Generate synthetic data for testing. X_a to X_b to Y; X_o"""
    np.random.seed(seed)
    total_samples = n_samples * 5
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    n_other_feats = n_feats - n_up_feats - n_down_feats

    # Generate random weights for upstream features
    # W1 = np.random.randn(n_latent, n_up_feats)
    W1 = generate_sparse_matrix(n_latent, n_up_feats)
    # W1[0, :] = 0
    b1 = np.random.randn(n_latent) * 1.0

    # Generate random weights for downstream features
    # W2 = np.random.randn(n_down_feat, n_latent)
    W2 = generate_sparse_matrix(n_down_feats, n_latent, k=1, replace=True)
    b2 = np.random.randn(n_down_feats) * 1.0

    # Generate random weights for other features
    W4 = generate_sparse_matrix(n_other_feats, n_latent, k=1, replace=True)
    b4 = np.random.randn(n_other_feats) * 1.0

    # Generate random weights for state variable
    sparsity = 0.1
    V1 = generate_block_matrix(n_down_feats, k=10, sparsity=0.5)
    c1 = np.random.randn(n_down_feats) * 1.0
    V2 = np.random.randn(1, n_down_feats)
    V2[np.random.rand(*V2.shape) < sparsity] = 0
    c2 = np.random.randn(1) * 1.0

    # Generate input features
    X_up = np.random.randn(total_samples, n_up_feats) * 6.0
    # X_up = X_up - X_up.min()
    # X_up, labels = make_blobs(n_samples=total_samples, n_features=n_up_feats, centers=2, cluster_std=1.0)

    X_latent1 = np.dot(X_up, W1.T) + b1
    X_latent1 = np.tanh(X_latent1)
    X_latent1 += np.random.randn(*X_latent1.shape) * 0.5 * X_latent1.std()

    # Generate downstream features
    X_down = 1 * np.dot(X_latent1, W2.T) + b2
    X_down += np.random.randn(*X_down.shape) * 0.5 * X_down.std()
    # X_down = X_down - X_down.min()
    # V1 = np.identity(n_down_feat)

    # Generate other features
    X_latent2 = X_latent1
    X_other = 1 * np.dot(X_latent2, W4.T) + b4
    X_other += np.random.randn(*X_other.shape) * 0.5 * X_other.std()

    # Generate state variable
    Y = np.dot(X_down, V1.T) + c1
    Y = np.dot(Y, V2.T) + c2

    # Generate indices for positive and negative samples
    sorted_indices = np.argsort(Y, axis=0).flatten()
    pos_indices = sorted_indices[-pos_samples:]
    neg_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([pos_indices, neg_indices])
    # Generate state variable
    print(Y.mean())
    labels = np.where(Y > Y.mean(), 1, 0)

    # Concatenate input features and labels
    X = np.concatenate([X_up, X_down, X_other], axis=1)
    if noise_scale > 0.0:
        n_samples, n_features = X.shape
        variances = np.var(X, axis=0)
        X = X + np.random.normal(
            loc=0, scale=np.sqrt(noise_scale * variances), size=(n_samples, n_features)
        )
    # X = StandardScaler().fit_transform(X)  # Standardize input features
    feat_types = np.repeat(
        ["upstream", "downstream", "others"], [n_up_feats, n_down_feats, n_other_feats]
    )

    adata = AnnData(X[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feature_types"] = feat_types

    return adata


def generate_synthetic_v2(
        n_samples, n_up_feats=10, n_down_feat=990, n_latent=5, seed=42
):
    """Generate synthetic data for testing. X_up has 2 clusters"""
    np.random.seed(seed)

    # Generate random weights for upstream features
    # W1 = np.random.randn(n_latent, n_up_feats)
    W1 = generate_sparse_matrix(n_latent, n_up_feats, sparsity=0.4)
    # W1[0, :] = 0
    b1 = np.random.randn(n_latent) * 1.0

    # Generate random weights for downstream features
    # W2 = np.random.randn(n_down_feat, n_latent)
    W2 = generate_sparse_matrix(n_down_feat, n_latent, sparsity=0.2)
    b2 = np.random.randn(n_down_feat) * 1.0

    # Generate random weights for state variable
    sparsity = 0.2
    V1 = generate_block_matrix(n_down_feat, k=10, sparse=0.1)

    c1 = np.random.randn(n_down_feat) * 1.0
    V2 = np.random.randn(1, n_down_feat)
    V2[np.random.rand(*V2.shape) < sparsity] = 0
    c2 = np.random.randn(1) * 1.0

    # Generate input features
    X_up, labels = make_blobs(
        n_samples=n_samples, n_features=n_up_feats, centers=2, cluster_std=1.0
    )

    X_latent = np.dot(X_up, W1.T) + b1
    X_latent += np.random.randn(*X_latent.shape) * 0.5
    X_latent = np.maximum(X_latent, 0)

    X_down = np.dot(X_latent, W2.T) + b2
    X_down += np.random.randn(*X_down.shape) * 0.5
    # V1 = np.identity(n_down_feat)
    Y = np.dot(X_down, V1.T) + c1
    Y = np.dot(Y, V2.T) + c2

    # Concatenate input features and labels
    X = np.concatenate([X_up, X_down], axis=1)
    # X = StandardScaler().fit_transform(X)  # Standardize input features
    feat_types = np.repeat(["upstream", "downstream"], [n_up_feats, n_down_feat])

    adata = AnnData(X)
    adata.obs["labels"] = labels
    adata.var["feature_types"] = feat_types

    return adata


def generate_synthetic_v3(
        n_samples,
        n_feats=1000,
        n_up_feats=10,
        n_down_feats=990,
        n_latent=5,
        noise_scale=1.0,
        seed=42,
):
    """Generate synthetic data for testing. X_a to X_b to Y; X_o"""
    np.random.seed(seed)
    total_samples = n_samples * 5
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples

    n_other_feats = n_feats - n_up_feats - n_down_feats

    # Generate random weights for upstream features
    # W1 = np.random.randn(n_latent, n_up_feats)
    W1 = generate_sparse_matrix(n_latent, n_up_feats)
    # W1[0, :] = 0
    b1 = np.random.randn(n_latent) * 1.0

    # Generate random weights for downstream features
    # W2 = np.random.randn(n_down_feat, n_latent)
    W2 = generate_sparse_matrix(n_down_feats, n_latent, k=1, replace=True)
    b2 = np.random.randn(n_down_feats) * 1.0

    W3 = generate_sparse_matrix(n_latent, n_up_feats)
    b3 = np.random.randn(n_latent) * 1.0
    W4 = generate_sparse_matrix(n_other_feats, n_latent, k=1, replace=True)
    b4 = np.random.randn(n_other_feats) * 1.0

    # Generate random weights for state variable
    sparsity = 0.1
    V1 = generate_block_matrix(n_down_feats, k=10, sparsity=0.5)
    c1 = np.random.randn(n_down_feats) * 1.0
    V2 = np.random.randn(1, n_down_feats)
    V2[np.random.rand(*V2.shape) < sparsity] = 0
    c2 = np.random.randn(1) * 1.0

    # Generate input features
    X_up = np.random.randn(total_samples, n_up_feats) * 6.0
    # X_up = X_up - X_up.min()
    # X_up, labels = make_blobs(n_samples=total_samples, n_features=n_up_feats, centers=2, cluster_std=1.0)

    X_latent1 = np.dot(X_up, W1.T) + b1
    X_latent1 = np.tanh(X_latent1)
    X_latent1 += np.random.randn(*X_latent1.shape) * 0.5 * X_latent1.std()

    # Generate downstream features
    X_down = 1 * np.dot(X_latent1, W2.T) + b2
    X_down += np.random.randn(*X_down.shape) * 0.5 * X_down.std()
    # X_down = X_down - X_down.min()
    # V1 = np.identity(n_down_feat)

    # Generate other features
    X_latent2 = np.dot(X_up, W3.T) + b3
    X_latent2 = np.tanh(X_latent2)
    X_latent2 += np.random.randn(*X_latent2.shape) * 0.5 * X_latent2.std()
    X_other = 1 * np.dot(X_latent2, W4.T) + b4
    X_other += np.random.randn(*X_other.shape) * 0.5 * X_other.std()

    # Generate state variable
    Y = np.dot(X_down, V1.T) + c1
    Y = np.dot(Y, V2.T) + c2

    # Generate indices for positive and negative samples
    sorted_indices = np.argsort(Y, axis=0).flatten()
    pos_indices = sorted_indices[-pos_samples:]
    neg_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([pos_indices, neg_indices])
    # Generate state variable
    print(Y.mean())
    labels = np.where(Y > Y.mean(), 1, 0)

    # Concatenate input features and labels
    X = np.concatenate([X_up, X_down, X_other], axis=1)
    if noise_scale > 0.0:
        n_samples, n_features = X.shape
        variances = np.var(X, axis=0)
        X = X + np.random.normal(
            loc=0, scale=np.sqrt(noise_scale * variances), size=(n_samples, n_features)
        )
    # X = StandardScaler().fit_transform(X)  # Standardize input features
    feat_types = np.repeat(
        ["upstream", "downstream", "others"], [n_up_feats, n_down_feats, n_other_feats]
    )

    adata = AnnData(X[indices, :])
    adata.obs["labels"] = labels[indices]
    adata.var["feature_types"] = feat_types

    return adata


def data_splitter(
        adata: AnnData,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
):
    """Split indices in train/test/val sets."""
    n_train, n_val = validate_data_split(adata.n_obs, train_size, validation_size)
    random_state = np.random.RandomState(seed=42)
    permutation = random_state.permutation(adata.n_obs)
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val: (n_val + n_train)]
    test_idx = permutation[(n_val + n_train):]

    train_adata = adata[train_idx]
    val_adata = adata[val_idx]
    if test_idx.shape[0] == 0:
        return train_adata, val_adata
    else:
        test_adata = adata[test_idx]
        return train_adata, val_adata, test_adata


def validate_data_split(
        n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


def batch_sampler(
        adata: AnnData,
        batch_size: int,
        shuffle: bool = False,
        drop_last: Union[bool, int] = False,
):
    """
    Custom torch Sampler that returns a list of indices of size batch_size.

    Parameters
    ----------
    adata
        adata to sample from
    batch_size
        batch size of each iteration
    shuffle
        if ``True``, shuffles indices before sampling
    drop_last
        if int, drops the last batch if its length is less than drop_last.
        if drop_last == True, drops last non-full batch.
        if drop_last == False, iterate over all batches.
    """
    if drop_last > batch_size:
        raise ValueError(
            "drop_last can't be greater than batch_size. "
            + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
        )

    last_batch_len = adata.n_obs % batch_size
    if (drop_last is True) or (last_batch_len < drop_last):
        drop_last_n = last_batch_len
    elif (drop_last is False) or (last_batch_len >= drop_last):
        drop_last_n = 0
    else:
        raise ValueError("Invalid input for drop_last param. Must be bool or int.")

    if shuffle is True:
        idx = torch.randperm(adata.n_obs).tolist()
    else:
        idx = torch.arange(adata.n_obs).tolist()

    if drop_last_n != 0:
        idx = idx[:-drop_last_n]

    adata_iter = [
        adata[idx[i: i + batch_size]] for i in range(0, len(idx), batch_size)
    ]
    return adata_iter


def generate_synthetic_nb(
        batch_size: int = 500,
        n_genes: int = 2000,
        n_proteins: int = 20,
        n_batches: int = 2,
        n_labels: int = 2,
) -> AnnData:
    #  Here samples are drawn from a negative binomial distribution with specified parameters,
    # `n` failures and `p` probability of failure where `n` is > 0 and `p` is in the interval
    #  [0, 1], `n` is equal to diverse dispersion parameter.
    data = np.zeros(shape=(batch_size * n_batches, n_genes))
    mu = np.random.randint(low=1, high=20, size=n_labels)
    p = mu / (mu + 5)
    for i in range(n_batches):
        data[batch_size * i: batch_size * (i + 1), :] = np.random.negative_binomial(
            5, 1 - p[i], size=(batch_size, n_genes)
        )
    data = np.random.negative_binomial(5, 0.3, size=(batch_size * n_batches, n_genes))
    mask = np.random.binomial(n=1, p=0.7, size=(batch_size * n_batches, n_genes))
    data = data * mask  # We put the batch index first
    labels = np.random.randint(0, n_labels, size=(batch_size * n_batches,))
    # labels = np.array(["label_%d" % i for i in labels])

    batch = []
    for i in range(n_batches):
        batch += ["batch_{}".format(i)] * batch_size

    adata = AnnData(data)
    batch = np.random.randint(
        high=n_batches, low=0, size=(batch_size * n_batches, 1)
    ).astype(np.float32)
    # adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["batch"] = batch
    # adata.obs["labels"] = pd.Categorical(labels)
    adata.obs["labels"] = labels
    adata.uns["n_batch"] = n_batches

    # Protein measurements
    p_data = np.zeros(shape=(adata.shape[0], n_proteins))
    mu = np.random.randint(low=1, high=20, size=n_labels)
    p = mu / (mu + 5)
    for i in range(n_batches):
        p_data[batch_size * i: batch_size * (i + 1), :] = np.random.negative_binomial(
            5, 1 - p[i], size=(batch_size, n_proteins)
        )
    p_data = np.random.negative_binomial(5, 0.3, size=(adata.shape[0], n_proteins))
    adata.obsm["protein_expression"] = p_data
    adata.uns["protein_names"] = np.arange(n_proteins).astype(str)

    return adata


def generate_X_confusing_variables(
        X,
        n_causal,
        confusing_number=20,
):
    X_causal = X[:, n_causal]

    # matrix mapping & data shifted
    np.random.seed(42)
    transformation_matrix = np.random.randn(len(n_causal), confusing_number)
    confusing_var = X_causal.dot(transformation_matrix)
    min_value = np.min(confusing_var)
    confusing_var = confusing_var - min_value

    # add noise
    noise_level = 0.1
    np.random.seed(42)

    def laplace_noise(shape, scale):
        return np.random.laplace(scale=scale, size=shape)

    noise = laplace_noise(confusing_var.shape, scale=noise_level)

    confusing_var_shifted_log_noise = np.log(confusing_var + 1) + noise
    combined_X = np.concatenate((X, confusing_var_shifted_log_noise), axis=1)

    # # var_df
    # feature_names = ["f_" + str(i + X.shape[1] + 1) for i in range(confusing_number)]
    # feat_type = ["confusing" for i in range(confusing_number)]
    # feat_label = [0 for i in range(confusing_number)]
    # confusing_var_df = pd.DataFrame()
    # confusing_var_df["feat_type"] = feat_type
    # confusing_var_df["feat_label"] = feat_label
    # confusing_var_df.index = feature_names
    # combined_var_df = pd.concat([var_df, confusing_var_df], axis=0)

    return combined_X


def generate_y_confusing_variables(X, y, confusing_number=20):
    n_samples = len(y)
    confusing_vars = np.zeros((n_samples, confusing_number))
    for i in range(confusing_number):
        transformation_matrix = np.random.randn(len(y), len(y))
        confusing_var = transformation_matrix.dot(y)
        min_value = np.min(confusing_var)
        confusing_var = confusing_var - min_value

        def laplace_noise(shape, scale):
            return np.random.laplace(scale=scale, size=shape)

        noise = laplace_noise(confusing_var.shape, scale=0.1)

        confusing_var_shifted_log_noise = np.log(confusing_var + 1) + noise
        confusing_vars[:, i] = confusing_var_shifted_log_noise.flatten()

    combined_X = np.concatenate((X, confusing_vars), axis=1)

    return combined_X


def generate_simulation_data(
        n_samples=300,
        n_up_features=50,
        n_down_features=150,
        n_up_causal=10,
        causal_strength=0,
        noise=1,
        seed=42,
):
    np.random.seed(seed)
    # pos_samples = n_samples // 2
    # neg_samples = n_samples - pos_samples
    n_up_spurious = n_up_features - n_up_causal

    Xc = np.random.uniform(1, 100, size=(n_samples, n_up_causal))
    Wc_d = np.random.normal(
        loc=causal_strength, scale=1, size=(n_up_causal, n_down_features)
    )
    Xd = np.dot(Xc, Wc_d) + noise * np.random.standard_normal(
        size=(n_samples, n_down_features)
    )
    Xd = (Xd - np.min(Xd)) / (np.max(Xd) - np.min(Xd)) * 99 + 1

    # Wd_y = np.random.normal(loc=causal_strength, scale=1, size=(n_down_features, 1))
    # Y = np.dot(Xd, Wd_y) + noise * np.random.standard_normal(size=(n_samples, 1))
    # Y = np.where(Y <= np.median(Y), 0, 1)

    Z = np.random.uniform(1, 100, size=(n_samples, 10))
    Wz_s = np.random.normal(loc=causal_strength, scale=1, size=(10, n_up_spurious))
    Xs = np.dot(Z, Wz_s) + noise * np.random.standard_normal(
        size=(n_samples, n_up_spurious)
    )
    Xs = (Xs - np.min(Xs)) / (np.max(Xs) - np.min(Xs)) * 99 + 1

    Y = np.sum(np.hstack((Xd, Z)), axis=1).reshape(
        -1, 1
    ) + noise * np.random.standard_normal(size=(n_samples, 1))
    Y = np.where(Y <= np.median(Y), 0, 1)

    # generate AnnData
    Xu = np.concatenate((Xc, Xs), axis=1)
    adata = AnnData(Xu)
    adata.obs["labels"] = Y
    # adata.var = np.arange(1, n_up_features+1)

    # add X_down into adata.obsm, and its feature info into adata.uns
    adata.obsm["X_down"] = Xd
    # adata.uns['X_down_feature'] = np.arange(n_up_features+1, n_up_features+n_down_features+1)

    return adata


def generate_simulation_data2(
        n_samples=300,
        n_up_features=50,
        n_down_features=150,
        n_up_causal=10,
        causal_strength=0,
        sample_y_num=100,
        noise=1,
        seed=42,
):
    np.random.seed(seed)
    # pos_samples = n_samples // 2
    # neg_samples = n_samples - pos_samples
    n_up_spurious = n_up_features - n_up_causal

    Xc = np.random.exponential(scale=1.0, size=(n_samples, n_up_causal))
    rate = np.mean(Xc, axis=1)
    scale = 1 / rate
    scale = scale.reshape(-1, 1)
    Xd = np.random.exponential(scale=scale, size=(n_samples, n_down_features))

    Z = np.random.exponential(scale=1, size=(n_samples, n_up_causal))
    rate = np.sum(Z, axis=1)
    scale = 1 / rate
    scale = scale.reshape(-1, 1)
    Xs = np.random.exponential(scale=scale, size=(n_samples, n_up_spurious))

    # Y = np.random.normal(np.mean(Xd) + np.mean(Z) * np.exp(-np.mean(Xd) * np.mean(Z)), 1, size=(n_samples, 1))

    Xd_mean = np.mean(Xd, axis=1, keepdims=True)
    Z_mean = np.mean(Z, axis=1, keepdims=True)
    Y_mean = Xd_mean + Z_mean * np.exp(-Xd_mean * Z_mean)

    if sample_y_num > 0:
        Y = np.random.normal(Y_mean, 1, size=(n_samples, sample_y_num)).mean(
            axis=1, keepdims=True
        )
    else:
        Y = Y_mean

    Y = np.where(Y <= np.median(Y), 0, 1)

    # generate AnnData
    Xu = np.concatenate((Xc, Xs), axis=1)
    adata = AnnData(Xu)
    adata.obs["labels"] = Y
    # adata.var = np.arange(1, n_up_features+1)

    # add X_down into adata.obsm, and its feature info into adata.uns
    adata.obsm["X_down"] = Xd
    # adata.uns['X_down_feature'] = np.arange(n_up_features+1, n_up_features+n_down_features+1)

    return adata

def apply_activation(x, activation):
    if activation == "relu":
        return np.maximum(x, 0)
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "log":
        return np.log(x + 1)
    return x


def zscore_normalization(X, epsilon=1e-10):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_adj = std + epsilon  # 防止除以0
    z_score_X = (X - mean) / std_adj

    return z_score_X



def generate_synthetic_jersey(
        # simulating data
        n_samples=200,
        oversampling_factor=5,
        n_up_features=100,
        n_down_features=100,
        n_causal=10,
        n_hidden=5,
        n_latent=5,
        noise_scale=0.1,
        causal_strength=0.5,
        direct_ratio=0.0,
        is_linear=False,
):

    if not 0 <= causal_strength <= 10:
        raise ValueError("causal_strength must be between 0 and 10")

    total_samples = n_samples * oversampling_factor
    pos_samples = n_samples // 2
    neg_samples = n_samples - pos_samples
    # 1. Generate Z and Xc
    z = np.random.standard_normal(size=(total_samples, n_latent))

    # mean_values1 = np.random.uniform(low=0.5, high=1.5, size=n_causal)
    # std_devs1 = np.random.uniform(low=1, high=2, size=n_causal)
    # Xc = np.random.normal(loc=mean_values1, scale=std_devs1, size=(total_samples, n_causal))
    c = np.random.standard_normal(size=(total_samples, n_latent))
    weights1_c = np.random.standard_normal(size=(n_latent, n_causal))
    weights2_c = np.random.standard_normal(size=(n_causal, n_causal))
    Xc = np.dot(np.dot(c, weights1_c), weights2_c)

    # Xc = np.random.exponential(scale=mean_values1, size=(total_samples, n_causal))

    # 2. Generate Xs and Xd1 from Z
    # 2.1. Construct weight layers
    # Xs
    weights1 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights2 = np.random.standard_normal(size=(n_hidden, (n_up_features - n_causal)))
    # Xd1
    weights3 = np.random.standard_normal(size=(n_latent, n_hidden))
    weights4 = np.random.standard_normal(size=(n_hidden, n_down_features))

    # 2.2. Generate Xs and Xd1
    noise1 = np.random.laplace(scale=0.5, size=(total_samples, n_down_features))
    Xs = np.dot(np.dot(z, weights1), weights2)
    # y2 = np.dot(np.dot(z, weights3), weights4) + noise1
    if is_linear:
        Xd1 = np.dot(np.dot(z, weights3), weights4) + noise1
    else:
        Xd1 = apply_activation(np.dot(apply_activation(np.dot(z, weights3), "tanh"), weights4), "tanh") + noise1

    # 3、Xc生成Xd2，将Xd1和Xd2直接叠加
    # Xc2 = np.zeros((total_samples, n_causal))
    # for i in range(n_causal):
    #     related_feature = Xs.dot(np.random.rand(int(n_features - n_causal), 1))
    #     Xc2[:, i] = related_feature.squeeze()
    # Xc2 = zscore_normalization(Xc2)
    # Xc += Xc2

    # 3. Generate Xd2 from Xc and combine Xd1 and Xd2
    weights7 = np.random.standard_normal(size=(n_causal, n_hidden))
    weights8 = np.random.standard_normal(size=(n_hidden, n_down_features))
    noise3 = np.random.laplace(scale=0.5, size=(total_samples, n_down_features))
    
    if is_linear:
        # Xd2 = Xc.dot(np.random.rand(n_causal, n_down_features))
        Xd2 = np.dot(np.dot(Xc, weights7), weights8) + noise3
    else:
        Xd2 = apply_activation(np.dot(apply_activation(np.dot(Xc, weights7), "tanh"), weights8), "tanh") + noise3
    
    rate = causal_strength
    Xd = rate * Xd2 + (1 - rate) * Xd1
    Xd = zscore_normalization(Xd)

    # 4. Generate y from Xd (and Xc)
    weights5 = np.random.standard_normal(size=(n_down_features, n_hidden))
    weights6 = np.random.standard_normal(size=(n_hidden, 1))
    noise2 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    # y1 = np.dot(np.dot(Xc, weights5), weights6) + noise2
    if is_linear:
        y1 = np.dot(np.dot(Xd, weights5), weights6) + noise2
    else:
        y1 = apply_activation(np.dot(apply_activation(np.dot(Xd, weights5), "tanh"), weights6), "tanh") + noise2

    # direct effect from Xc to y
    weights9 = np.random.standard_normal(size=(n_causal, n_hidden))
    weights10 = np.random.standard_normal(size=(n_hidden, 1))
    noise4 = np.random.laplace(scale=0.5, size=(total_samples, 1))
    if is_linear:
        y2 = np.dot(np.dot(Xc, weights9), weights10) + noise4
    else:
        y2 = apply_activation(np.dot(apply_activation(np.dot(Xc, weights9), "tanh"), weights10), "tanh") + noise4

    y = (1 - direct_ratio) * y1 + direct_ratio * y2
    # y = y1

    # 5. Combine and normalize data
    data = np.hstack((Xc, Xs, Xd))
    noise = np.random.laplace(scale=noise_scale, size=(total_samples, n_up_features+n_down_features))
    data = zscore_normalization(data) + noise

    # 6. Construct labels and return AnnData object
    sorted_indices = np.argsort(y, axis=0).flatten()
    top_indices = sorted_indices[-pos_samples:]
    down_indices = sorted_indices[:neg_samples]
    indices = np.concatenate([top_indices, down_indices])

    labels = np.where(y > np.median(y), 1, 0)
    # threshold = np.percentile(y, 25)
    # labels = np.where(y <= threshold, 1, 0)
    # indices = np.random.permutation(total_samples)

    feature_names = ["f_" + str(i + 1) for i in range(n_up_features)]
    feature_types = np.repeat(['causal', 'spurious'], [n_causal, n_up_features-n_causal])

    adata = AnnData(data[indices, 0:n_up_features], dtype=data.dtype)
    adata.obs["labels"] = labels[indices]
    adata.var["feat_type"] = feature_types
    adata.var["feat_label"] = (feature_types == 'causal').astype(int)
    adata.var.index = feature_names
    adata.obsm["X_down"] = data[indices, n_up_features:]

    return adata