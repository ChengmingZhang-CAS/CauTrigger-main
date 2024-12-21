import numpy as np
import pandas as pd
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_features(df, threshold=None, topk=None, elbow=False):
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    selected_features = []
    for column in df.columns:
        sorted_column = df[column].sort_values(ascending=False)

        if elbow:
            weights = sorted_column.values
            cumulative_weights = weights.cumsum()
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

