import collections
from typing import Callable, Iterable, List, Optional, Literal

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList


class FeatureSplit(nn.Module):
    def __init__(self, n_features, init_weight=None, init_thresh=0.2, thresh_grad=True, attention=False, att_mean=False):
        super(FeatureSplit, self).__init__()
        self.n_features = n_features
        self.weight = torch.nn.Parameter(torch.zeros(n_features))
        self.thresh = nn.Parameter(torch.tensor(init_thresh), requires_grad=thresh_grad)
        self.attention = attention
        self.att_net = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.LeakyReLU(),
            nn.Linear(n_features, n_features),
            # nn.ReLU(),
            # nn.Linear(n_features, n_features)
        )
        self.att_mean = att_mean
        if init_weight is not None:
            assert len(init_weight) == n_features, "The length of initial_weight should be equal to n_features"
            self.weight.data = torch.tensor(init_weight, dtype=torch.float32)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        # nn.init.uniform_(self.weight, 0.999999, 0.9999999)
        # nn.init.constant_(self.weight, 0.5)
        nn.init.constant_(self.weight, 0.0)
        # sorted_weight, sorted_idx = torch.sort(self.weight.data, descending=True)
        # print(sorted_weight)
        # print(sorted_idx)

    def forward(self, x, mode='causal'):
        if self.attention:
            attention_scores = self.att_net(x)
            w = torch.sigmoid(attention_scores)
            w = torch.where(w.gt(self.thresh), w, torch.zeros_like(w))
            w_used = torch.mean(w, dim=0) if self.att_mean else w
        else:
            # use model weight
            w = torch.sigmoid(self.weight)
            w = torch.where(w.gt(self.thresh), w, torch.zeros_like(w))
            w_used = w

        x_mask = None
        if mode not in ['causal', 'spurious']:
            raise ValueError("Mode must be one of 'causal' or 'spurious'")
        elif mode == 'causal':
            x_mask = torch.mul(x, w_used)
        elif mode == 'spurious':
            x_mask = torch.mul(x, 1 - w_used)

        return x_mask, w


class FeatureWeight(nn.Module):
    def __init__(self, n_features, initial_weight=None, update_weight=True):
        super(FeatureWeight, self).__init__()
        self.n_features = n_features
        self.weight = torch.nn.Parameter(torch.ones(n_features), requires_grad=update_weight)
        self.net = nn.Linear(n_features, n_features)

        if initial_weight is not None:
            assert len(initial_weight) == n_features, "The length of initial_weight should be equal to n_features"
            self.weight.data = torch.tensor(initial_weight)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0.999999, 0.9999999)
        # nn.init.constant_(self.weight, 0.5)
        # sorted_weight, sorted_idx = torch.sort(self.weight.data, descending=True)
        # print(sorted_weight)
        # print(sorted_idx)

    def forward(self, x, mode='causal'):
        w = torch.relu(self.weight)
        x_mask = None
        if mode not in ['causal', 'spurious']:
            raise ValueError("Mode must be one of 'causal' or 'spurious'")
        elif mode == 'causal':
            x_mask = torch.mul(x, w)
        elif mode == 'spurious':
            x_mask = torch.mul(x, 1 - w)

        return x_mask, w


# Feature Selector with Top-k
class FeatureSelector(nn.Module):
    def __init__(self, n_features, n_controls=10, initial_weight=None):
        super(FeatureSelector, self).__init__()
        self.n_features = n_features
        self.n_controls = n_controls

        self.weight = torch.nn.Parameter(torch.ones(n_features))
        # self.weight.data[11:20] = 1.5

        assert 0 <= self.n_controls <= self.n_features, f"n_controls should be between 0 and {self.n_features}"
        assert isinstance(self.n_controls, int), "n_controls should be an integer"

        if initial_weight is not None:
            assert len(initial_weight) == n_features, "The length of initial_weight should be equal to n_features"
            self.weight.data = torch.tensor(initial_weight)
        else:
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, 0.999999, 0.9999999)
        # sorted_weight, sorted_idx = torch.sort(self.weight.data, descending=True)
        # print(sorted_weight)
        # print(sorted_idx)

    def forward(self, x, keep_top=True, keep_not_top=False):
        # Normalized feature weight vectors
        # w = self.weight
        w = torch.relu(self.weight)
        # print(w.min(), w.max())
        # w = torch.pow(self.weight, 2)
        # w = nn.functional.normalize(self.weight, p=1, dim=0)
        # Find the indices of the top k largest feature weights
        sorted_weight, sorted_idx = torch.sort(w, descending=True)
        # Construct a tensor where the selected features have a value of 1 and the rest have a value of 0
        mask = torch.zeros(self.n_features)
        device = x.device  # or w.device
        mask = mask.to(device)

        w_mask = None
        if not keep_top and not keep_not_top:
            raise ValueError("At least one of 'keep_top' or 'keep_not_top' must be True")
        elif keep_top and keep_not_top:
            w_mask = w
        elif keep_top and not keep_not_top:
            # Select the top k largest feature weights
            selected_idx = sorted_idx[:self.n_controls]
            mask[selected_idx] = 1
            w_mask = w * mask
        elif not keep_top and keep_not_top:
            # Select all feature weights except for the top k largest ones
            selected_idx = sorted_idx[self.n_controls:]
            mask[selected_idx] = 1
            w_mask = mask  # w_mask = (1 - w) * mask
            # w_mask = w * mask

        output = torch.mul(x, w_mask)

        return output, w


class MLP(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None, activation='relu', batch_norm=False, layer_norm=False,
                 dropout_rate=0.0, init_type='kaiming'):
        super(MLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = []
        dims = [input_dim] + hidden_dims + [output_dim]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 1:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))

                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'none':
                    pass  # no activation
                else:
                    raise ValueError("Invalid activation option")

                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))

        self.layers = nn.Sequential(*layers)

        self.init_weights(init_type)

    def init_weights(self, init_type='kaiming'):
        if init_type is None:
            return
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.layers(x)
        return x


# Encoder
class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            var_eps: float = 1e-4,
            **kwargs,
    ):
        super(Encoder, self).__init__()

        self.var_eps = var_eps
        self.encoder = MLP(
            input_dim=n_input,
            output_dim=n_hidden,
            hidden_dims=[n_hidden] * n_layers,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.log_var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # Parameters for latent distribution
        qz = self.encoder(x)
        qz_m = self.mean_encoder(qz)
        qz_v = torch.exp(self.log_var_encoder(qz)) + self.var_eps
        z = Normal(qz_m, torch.clamp(qz_v, min=1e-8, max=8).sqrt()).rsample()  # torch.clamp(qz_v, max=10)
        # z = Normal(qz_m, torch.clamp(qz_v, max=5).sqrt()).rsample()
        return dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
        )


# Decoder
class Decoder(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.0,
            linear: bool = False,  # new parameter
            **kwargs,
    ):
        super(Decoder, self).__init__()
        self.linear = linear
        if self.linear:
            self.decoder1 = nn.Linear(n_input, n_output)
        else:
            self.decoder1 = MLP(
                input_dim=n_input,
                output_dim=n_hidden,
                hidden_dims=[n_hidden] * n_layers,
                dropout_rate=dropout_rate,
                **kwargs,
            )
            self.decoder2 = nn.Linear(n_hidden, n_output)

    def forward(self, z):
        if self.linear:
            x_rec = self.decoder1(z)
        else:
            x_rec = self.decoder1(z)
            x_rec = self.decoder2(x_rec)
        return x_rec


class DynamicPhenotypeDescriptor(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_output: int = 1,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.0,
            linear: bool = False,
            **kwargs,
    ):
        super(DynamicPhenotypeDescriptor, self).__init__()
        self.linear = linear
        if self.linear:
            self.dpd1 = nn.Linear(n_input, n_output)
        else:
            self.dpd1 = MLP(
                input_dim=n_input,
                output_dim=n_hidden,
                hidden_dims=[n_hidden] * n_layers,
                dropout_rate=dropout_rate,
                **kwargs,
            )
            self.dpd2 = nn.Linear(n_hidden, n_output)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        if self.linear:
            logit = self.dpd1(x)
        else:
            x = self.dpd1(x)
            logit = self.dpd2(x)
        prob = self.activation(logit)
        return dict(
            logit=logit,
            prob=prob,
        )

