import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, Literal

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from anndata import AnnData
from tqdm import tqdm
import shap
from torch.distributions import Normal, Poisson
from scipy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

from CauTrigger.dataloaders import data_splitter, batch_sampler
from CauTrigger.module import CauVAE, FVAE, DualVAE
from CauTrigger.causaleffect import joint_uncond, beta_info_flow, joint_uncond_single_dim
from CauTrigger.causaleffect import joint_uncond_v1, beta_info_flow_v1, joint_uncond_single_dim_v1
from CauTrigger.utils import set_seed


class CauTrigger(nn.Module):
    """
    Casual control of phenotype and state transitions
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_causal: int = 2,  # Number of casual factors
            **model_kwargs,
    ):
        super(CauTrigger, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = DualVAE(
            n_input_up=adata.X.shape[1],
            n_input_down=adata.obsm['X_down'].shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            **model_kwargs,
        )
    
    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 5,
            n_alpha: int = 25,
            n_beta: int = 100,
            recons_weight: float = 1.0,
            kl_weight: float = 0.02,
            up_weight: float = 1.0,
            down_weight: float = 1.0,
            feat_l1_weight: float = 0.05,
            dpd_weight: float = 3.0,
            fide_kl_weight: float = 0.05,
            causal_weight: float = 1.0,
            down_fold: float = 1.0,
            causal_fold: float = 1.0,
            spurious_fold: float = 1.0,
            stage_training: bool = True,
            weight_scheme: str = None,
            im_factor: Optional[float] = None,
            **kwargs,
    ):
        """
        Trains the model using fractal variational autoencoder.
        """
        # set_seed(42)
        # torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            'N_alpha': n_alpha,
            'N_beta': n_beta,
            'K': self.n_causal,
            'L': self.n_latent - self.n_causal,
            'z_dim': self.n_latent,
            'M': 2}
        self.ce_params = ce_params
        loss_weights = {
            'up_rec_loss': up_weight * recons_weight,
            'down_rec_loss': down_weight * recons_weight,
            'up_kl_loss': kl_weight,
            'feat_l1_loss_up': feat_l1_weight,
            'dpd_loss': dpd_weight,
            'fide_kl_loss': fide_kl_weight,
            'causal_loss': causal_weight,
        }

        self.batch_size = batch_size
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down_rec_loss': [], 'up_kl_loss': [],
                        'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [], 'fide_kl_loss': [],
                        'causal_loss': []}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_losses = {'total_loss': [], 'up_rec_loss1': [], 'up_rec_loss2': [], 'down_rec_loss': [],
                            'up_kl_loss': [], 'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [],
                            'fide_kl_loss': [], 'causal_loss': []}
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(epoch, max_epochs, scheme=weight_scheme)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                inputs_down = torch.tensor(train_batch.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(train_batch.obs['labels'], dtype=torch.float32, device=device)
                model_outputs = self.module(inputs_up)
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down, labels, imb_factor=im_factor)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    if loss_weights["causal_loss"] == 0:
                        causal_loss_list = [torch.tensor(0.0, device=device)]
                        break
                    _causal_loss1, _ = joint_uncond_v1(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                       beta_vi=True, device=device)
                    _causal_loss2, _ = beta_info_flow_v1(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                         beta_vi=False, device=device)
                    _causal_loss = _causal_loss1 * causal_fold - _causal_loss2 * spurious_fold
                    # _causal_loss = _causal_loss1 - _causal_loss2 * 3.0
                    causal_loss_list += [_causal_loss]
                up_rec_loss1 = loss_dict['up_rec_loss1'].mean()
                up_rec_loss2 = loss_dict['up_rec_loss2'].mean()
                down_rec_loss = loss_dict['down_rec_loss'].mean()
                up_kl_loss = loss_dict['up_kl_loss'].mean()
                feat_l1_loss_up = loss_dict['feat_l1_loss_up'].mean()
                feat_l1_loss_down = loss_dict['feat_l1_loss_down'].mean()
                dpd_loss = loss_dict['dpd_loss'].mean()
                fide_kl_loss = loss_dict['fide_kl_loss'].mean()
                causal_loss = torch.stack(causal_loss_list).mean()
                if self.module.feature_mapper_up.attention:
                    loss_weights["feat_l1_loss_up"] = 0.001
                total_loss = loss_weights['up_rec_loss'] * up_rec_loss1 + \
                             loss_weights['up_rec_loss'] * up_rec_loss2 + \
                             loss_weights['down_rec_loss'] * down_rec_loss + \
                             loss_weights['up_kl_loss'] * up_kl_loss + \
                             loss_weights['feat_l1_loss_up'] * feat_l1_loss_up + \
                             loss_weights['feat_l1_loss_down'] * feat_l1_loss_down * down_fold + \
                             loss_weights['dpd_loss'] * dpd_loss + \
                             loss_weights['fide_kl_loss'] * fide_kl_loss + \
                             loss_weights['causal_loss'] * causal_loss

                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     total_loss.backward()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()

                # update batch losses
                batch_losses['total_loss'].append(total_loss.item())
                batch_losses['up_rec_loss1'].append(up_rec_loss1.item())
                batch_losses['up_rec_loss2'].append(up_rec_loss2.item())
                batch_losses['down_rec_loss'].append(down_rec_loss.item())
                batch_losses['up_kl_loss'].append(up_kl_loss.item())
                batch_losses['feat_l1_loss_up'].append(feat_l1_loss_up.item())
                batch_losses['feat_l1_loss_down'].append(feat_l1_loss_down.item())
                batch_losses['dpd_loss'].append(dpd_loss.item())
                batch_losses['fide_kl_loss'].append(fide_kl_loss.item())
                batch_losses['causal_loss'].append(causal_loss.item())

            # update epochs losses
            epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
            epoch_losses['up_rec_loss1'].append(np.mean(batch_losses['up_rec_loss1']))
            epoch_losses['up_rec_loss2'].append(np.mean(batch_losses['up_rec_loss2']))
            epoch_losses['down_rec_loss'].append(np.mean(batch_losses['down_rec_loss']))
            epoch_losses['up_kl_loss'].append(np.mean(batch_losses['up_kl_loss']))
            epoch_losses['feat_l1_loss_up'].append(np.mean(batch_losses['feat_l1_loss_up']))
            epoch_losses['feat_l1_loss_down'].append(np.mean(batch_losses['feat_l1_loss_down']))
            epoch_losses['dpd_loss'].append(np.mean(batch_losses['dpd_loss']))
            epoch_losses['fide_kl_loss'].append(np.mean(batch_losses['fide_kl_loss']))
            epoch_losses['causal_loss'].append(np.mean(batch_losses['causal_loss']))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses['total_loss'])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def pretrain_attention(
            self,
            prior_probs: Optional[np.ndarray] = None,
            max_epochs: Optional[int] = 50,
            pretrain_lr: float = 1e-3,
            batch_size: int = 128,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None
    ):
        """
        Pretrain attention network.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, _ = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )

        if prior_probs is None:
            prior_probs = np.ones(self.module.feature_mapper_up.n_features)*0.5
        elif not isinstance(prior_probs, np.ndarray):
            prior_probs = np.array(prior_probs)

        prior_probs_tensor = torch.tensor(prior_probs, dtype=torch.float32).view(1, -1).to(device)

        criterion = torch.nn.MSELoss()
        pretrain_optimizer = torch.optim.Adam(self.module.feature_mapper_up.att_net.parameters(), lr=pretrain_lr)

        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="pretraining", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)

                attention_scores = self.module.feature_mapper_up.att_net(inputs_up)
                # Repeat prior_probs_tensor to match the batch size
                repeated_prior_probs = prior_probs_tensor.repeat(attention_scores.size(0), 1)

                loss = criterion(torch.sigmoid(attention_scores), repeated_prior_probs)

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()

        print("Pretraining attention net completed.")

    def plot_train_losses(self, fig_size=(8, 8)):
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 4, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # show figure
        plt.show()

    def get_up_feature_weights(
            self,
            method: Optional[str] = "SHAP",
            n_bg_samples: Optional[int] = 100,
            grad_source: Optional[str] = "prob",
            normalize: Optional[bool] = True,
            sort_by_weight: Optional[bool] = True):
        r"""
        Return the weights of features.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata_batch = batch_sampler(self.adata, self.batch_size, shuffle=False)

        def compute_shap_weights(key="prob"):
            # key = "prob" or "logit"
            shap_weights_full = []
            idx = np.random.permutation(self.adata.shape[0])[0:n_bg_samples]
            background_data = torch.tensor(self.adata.X[idx], dtype=torch.float32)
            background_data = background_data.to(device)

            model = ShapModel1(self.module, key).to(device)
            explainer = shap.DeepExplainer(model, background_data)

            for data in adata_batch:
                inputs_up = torch.tensor(data.X, dtype=torch.float32, device=device)
                shap_value = explainer.shap_values(inputs_up)
                shap_weights_full.append(shap_value)

            return np.concatenate(shap_weights_full, axis=0)

        def compute_grad_weights(grad_source="prob"):
            grad_weights_full = []
            for data in adata_batch:
                inputs_up = torch.tensor(data.X, dtype=torch.float32, device=device)
                inputs_down = torch.tensor(data.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(data.obs['labels'], dtype=torch.float32, device=device)

                inputs_up.requires_grad = True
                model_outputs = self.module(inputs_up, use_mean=True)
                
                if grad_source == "loss":
                    loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down, labels)
                    dpd_loss = loss_dict['dpd_loss']
                    dpd_loss.sum().backward()  # mean()
                elif grad_source == "prob":
                    prob = model_outputs["alpha_dpd"]["prob"]  # prob
                    prob.sum().backward()
                elif grad_source == 'logit':
                    prob = model_outputs["alpha_dpd"]["logit"]
                    prob.sum().backward()
                grad_weights_full.append(inputs_up.grad.cpu().numpy())

            return np.concatenate(grad_weights_full, axis=0)

        def compute_model_weights():
            if self.module.feature_mapper_up.attention:
                attention_weights_full = []
                for data in adata_batch:
                    inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
                    model_outputs = self.module(inputs, use_mean=True)
                    att_w = model_outputs["feat_w_up"].cpu().detach().numpy()
                    attention_weights_full.append(att_w)
                weight_matrix = np.concatenate(attention_weights_full, axis=0)
            else:
                weight_vector = torch.sigmoid(self.module.feature_mapper_up.weight).cpu().detach().numpy()
                # Expand weight vector to a matrix with the same weight vector repeated for each sample in adata_batch
                weight_matrix = np.tile(weight_vector, (len(self.adata), 1))
            return weight_matrix

        weights_full = None
        if method == "Model":
            weights_full = compute_model_weights()
        elif method == "SHAP":
            weights_full = compute_shap_weights()
        elif method == "Grad":
            weights_full = compute_grad_weights(grad_source=grad_source)
        elif method == "Ensemble":
            model_weights = np.abs(compute_model_weights())
            shap_weights = np.abs(compute_shap_weights())
            grad_weights = np.abs(compute_grad_weights())

            # Normalize each set of weights
            model_sum = np.sum(model_weights, axis=1, keepdims=True)
            model_weights = np.where(model_sum != 0, model_weights / model_sum, 0)

            shap_sum = np.sum(shap_weights, axis=1, keepdims=True)
            shap_weights = np.where(shap_sum != 0, shap_weights / shap_sum, 0)

            grad_sum = np.sum(grad_weights, axis=1, keepdims=True)
            grad_weights = np.where(grad_sum != 0, grad_weights / grad_sum, 0)

            # Combine the weights
            weights_full = (model_weights + shap_weights + grad_weights) / 3

        # Get the mean of the weights for each feature
        weights = np.mean(np.abs(weights_full), axis=0)

        # Normalize the weights if required
        if normalize:
            weights = weights / np.sum(weights)

        # Create a new DataFrame with the weights
        weights_df = self.adata.var.copy()
        weights_df['weight'] = weights

        # Sort the DataFrame by weight if required
        if sort_by_weight:
            weights_df = weights_df.sort_values(by='weight', ascending=False)

        return weights_df, weights_full

    @torch.no_grad()
    def get_down_feature_weights(self, normalize: Optional[bool] = True, sort_by_weight: Optional[bool] = True):
        r"""
        Return the weights of features.
        """

        def process_weights(feature_mapper, feature_names, original_df):
            weights = feature_mapper.weight.cpu().detach().numpy()
            weights = np.maximum(weights, 0)
            if normalize:
                weights = weights / np.sum(weights)
            weights_df = pd.DataFrame(weights, index=feature_names, columns=['weight'])
            final_df = original_df.copy().join(weights_df)
            if sort_by_weight:
                final_df = final_df.sort_values(by='weight', ascending=False)
            return final_df

        # final_df_up = process_weights(self.module.feature_mapper_up, self.adata.var_names, self.adata.var)
        final_df_down = process_weights(self.module.feature_mapper_down, self.adata.uns['X_down_feature'].index,
                                        self.adata.uns['X_down_feature'])

        return final_df_down

    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        logits = []
        probs = []
        preds = []
        x_down_rec_alpha = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs, use_mean=True)
            latent_z = torch.cat([model_outputs["latent1"]["z"], model_outputs["latent2"]["z"]], dim=1)
            latent.append(latent_z.cpu().numpy())
            # latent.append(model_outputs['latent_up']['qz_m'].cpu().numpy())
            logits.append(model_outputs['alpha_dpd']['logit'].cpu().numpy())
            probs.append(model_outputs["alpha_dpd"]["prob"].cpu().numpy())
            preds.append(np.int_(model_outputs['alpha_dpd']['prob'].cpu().numpy() > 0.5))
            x_down_rec_alpha.append(model_outputs["x_down_rec_alpha"].cpu().numpy())

        output = dict(latent=np.concatenate(latent, axis=0),
                      logits=np.concatenate(logits, axis=0),
                      probs=np.concatenate(probs, axis=0),
                      preds=np.concatenate(preds, axis=0),
                      x_down_rec_alpha=np.concatenate(x_down_rec_alpha, axis=0))

        return output

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            plot_info_flow: Optional[bool] = True,
            save_fig: Optional[bool] = False,
            save_dir: Optional[str] = None,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim_v1(ce_params, self.module, inputs, i, j, alpha_vi=False, beta_vi=True,
                                                  device=device)
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        info_flow = info_flow.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)

        # Calculate information flow for causal and spurious dimensions
        dims = ['causal', 'spurious']
        info_flow_cat = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            # Get the latent space of the current sample
            inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
            # Calculate the information flow
            info_c, _ = joint_uncond_v1(ce_params, self.module, inputs, i, alpha_vi=False, beta_vi=True, device=device)
            info_s, _ = beta_info_flow_v1(ce_params, self.module, inputs, i, alpha_vi=True, beta_vi=False,
                                          device=device)
            info_flow_cat.loc[i, 'causal'] = -info_c.item()
            info_flow_cat.loc[i, 'spurious'] = -info_s.item()
        info_flow_cat.set_index(adata.obs_names, inplace=True)
        info_flow_cat = info_flow_cat.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)

        if plot_info_flow:
            # plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(save_dir + "info_flow.png")
            plt.show()

            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow_cat, palette="pastel")
            plt.xlabel("Dimensions")
            plt.ylabel("Information Measurements")
            if save_fig:
                plt.savefig(save_dir + "info_flow_cat.png")
            plt.show()

        return info_flow, info_flow_cat

    def perform_state_transition(
            self,
            adata=None,
            causal_features=None,
            causal_idx=None,  # Causal feature indices
            grad_source="prob",  # gradient source
            lr=0.01,  # learning rate
            max_iter=100,  # number of iterations
            min_iter=10,  # minimum number of iterations
            optimizer_type="Adam",  # optimizer type
            save_step=1,  # interval for saving the data
            stop_thresh=1e-8,  # early stopping threshold
            control_direction="increase",  # control direction
            num_sampling=200,  # number of sampling
            verbose=False,  # print training process
    ):
        self.module.eval() if self.module.training else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata.copy() if adata is not None else self.adata.copy()
        # Determine causal indices from causal features if provided
        if causal_features is not None:
            causal_idx = [adata.var_names.get_loc(feat) for feat in causal_features]
        elif causal_idx is None:
            causal_idx = list(range(adata.shape[1]))
            print("Warning: No causal features or indices provided. Using all features.")

        causal_update = {}
        causal_sampling = {}  # causal sampling
        control_details = pd.DataFrame()

        for i, sample in enumerate(adata.X):
            orig_causal_sample = sample[causal_idx].copy()  # Original causal features
            causal_sample = sample[causal_idx]
            sample_update = []
            initial_prob = None
            last_prob = None  # last prob
            print(f"Processing sample {i}, Target direction: {control_direction}")

            tensor_sample = torch.tensor(sample, dtype=torch.float32, device=device)
            causal_tensor = torch.tensor(causal_sample, dtype=torch.float32, device=device, requires_grad=True)

            # Initialize optimizer for causal_tensor
            if optimizer_type == "Adam":  # default
                optimizer = optim.Adam([causal_tensor], lr=lr)
            elif optimizer_type == "SGD":  # not recommended
                optimizer = optim.SGD([causal_tensor], lr=lr)
            elif optimizer_type == "RMSprop":  # adaptive learning rate
                optimizer = optim.RMSprop([causal_tensor], lr=lr)
            # elif optimizer_type == "Adagrad":  # sparse data
            #     optimizer = optim.Adagrad([causal_tensor], lr=lr)
            # elif optimizer_type == "AdamW":  # adam with weight decay
            #     optimizer = optim.AdamW([causal_tensor], lr=lr)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            # =================== causal feature update ===================
            prob = None
            for iter in range(max_iter):
                optimizer.zero_grad()
                tensor_sample = tensor_sample.clone().detach()  # Clone and detach tensor_sample
                tensor_sample[causal_idx] = causal_tensor

                # forward propagation
                outputs = self.module(tensor_sample.unsqueeze(0), use_mean=True)
                prob = outputs["alpha_dpd"]["prob"]
                logit = outputs["alpha_dpd"]["logit"]
                current_prob = prob.item()

                # initial_prob
                if iter == 0:
                    initial_prob = current_prob
                else:
                    prob_change = current_prob - last_prob
                    if iter > min_iter and abs(prob_change) < stop_thresh:
                        print(f"Early stopping at iteration {iter} for sample {i}")
                        break
                last_prob = current_prob  # update last prob

                # backward propagation
                target = logit if grad_source == "logit" else prob
                target = -target if control_direction == "increase" else target
                target.backward()

                # update causal features
                optimizer.step()

                # save updated sample and probability
                if iter % save_step == 0:
                    x_delta = np.linalg.norm(causal_tensor.detach().cpu().numpy() - orig_causal_sample)
                    record = {'iteration': iter, 'prob': prob.item(), 'x_delta': x_delta}
                    if verbose:
                        print(record)
                    for feature_name, feature_value in zip(adata.var_names[causal_idx],
                                                           tensor_sample[causal_idx].detach().cpu().numpy()):
                        record[feature_name] = feature_value
                    sample_update.append(record)

            # Convert updates to DataFrame and store
            update_data = pd.DataFrame(sample_update)
            causal_update[i] = update_data

            # ==================== calculate controllability score ====================
            causal_delta = np.linalg.norm(orig_causal_sample - causal_tensor.detach().cpu().numpy())
            prob_delta = abs(prob.item() - initial_prob)
            score = prob_delta / (max(np.log(iter), 1) * causal_delta)
            control_item = {
                'sample_idx': int(i),
                'sample_name': adata.obs_names[i],  # sample name
                'score': score,
                'prob_delta': prob_delta,
                'causal_delta': causal_delta,
                'n_iter': iter,
            }
            control_item_df = pd.DataFrame.from_dict(control_item, orient='index').T
            control_details = pd.concat([control_details, control_item_df], ignore_index=True)

            # causal sampling for surface plot
            feature_columns = update_data.columns[3:]  # causal feature columns

            # Sampling from the causal feature space
            sampled_points = np.zeros((num_sampling, len(feature_columns)))

            for j, feature in enumerate(feature_columns):
                min_value = adata.X[:, causal_idx[j]].min()
                max_value = adata.X[:, causal_idx[j]].max()
                # min_value = update_data[feature].min()
                # max_value = update_data[feature].max()
                sampled_points[:, j] = np.random.uniform(low=min_value, high=max_value, size=num_sampling)

            # =================== sampling from the causal feature space ===================
            batch_samples = np.tile(sample, (num_sampling, 1))  # repeat the sample
            batch_samples[:, causal_idx] = sampled_points  # replace causal features

            # get the probability of the sampled points
            tensor_batch_samples = torch.tensor(batch_samples, dtype=torch.float32).to(device)
            outputs = self.module(tensor_batch_samples, use_mean=True)
            probs = outputs["alpha_dpd"]["prob"].detach().cpu().numpy()

            # concat sampled points and probability
            sampled_data = pd.DataFrame(sampled_points, columns=feature_columns)
            sampled_data['prob'] = probs
            causal_sampling[i] = sampled_data

        # save updated data and control score
        adata.uns["causal_update"] = causal_update
        adata.uns["causal_sampling"] = causal_sampling
        adata.uns["control_details"] = control_details
        adata.uns["control_direction"] = control_direction

        return adata


class ShapModel1(nn.Module):
    def __init__(self, original_model, key='prob'):
        super().__init__()
        self.original_model = original_model
        self.key = key

    def forward(self, x):
        model_outputs = self.original_model(x, use_mean=True)
        output = model_outputs["alpha_dpd"][self.key]
        return output

    # def forward(self, x):
    #     x_up = x
    #     feature_mapper_up = self.original_model.feature_mapper_up
    #     feature_mapper_down = self.original_model.feature_mapper_down
    #     w_up = torch.sigmoid(feature_mapper_up.weight)
    #     w_down = torch.relu(feature_mapper_down.weight)
    #
    #     x1 = torch.mul(x, w_up)
    #     latent1 = self.original_model.encoder1(x1)
    #     x2 = torch.mul(x, 1 - w_up)
    #     latent2 = self.original_model.encoder2(x2)
    #     z = torch.cat((latent1["z"], latent2["z"]), dim=1)
    #
    #     x_down_pred = self.original_model.decoder_down(z)
    #     dpd_x = torch.mul(x_down_pred, w_down)
    #     org_dpd = self.original_model.dpd_model(dpd_x)
    #
    #     alpha_z = torch.zeros_like(z)
    #     alpha_z[:, :self.original_model.n_causal] = latent1["z"]
    #     alpha_z[:, self.original_model.n_causal:] = latent2["z"].mean(dim=0, keepdim=True)
    #     x_down_pred_alpha = self.original_model.decoder_down(alpha_z)
    #     x2_alpha = torch.mul(x_down_pred_alpha, w_down)
    #     alpha_dpd = self.original_model.dpd_model(x2_alpha)
    #
    #     return alpha_dpd['prob']
    #     # return alpha_dpd['logit']


class CausalFlow(nn.Module):
    """
    Casual control of phenotype and state transitions
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            n_causal: int = 2,  # Number of casual factors
            n_controls: int = 10,  # Number of upstream causal features
            **model_kwargs,
    ):
        super(CausalFlow, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_controls = n_controls
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = CauVAE(
            n_input_up=adata.X.shape[1],
            n_input_down=adata.obsm['X_down'].shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            n_controls=n_controls,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 5,
            n_alpha: int = 25,
            n_beta: int = 100,
            recons_weight: float = 1.0,
            kl_weight: float = 0.02,
            up_weight: float = 1.0,
            down_weight: float = 1.0,
            feat_l1_weight: float = 0.05,
            dpd_weight: float = 3.0,
            fide_kl_weight: float = 0.05,
            causal_weight: float = 1.0,
            down_fold: float = 1.0,
            causal_fold: float = 1.0,
            spurious_fold: float = 1.0,
            stage_training: bool = True,
            weight_scheme: str = None,
            **kwargs,
    ):
        """
        Trains the model using fractal variational autoencoder.
        """
        # torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            'N_alpha': n_alpha,
            'N_beta': n_beta,
            'K': self.n_causal,
            'L': self.n_latent - self.n_causal,
            'z_dim': self.n_latent,
            'M': 2}
        self.ce_params = ce_params
        loss_weights = {
            'up_rec_loss': up_weight * recons_weight,
            'down_rec_loss': down_weight * recons_weight,
            'up_kl_loss': kl_weight,
            'feat_l1_loss_up': feat_l1_weight,
            'feat_l1_loss_down': feat_l1_weight * down_fold,
            'dpd_loss': dpd_weight,
            'fide_kl_loss': fide_kl_weight,
            'causal_loss': causal_weight,
        }

        self.batch_size = batch_size
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        epoch_losses = {'total_loss': [], 'up_rec_loss': [], 'down_rec_loss': [], 'up_kl_loss': [],
                        'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [], 'fide_kl_loss': [],
                        'causal_loss': []}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_losses = {'total_loss': [], 'up_rec_loss': [], 'down_rec_loss': [], 'up_kl_loss': [],
                            'feat_l1_loss_up': [], 'feat_l1_loss_down': [], 'dpd_loss': [], 'fide_kl_loss': [],
                            'causal_loss': []}
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(epoch, max_epochs, scheme=weight_scheme)
            for train_batch in train_adata_batch:
                inputs_up = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                inputs_down = torch.tensor(train_batch.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(train_batch.obs['labels'], dtype=torch.float32, device=device)
                model_outputs = self.module(inputs_up)
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down, labels)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    _causal_loss1, _ = joint_uncond(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                    beta_vi=True, device=device)
                    _causal_loss2, _ = beta_info_flow(ce_params, self.module, inputs_up, idx, alpha_vi=True,
                                                      beta_vi=False, device=device)
                    _causal_loss = _causal_loss1 * causal_fold - _causal_loss2 * spurious_fold
                    # _causal_loss = _causal_loss1 - _causal_loss2 * 3.0
                    causal_loss_list += [_causal_loss]
                up_rec_loss = loss_dict['up_rec_loss'].mean()
                down_rec_loss = loss_dict['down_rec_loss'].mean()
                up_kl_loss = loss_dict['up_kl_loss'].mean()
                feat_l1_loss_up = loss_dict['feat_l1_loss_up'].mean()
                feat_l1_loss_down = loss_dict['feat_l1_loss_down'].mean()
                dpd_loss = loss_dict['dpd_loss'].mean()
                fide_kl_loss = loss_dict['fide_kl_loss'].mean()
                causal_loss = torch.stack(causal_loss_list).mean()

                total_loss = loss_weights['up_rec_loss'] * up_rec_loss + \
                             loss_weights['down_rec_loss'] * down_rec_loss + \
                             loss_weights['up_kl_loss'] * up_kl_loss + \
                             loss_weights['feat_l1_loss_up'] * feat_l1_loss_up + \
                             loss_weights['feat_l1_loss_down'] * feat_l1_loss_down * down_fold + \
                             loss_weights['dpd_loss'] * dpd_loss + \
                             loss_weights['fide_kl_loss'] * fide_kl_loss + \
                             loss_weights['causal_loss'] * causal_loss

                optimizer.zero_grad()
                # with torch.autograd.detect_anomaly():
                #     total_loss.backward()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()

                # update batch losses
                batch_losses['total_loss'].append(total_loss.item())
                batch_losses['up_rec_loss'].append(up_rec_loss.item())
                batch_losses['down_rec_loss'].append(down_rec_loss.item())
                batch_losses['up_kl_loss'].append(up_kl_loss.item())
                batch_losses['feat_l1_loss_up'].append(feat_l1_loss_up.item())
                batch_losses['feat_l1_loss_down'].append(feat_l1_loss_down.item())
                batch_losses['dpd_loss'].append(dpd_loss.item())
                batch_losses['fide_kl_loss'].append(fide_kl_loss.item())
                batch_losses['causal_loss'].append(causal_loss.item())

            # update epochs losses
            epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
            epoch_losses['up_rec_loss'].append(np.mean(batch_losses['up_rec_loss']))
            epoch_losses['down_rec_loss'].append(np.mean(batch_losses['down_rec_loss']))
            epoch_losses['up_kl_loss'].append(np.mean(batch_losses['up_kl_loss']))
            epoch_losses['feat_l1_loss_up'].append(np.mean(batch_losses['feat_l1_loss_up']))
            epoch_losses['feat_l1_loss_down'].append(np.mean(batch_losses['feat_l1_loss_down']))
            epoch_losses['dpd_loss'].append(np.mean(batch_losses['dpd_loss']))
            epoch_losses['fide_kl_loss'].append(np.mean(batch_losses['fide_kl_loss']))
            epoch_losses['causal_loss'].append(np.mean(batch_losses['causal_loss']))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses['total_loss'])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def plot_train_losses(self, fig_size=(8, 8)):
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 3, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # show figure
        plt.show()

    def get_up_feature_weights(
            self,
            method: Optional[str] = "SHAP",
            n_bg_samples: Optional[int] = 100,
            normalize: Optional[bool] = True,
            sort_by_weight: Optional[bool] = True):
        r"""
        Return the weights of features.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata_batch = batch_sampler(self.adata, self.batch_size, shuffle=False)

        def compute_shap_weights():
            shap_weights_full = []
            idx = np.random.permutation(self.adata.shape[0])[0:n_bg_samples]
            background_data = torch.tensor(self.adata.X[idx], dtype=torch.float32)
            background_data = background_data.to(device)

            model = ShapModel(self.module).to(device)
            explainer = shap.DeepExplainer(model, background_data)

            for data in adata_batch:
                inputs_up = torch.tensor(data.X, dtype=torch.float32, device=device)

                shap_value = explainer.shap_values(inputs_up)
                shap_weights_full.append(shap_value)

            return np.concatenate(shap_weights_full, axis=0)

        def compute_grad_weights():
            grad_weights_full = []
            for data in adata_batch:
                inputs_up = torch.tensor(data.X, dtype=torch.float32, device=device)
                inputs_down = torch.tensor(data.obsm['X_down'], dtype=torch.float32, device=device)
                labels = torch.tensor(data.obs['labels'], dtype=torch.float32, device=device)

                inputs_up.requires_grad = True
                model_outputs = self.module(inputs_up)
                loss_dict = self.module.compute_loss(model_outputs, inputs_up, inputs_down, labels)
                dpd_loss = loss_dict['dpd_loss']
                dpd_loss.mean().backward()
                grad_weights_full.append(inputs_up.grad.cpu().numpy())

            return np.concatenate(grad_weights_full, axis=0)

        def compute_model_weights():
            weight_vector = torch.relu(self.module.feature_mapper_up.weight).cpu().detach().numpy()
            # Expand the weight vector to a matrix with the same weight vector repeated for each sample in adata_batch
            weight_matrix = np.tile(weight_vector, (len(self.adata), 1))
            return weight_matrix

        weights_full = None
        if method == "SHAP":
            weights_full = compute_shap_weights()
        elif method == "Grad":
            weights_full = compute_grad_weights()
        elif method == "Both":
            shap_weights = np.abs(compute_shap_weights())
            grad_weights = np.abs(compute_grad_weights())
            # Normalize shap_weights if sum is not zero, otherwise keep as zeros
            shap_sum = np.sum(shap_weights, axis=1, keepdims=True)
            shap_weights = np.where(shap_sum != 0, shap_weights / shap_sum, 0)
            # Normalize grad_weights if sum is not zero, otherwise keep as zeros
            grad_sum = np.sum(grad_weights, axis=1, keepdims=True)
            grad_weights = np.where(grad_sum != 0, grad_weights / grad_sum, 0)
            weights_full = (shap_weights + grad_weights) / 2

        elif method == "Model":
            weights_full = compute_model_weights()

        # Get the mean of the weights for each feature
        weights = np.mean(np.abs(weights_full), axis=0)

        # Normalize the weights if required
        if normalize:
            weights = weights / np.sum(weights)

        # Create a new DataFrame with the weights
        weights_df = self.adata.var.copy()
        weights_df['weight'] = weights

        # Sort the DataFrame by weight if required
        if sort_by_weight:
            weights_df = weights_df.sort_values(by='weight', ascending=False)

        return weights_df

    @torch.no_grad()
    def get_down_feature_weights(self, normalize: Optional[bool] = True, sort_by_weight: Optional[bool] = True):
        r"""
        Return the weights of features.
        """

        def process_weights(feature_mapper, feature_names, original_df):
            weights = feature_mapper.weight.cpu().detach().numpy()
            weights = np.maximum(weights, 0)
            if normalize:
                weights = weights / np.sum(weights)
            weights_df = pd.DataFrame(weights, index=feature_names, columns=['weight'])
            final_df = original_df.copy().join(weights_df)
            if sort_by_weight:
                final_df = final_df.sort_values(by='weight', ascending=False)
            return final_df

        # final_df_up = process_weights(self.module.feature_mapper_up, self.adata.var_names, self.adata.var)
        final_df_down = process_weights(self.module.feature_mapper_down, self.adata.uns['X_down_feature'].index,
                                        self.adata.uns['X_down_feature'])

        return final_df_down

    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        logits = []
        preds = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs)
            latent.append(model_outputs['latent_up']['qz_m'].cpu().numpy())
            logits.append(model_outputs['alpha_dpd']['logit'].cpu().numpy())
            preds.append((model_outputs['alpha_dpd']['prob'].cpu().numpy() > 0.5).astype(np.int))

        output = dict(latent=np.concatenate(latent, axis=0),
                      logits=np.concatenate(logits, axis=0),
                      preds=np.concatenate(preds, axis=0))

        return output

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            plot_info_flow: Optional[bool] = True,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim(ce_params, self.module, inputs, i, j, alpha_vi=False, beta_vi=True,
                                               device=device)
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        info_flow = info_flow.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)

        if plot_info_flow:
            # plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette='pastel')
            plt.xlabel('Dimensions')
            plt.ylabel('Information Measurements')
            plt.show()
        return info_flow


class ShapModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model

    def forward(self, x):
        x_up = x
        feature_mapper_up = self.original_model.feature_mapper_up
        feature_mapper_down = self.original_model.feature_mapper_down
        w_up = torch.relu(feature_mapper_up.weight)
        w_down = torch.relu(feature_mapper_down.weight)

        x1 = torch.mul(x, w_up)
        latent_up = self.original_model.encoder(x1)
        # x_up_rec = self.decoder_up(latent_up['z'])

        x_down_pred = self.original_model.decoder_down(latent_up['z'])
        x2 = torch.mul(x_down_pred, w_down)
        org_dpd = self.original_model.dpd_model(x2)

        alpha_z = torch.zeros_like(latent_up['z'])
        alpha_z[:, :self.original_model.n_causal] = latent_up['z'][:, :self.original_model.n_causal]
        x_down_pred_alpha = self.original_model.decoder_down(alpha_z)
        x2_alpha = torch.mul(x_down_pred_alpha, w_down)
        alpha_dpd = self.original_model.dpd_model(x2_alpha)

        return alpha_dpd['prob']
        # return alpha_dpd['logit']


class CCPSTModel(nn.Module):
    """
    Casual control of phenotype and state transitions
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 20,
            n_causal: int = 2,  # Number of casual factors
            n_controls: int = 10,  # Number of upstream features
            **model_kwargs,
    ):
        super(CCPSTModel, self).__init__()
        self.adata = adata
        self.train_adata = None
        self.val_adata = None
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_controls = n_controls
        self.batch_size = None
        self.ce_params = None
        self.history = {}

        self.module = FVAE(
            n_input=adata.X.shape[1],
            n_latent=n_latent,
            n_causal=n_causal,
            n_controls=n_controls,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 5e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 1.0,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = False,
            weight_decay: float = 1e-6,
            n_x: int = 3,
            n_alpha: int = 25,
            n_beta: int = 100,
            recons_weight: float = 1.0,
            kl_weight: float = 0.02,
            glo_weight: float = 1.0,
            sub_weight: float = 2.0,
            feat_l1_weight: float = 0.05,
            dpd_weight: float = 3.0,
            fide_kl_weight: float = 0.05,
            causal_weight: float = 1.0,
            spurious_fold: float = 2.0,
            stage_training: bool = True,
            **kwargs,
    ):
        """
        Trains the model using fractal variational autoencoder.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            use_gpu=use_gpu,
        )
        self.train_adata, self.val_adata = train_adata, val_adata
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        ce_params = {
            'N_alpha': n_alpha,
            'N_beta': n_beta,
            'K': self.n_causal,
            'L': self.n_latent - self.n_causal,
            'z_dim': self.n_latent,
            'M': 2}
        self.ce_params = ce_params
        loss_weights = {
            'sub_rec_loss': sub_weight * recons_weight,
            'glo_rec_loss': glo_weight * recons_weight,
            'sub_kl_loss': sub_weight * recons_weight,
            'glo_kl_loss': glo_weight * kl_weight,
            'feat_l1_loss': feat_l1_weight,
            'dpd_loss': dpd_weight,
            'fide_kl_loss': fide_kl_weight,
            'causal_loss': causal_weight}
        self.batch_size = batch_size
        params_w = list(self.module.feature_selector.parameters())
        params_net = list(self.module.encoder.parameters()) + list(self.module.decoder.parameters()) + list(
            self.module.dpd_model.parameters())
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer1 = optim.Adam(params_w, lr=lr, weight_decay=weight_decay)
        # optimizer2 = optim.Adam(params_net, lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_losses = {'total_loss': [], 'sub_rec_loss': [], 'glo_rec_loss': [], 'sub_kl_loss': [],
                        'glo_kl_loss': [], 'feat_l1_loss': [], 'dpd_loss': [], 'fide_kl_loss': [], 'causal_loss': []}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            # print('Epoch: ', epoch)
            # if epoch == 346:
            #     print('stop')
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_losses = {'total_loss': [], 'sub_rec_loss': [], 'glo_rec_loss': [], 'sub_kl_loss': [],
                            'glo_kl_loss': [], 'feat_l1_loss': [], 'dpd_loss': [], 'fide_kl_loss': [],
                            'causal_loss': []}
            if stage_training:
                # loss_weights = self.module.update_loss_weights_sc(epoch, max_epochs, loss_weights)
                loss_weights = self.module.update_loss_weights(epoch, max_epochs, loss_weights)
            for train_batch in train_adata_batch:
                inputs = torch.tensor(train_batch.X, dtype=torch.float32, device=device)
                labels = torch.tensor(train_batch.obs['labels'], dtype=torch.float32, device=device)
                model_outputs = self.module(inputs)
                loss_dict = self.module.compute_loss(model_outputs, inputs, labels)

                causal_loss_list = []
                for idx in np.random.permutation(train_batch.shape[0])[:n_x]:
                    _causal_loss1, _ = joint_uncond(ce_params, self.module, inputs, idx, device=device)
                    _causal_loss2, _ = beta_info_flow(ce_params, self.module, inputs, idx, device=device)
                    _causal_loss = _causal_loss1 - _causal_loss2 * spurious_fold
                    # _causal_loss = _causal_loss1 - _causal_loss2 * 3.0
                    causal_loss_list += [_causal_loss]
                sub_rec_loss = loss_dict['sub_rec_loss'].mean()
                glo_rec_loss = loss_dict['glo_rec_loss'].mean()
                sub_kl_loss = loss_dict['sub_kl_loss'].mean()
                glo_kl_loss = loss_dict['glo_kl_loss'].mean()
                feat_l1_loss = loss_dict['feat_l1_loss'].mean()
                dpd_loss = loss_dict['dpd_loss'].mean()
                fide_kl_loss = loss_dict['fide_kl_loss'].mean()
                causal_loss = torch.stack(causal_loss_list).mean()

                loss_weights['fide_kl_loss'] = 0.0
                total_loss = loss_weights['sub_rec_loss'] * sub_rec_loss + loss_weights['glo_rec_loss'] * glo_rec_loss + \
                             loss_weights['sub_kl_loss'] * sub_kl_loss + loss_weights['glo_kl_loss'] * glo_kl_loss + \
                             loss_weights['feat_l1_loss'] * feat_l1_loss + loss_weights['dpd_loss'] * dpd_loss + \
                             loss_weights['fide_kl_loss'] * fide_kl_loss + loss_weights['causal_loss'] * causal_loss

                # if (epoch+1) % 2 == 0:
                #     optimizer1.zero_grad()
                #     total_loss.backward(retain_graph=True)
                #     nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                #     optimizer1.step()
                # else:
                #     optimizer2.zero_grad()
                #     total_loss.backward()
                #     nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                #     optimizer2.step()
                optimizer.zero_grad()
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()
                # print("=====" * 10)
                # for name, param in self.module.named_parameters():
                #     if param.grad is not None:
                #         print("Parameter Name:", name, param.shape)
                #         print("Gradient:")
                #         print(param.grad.min(), param.grad.max(), param.grad.mean())
                #         print("Value:")
                #         print(param.data.min(), param.data.max(), param.data.mean())

                # update batch losses
                batch_losses['total_loss'].append(total_loss.item())
                batch_losses['sub_rec_loss'].append(sub_rec_loss.item())
                batch_losses['glo_rec_loss'].append(glo_rec_loss.item())
                batch_losses['sub_kl_loss'].append(sub_kl_loss.item())
                batch_losses['glo_kl_loss'].append(glo_kl_loss.item())
                batch_losses['feat_l1_loss'].append(feat_l1_loss.item())
                batch_losses['dpd_loss'].append(dpd_loss.item())
                batch_losses['fide_kl_loss'].append(fide_kl_loss.item())
                batch_losses['causal_loss'].append(causal_loss.item())

            # scheduler.step()
            # update epochs losses
            epoch_losses['total_loss'].append(np.mean(batch_losses['total_loss']))
            epoch_losses['sub_rec_loss'].append(np.mean(batch_losses['sub_rec_loss']))
            epoch_losses['glo_rec_loss'].append(np.mean(batch_losses['glo_rec_loss']))
            epoch_losses['sub_kl_loss'].append(np.mean(batch_losses['sub_kl_loss']))
            epoch_losses['glo_kl_loss'].append(np.mean(batch_losses['glo_kl_loss']))
            epoch_losses['feat_l1_loss'].append(np.mean(batch_losses['feat_l1_loss']))
            epoch_losses['dpd_loss'].append(np.mean(batch_losses['dpd_loss']))
            epoch_losses['fide_kl_loss'].append(np.mean(batch_losses['fide_kl_loss']))
            epoch_losses['causal_loss'].append(np.mean(batch_losses['causal_loss']))

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                total_loss = np.mean(batch_losses['total_loss'])
                logging.info(f"Epoch {epoch} training loss: {total_loss:.4f}")

        self.history = epoch_losses

    def plot_train_losses(self, fig_size=(8, 8)):
        # Set figure size
        fig = plt.figure(figsize=fig_size)
        if self.history is None:
            raise ValueError("You should train the model first!")
        epoch_losses = self.history
        # Plot a subplot of each loss
        for i, loss_name in enumerate(epoch_losses.keys()):
            # Gets the value of the current loss
            loss_values = epoch_losses[loss_name]
            # Create subplot
            ax = fig.add_subplot(3, 3, i + 1)
            # Draw subplot
            ax.plot(range(len(loss_values)), loss_values)
            # Set the subplot title
            ax.set_title(loss_name)
            # Set the subplot x-axis and y-axis labels
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

        # adjust the distance and edges between sub-graphs
        plt.tight_layout()
        # show figure
        plt.show()

    @torch.no_grad()
    def get_top_features(
            self,
            n_controls: Optional[int] = None,
            normalize: Optional[bool] = True,
    ):
        r"""
        Return the top k control features.
        """
        k = n_controls if n_controls is not None else self.n_controls
        feature_names = list(self.adata.var_names)
        weights = self.module.feature_selector.weight.cpu().detach().numpy()
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights) if normalize else weights

        top_k_indices = np.argsort(-weights)[:k]
        top_k_names = [feature_names[i] for i in top_k_indices]
        top_k_weights = weights[top_k_indices]

        return top_k_names, top_k_weights, weights

    @torch.no_grad()
    def get_model_output(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        latent = []
        dpd = []
        preds = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            inputs = torch.tensor(data.X, dtype=torch.float32, device=device)
            model_outputs = self.module(inputs)
            latent.append(model_outputs['latent_top']['qz_m'].cpu().numpy())
            dpd.append(model_outputs['alpha_dpd'].cpu().numpy())
            preds.append((model_outputs['org_prob'].cpu().numpy() > 0.5).astype(np.int))

        output = dict(latent=np.concatenate(latent, axis=0),
                      dpd=np.concatenate(dpd, axis=0),
                      preds=np.concatenate(preds, axis=0))

        return output

    @torch.no_grad()
    def compute_information_flow(
            self,
            adata: Optional[AnnData] = None,
            dims: Optional[List[int]] = None,
            plot_info_flow: Optional[bool] = True,
    ):
        """
        Return the latent, dpd and predict label for each sample.
        """
        if self.module.training:
            self.module.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adata = adata if adata is not None else self.adata
        ce_params = self.ce_params
        if dims is None:
            dims = list(range(self.module.n_latent))

        # Calculate information flow
        info_flow = pd.DataFrame(index=range(adata.shape[0]), columns=dims)
        for i in range(adata.shape[0]):
            for j in dims:
                # Get the latent space of the current sample
                inputs = torch.tensor(adata.X, dtype=torch.float32, device=device)
                # Calculate the information flow
                info = joint_uncond_single_dim(ce_params, self.module, inputs, i, j, alpha_vi=False, beta_vi=True,
                                               device=device)
                info_flow.loc[i, j] = info.item()
        info_flow.set_index(adata.obs_names, inplace=True)
        info_flow = info_flow.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1)

        if plot_info_flow:
            # plot the information flow
            plt.figure(figsize=(10, 5))
            ax = sns.boxplot(data=info_flow, palette='pastel')
            plt.xlabel('Dimensions')
            plt.ylabel('Information Measurements')
            plt.show()
        return info_flow