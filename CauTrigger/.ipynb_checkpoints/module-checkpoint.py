# -*- coding: utf-8 -*-
"""
Created by Chengming Zhang, Mar 31st, 2023
"""
from typing import Dict, Iterable, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl
from torch.nn.functional import softplus

from CauTrigger.layers import FeatureSelector, FeatureSplit, FeatureWeight, Encoder, Decoder, DynamicPhenotypeDescriptor


class DualVAE(nn.Module):
    """
    Dual Variational Autoencoder for Feature Selection.
    """
    def __init__(
            self,
            n_input_up: int,
            n_input_down: int,
            n_hidden: int = 128,
            n_latent: int = 20,
            n_causal: int = 5,
            n_layers_encoder: int = 1,
            n_layers_decoder: int = 1,
            n_layers_dpd: int = 1,
            dropout_rate_encoder: float = 0.1,
            dropout_rate_decoder: float = 0.0,
            dropout_rate_dpd: float = 0.1,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            use_batch_norm_dpd: bool = True,
            init_weight=None,
            init_thresh: float = 0.3,
            attention: bool = False,
            att_mean: bool = False,
            update_down_weight: bool = False,
            decoder_linear: bool = True,
            dpd_linear: bool = True,
    ):
        super(DualVAE, self).__init__()
        self.n_input_up = n_input_up
        self.n_input_down = n_input_down
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_spurious = n_latent - n_causal
        # self.warm_up = 0
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.feature_mapper_up = FeatureSplit(
            self.n_input_up, init_weight=init_weight, init_thresh=init_thresh, thresh_grad=True,
            attention=attention, att_mean=att_mean
        )
        # self.feature_mapper_up = FeatureWeight(self.n_input_up, update_weight=update_up_weight)
        self.feature_mapper_down = FeatureWeight(self.n_input_down, update_weight=update_down_weight)
        self.encoder1 = Encoder(
            self.n_input_up,
            self.n_causal,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder1_up = Decoder(
            self.n_causal,
            self.n_input_up,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.encoder2 = Encoder(
            self.n_input_up,
            self.n_spurious,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder2_up = Decoder(
            self.n_spurious,
            self.n_input_up,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.decoder_down = Decoder(
            self.n_latent,
            self.n_input_down,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
            linear=decoder_linear,
        )
        self.dpd_model = DynamicPhenotypeDescriptor(
            self.n_input_down,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_dpd,
            n_layers=n_layers_dpd,
            batch_norm=use_batch_norm_dpd,
            linear=dpd_linear,
        )

    def forward(self, x_up, use_mean=False):
        """
        Forward pass through the whole network.
        """
        x1, feat_w_up = self.feature_mapper_up(x_up, mode="causal")
        latent1 = self.encoder1(x1)
        latent1["z"] = latent1["qz_m"] if use_mean else latent1["z"]
        x_up_rec1 = self.decoder1_up(latent1['z'])

        x2, _ = self.feature_mapper_up(x_up, mode="spurious")
        latent2 = self.encoder2(x2)
        latent2["z"] = latent2["qz_m"] if use_mean else latent2["z"]
        x_up_rec2 = self.decoder2_up(latent2['z'])

        z = torch.cat((latent1["z"], latent2["z"]), dim=1)
        x_down_rec = self.decoder_down(z)
        dpd_x, feat_w_down = self.feature_mapper_down(x_down_rec, mode="causal")
        org_dpd = self.dpd_model(dpd_x)

        alpha_z = torch.zeros_like(z)
        alpha_z[:, :self.n_causal] = latent1["z"]
        alpha_z[:, self.n_causal:] = latent2["z"].mean(dim=0, keepdim=True)
        x_down_rec_alpha = self.decoder_down(alpha_z)
        dpd_x_alpha, _ = self.feature_mapper_down(x_down_rec_alpha, mode="causal")
        alpha_dpd = self.dpd_model(dpd_x_alpha)

        return dict(
            latent1=latent1,
            latent2=latent2,
            x_up_rec1=x_up_rec1,
            x_up_rec2=x_up_rec2,
            x_down_rec=x_down_rec,
            x_down_rec_alpha=x_down_rec_alpha,
            feat_w_up=feat_w_up,
            feat_w_down=feat_w_down,
            org_dpd=org_dpd,
            alpha_dpd=alpha_dpd,
        )

    @staticmethod
    def compute_loss(model_outputs, x_up, x_down, y, imb_factor=None):
        # get model outputs
        latent1, latent2, x_up_rec1, x_up_rec2, x_down_rec, x_down_rec_alpha, feat_w_up, feat_w_down, org_dpd, alpha_dpd = model_outputs.values()
        qz1_m, qz1_v = latent1["qz_m"], latent1["qz_v"]
        qz2_m, qz2_v = latent2["qz_m"], latent2["qz_v"]
        qz_m_up = torch.cat((qz1_m, qz2_m), dim=1)
        qz_v_up = torch.cat((qz1_v, qz2_v), dim=1)
        org_logit, org_prob = org_dpd['logit'], org_dpd['prob']
        alpha_logit, alpha_prob = alpha_dpd['logit'], alpha_dpd['prob']

        # up feature reconstruction loss
        feat_w_up = feat_w_up.mean(dim=0) if feat_w_up.dim() == 2 else feat_w_up
        full_rec_loss1 = F.mse_loss(x_up_rec1, x_up, reduction="none") * feat_w_up
        up_rec_loss1 = full_rec_loss1.mean(dim=1)
        full_rec_loss2 = F.mse_loss(x_up_rec2, x_up, reduction="none") * (1 - feat_w_up)
        up_rec_loss2 = full_rec_loss2.mean(dim=1)
        # up_rec_loss = F.mse_loss(x_up_rec, x_up, reduction='none').mean(dim=1)

        # down feature reconstruction loss
        down_rec_loss = F.mse_loss(x_down_rec, x_down, reduction='none').mean(dim=1)

        # up latent kl divergence loss
        # qz_v_up_clamped = torch.clamp(qz_v_up, min=1e-8, max=5)
        # up_kl_loss = kl(Normal(qz_m_up, torch.sqrt(qz_v_up_clamped)), Normal(0, 1)).sum(dim=1)
        # up_kl_loss = kl(Normal(qz_m_up, torch.sqrt(qz_v_up+1e-8)), Normal(0, 1)).sum(dim=1)
        qz_v_up = torch.nn.functional.softplus(qz_v_up)
        up_kl_loss = 0.5 * (qz_m_up.pow(2) + qz_v_up - qz_v_up.log() - 1).sum(dim=1)

        # feature weight l1 loss
        # feat_w_up = feat_w_up.mean(dim=0) if feat_w_up.dim() == 2 else feat_w_up
        feat_l1_loss_up = torch.sum(torch.abs(feat_w_up))
        feat_l1_loss_down = torch.sum(torch.abs(feat_w_down))

        # DPD loss
        # dpd_loss = F.binary_cross_entropy(org_prob.squeeze(), y, reduction='none')
        # dpd_loss = F.binary_cross_entropy_with_logits(org_logit.squeeze(), y, reduction='none')
        if imb_factor is not None:
            num_pos = y.sum()
            num_neg = y.size(0) - num_pos
            if num_pos == 0 or num_neg == 0:
                pos_weight = torch.tensor(1.0, dtype=torch.float32, device=y.device)
            else:
                pos_weight = (num_neg / num_pos) * imb_factor
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=y.device)
        else:
            pos_weight = torch.tensor(1.0, dtype=torch.float32, device=y.device)
        # dpd_loss = F.binary_cross_entropy(org_prob.squeeze(), y, reduction="none")
        dpd_loss = F.binary_cross_entropy_with_logits(org_logit.squeeze(), y, pos_weight=pos_weight, reduction='none')

        # # fidelity kl divergence loss
        # epsilon = 1e-6
        # alpha_probs = torch.clamp(torch.cat((alpha_prob, 1 - alpha_prob), dim=1), epsilon, 1 - epsilon)
        # org_probs = torch.clamp(torch.cat((org_prob, 1 - org_prob), dim=1), epsilon, 1 - epsilon)
        # fide_kl_loss = F.kl_div(torch.log(alpha_probs), org_probs, reduction='none').sum(dim=1)
        
        # fidelity kl divergence loss
        # epsilon = 1e-6
        alpha_probs = torch.cat((alpha_prob, 1 - alpha_prob), dim=1)
        org_probs = torch.cat((org_prob, 1 - org_prob), dim=1)
        fide_kl_loss = F.kl_div(torch.log(alpha_probs+1e-8), org_probs+1e-8, reduction='none').sum(dim=1)

        # softmax_org_probs = F.softmax(org_probs, dim=1)
        # log_softmax_alpha_probs = F.log_softmax(alpha_probs, dim=1)
        # fide_kl_loss = F.kl_div(log_softmax_alpha_probs, softmax_org_probs, reduction='none').sum(dim=1)

        # alpha_probs = torch.cat((alpha_prob, 1 - alpha_prob), dim=1)
        # org_probs = torch.cat((org_prob, 1 - org_prob), dim=1)
        # fide_kl_loss = F.kl_div(torch.log(alpha_probs + 1e-8), org_probs, reduction='none').sum(dim=1)

        # Save each loss to the dictionary to return
        loss_dict = dict(
            up_rec_loss1=up_rec_loss1,
            up_rec_loss2=up_rec_loss2,
            down_rec_loss=down_rec_loss,
            up_kl_loss=up_kl_loss,
            feat_l1_loss_up=feat_l1_loss_up,
            feat_l1_loss_down=feat_l1_loss_down,
            dpd_loss=dpd_loss,
            fide_kl_loss=fide_kl_loss,
        )
        return loss_dict

    @staticmethod
    def update_loss_weights(current_epoch, max_epochs, scheme=None):
        # Update loss weights based on current epoch and maximum number of epochs
        loss_weights = None
        if scheme is None:
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {'up_rec_loss': 2.0, 'down_rec_loss': 2.0, 'up_kl_loss': 0.01, 'feat_l1_loss_up': 0.5,
                                'feat_l1_loss_down': 1.0, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {'up_rec_loss': 1.0, 'down_rec_loss': 1.0, 'up_kl_loss': 0.10, 'feat_l1_loss_up': 0.5,
                                'feat_l1_loss_down': 0.5, 'dpd_loss': 0.0, 'fide_kl_loss': 0.01, 'causal_loss': 0.01}
            elif current_epoch < max_epochs * 0.7:
                # Third quarter of training: emphasize KL loss
                loss_weights = {'up_rec_loss': 0.5, 'down_rec_loss': 0.5, 'up_kl_loss': 0.10, 'feat_l1_loss_up': 0.1,
                                'feat_l1_loss_down': 0.1, 'dpd_loss': 2.0, 'fide_kl_loss': 0.1, 'causal_loss': 0.1}
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {'up_rec_loss': 0.2, 'down_rec_loss': 0.2, 'up_kl_loss': 0.01, 'feat_l1_loss_up': 0.01,
                                'feat_l1_loss_down': 0.01, 'dpd_loss': 2.0, 'fide_kl_loss': 2.0, 'causal_loss': 2.0}
        elif scheme == 'sc':
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {'sub_rec_loss': 2.0, 'glo_rec_loss': 1.0, 'sub_kl_loss': 0.01, 'glo_kl_loss': 0.01,
                                'feat_l1_loss': 0.01, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {'sub_rec_loss': 1.0, 'glo_rec_loss': 0.5, 'sub_kl_loss': 0.2, 'glo_kl_loss': 0.1,
                                'feat_l1_loss': 0.00, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            elif current_epoch < max_epochs * 0.80:
                # Third quarter of training: emphasize KL loss
                loss_weights = {'sub_rec_loss': 0.5, 'glo_rec_loss': 0.25, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                                'feat_l1_loss': 0.00, 'dpd_loss': 2.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {'sub_rec_loss': 0.2, 'glo_rec_loss': 0.1, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                                'feat_l1_loss': 0.00, 'dpd_loss': 2.0, 'fide_kl_loss': 1.0, 'causal_loss': 1.0}

        else:
            if current_epoch < max_epochs * scheme['stage1'][0]:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {'up_rec_loss': scheme['stage1'][1], 'down_rec_loss': scheme['stage1'][2], 'up_kl_loss': scheme['stage1'][3], 'feat_l1_loss_up': scheme['stage1'][4],
                                'feat_l1_loss_down': scheme['stage1'][5], 'dpd_loss':scheme['stage1'][6], 'fide_kl_loss': scheme['stage1'][7], 'causal_loss': scheme['stage1'][8]}
            elif current_epoch < max_epochs * scheme['stage2'][0]:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {'up_rec_loss': scheme['stage2'][1], 'down_rec_loss': scheme['stage2'][2], 'up_kl_loss': scheme['stage2'][3], 'feat_l1_loss_up': scheme['stage2'][4],
                                'feat_l1_loss_down': scheme['stage2'][5], 'dpd_loss':scheme['stage2'][6], 'fide_kl_loss': scheme['stage2'][7], 'causal_loss': scheme['stage2'][8]}
            elif current_epoch < max_epochs * scheme['stage3'][0]:
                # Third quarter of training: emphasize KL loss
                loss_weights = {'up_rec_loss': scheme['stage3'][1], 'down_rec_loss': scheme['stage3'][2], 'up_kl_loss': scheme['stage3'][3], 'feat_l1_loss_up': scheme['stage3'][4],
                                'feat_l1_loss_down': scheme['stage3'][5], 'dpd_loss':scheme['stage3'][6], 'fide_kl_loss': scheme['stage3'][7], 'causal_loss': scheme['stage3'][8]}
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {'up_rec_loss': scheme['stage4'][1], 'down_rec_loss': scheme['stage4'][2], 'up_kl_loss': scheme['stage4'][3], 'feat_l1_loss_up': scheme['stage4'][4],
                                'feat_l1_loss_down': scheme['stage4'][5], 'dpd_loss':scheme['stage4'][6], 'fide_kl_loss': scheme['stage4'][7], 'causal_loss': scheme['stage4'][8]}
        return loss_weights


# FVAE model
class CauVAE(nn.Module):
    """
    Fractal Variational Autoencoder for Feature Selection.
    """

    def __init__(
            self,
            n_input_up: int,
            n_input_down: int,
            n_hidden: int = 128,
            n_latent: int = 20,
            n_causal: int = 5,
            n_controls: int = 10,
            n_layers_encoder: int = 1,
            n_layers_decoder: int = 1,
            n_layers_dpd: int = 1,
            dropout_rate_encoder: float = 0.1,
            dropout_rate_decoder: float = 0.0,
            dropout_rate_dpd: float = 0.1,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            use_batch_norm_dpd: bool = True,
            update_up_weight: bool = True,
            update_down_weight: bool = False,
            decoder_linear: bool = True,
            dpd_linear: bool = True,
    ):
        super(CauVAE, self).__init__()
        self.n_input_up = n_input_up
        self.n_input_down = n_input_down
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_controls = n_controls
        # self.warm_up = 0
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.feature_mapper_up = FeatureWeight(self.n_input_up, update_weight=update_up_weight)
        self.feature_mapper_down = FeatureWeight(self.n_input_down, update_weight=update_down_weight)
        self.encoder = Encoder(
            self.n_input_up,
            self.n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder_up = Decoder(
            self.n_latent,
            self.n_input_up,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.decoder_down = Decoder(
            self.n_latent,
            self.n_input_down,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
            linear=decoder_linear,
        )
        self.dpd_model = DynamicPhenotypeDescriptor(
            self.n_input_down,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_dpd,
            n_layers=n_layers_dpd,
            batch_norm=use_batch_norm_dpd,
            linear=dpd_linear,
        )

    def forward(self, x_up):
        """
        Forward pass through the whole network.
        """
        x1, feat_w_up = self.feature_mapper_up(x_up)
        latent_up = self.encoder(x1)
        x_up_rec = self.decoder_up(latent_up['z'])

        x_down_pred = self.decoder_down(latent_up['z'])
        x2, feat_w_down = self.feature_mapper_down(x_down_pred)
        org_dpd = self.dpd_model(x2)

        alpha_z = torch.zeros_like(latent_up['z'])
        alpha_z[:, :self.n_causal] = latent_up['z'][:, :self.n_causal]
        alpha_z[:, self.n_causal:] = latent_up['z'][:, self.n_causal:].mean(dim=0, keepdim=True)
        x_down_pred_alpha = self.decoder_down(alpha_z)
        x2_alpha, _ = self.feature_mapper_down(x_down_pred_alpha)
        alpha_dpd = self.dpd_model(x2_alpha)

        return dict(
            latent_up=latent_up,
            x_up_rec=x_up_rec,
            x_down_pred=x_down_pred,
            feat_w_up=feat_w_up,
            feat_w_down=feat_w_down,
            org_dpd=org_dpd,
            alpha_dpd=alpha_dpd,
        )

    @staticmethod
    def compute_loss(model_outputs, x_up, x_down, y):
        # get model outputs
        latent_up, x_up_rec, x_down_pred, feat_w_up, feat_w_down, org_dpd, alpha_dpd = model_outputs.values()
        qz_m_up, qz_v_up = latent_up['qz_m'], latent_up['qz_v']
        org_logit, org_prob = org_dpd['logit'], org_dpd['prob']
        alpha_logit, alpha_prob = alpha_dpd['logit'], alpha_dpd['prob']

        # up feature reconstruction loss
        up_rec_loss = F.mse_loss(x_up_rec, x_up, reduction='none').mean(dim=1)

        # down feature reconstruction loss
        down_rec_loss = F.mse_loss(x_down_pred, x_down, reduction='none').mean(dim=1)

        # up latent kl divergence loss
        # qz_v_up_clamped = torch.clamp(qz_v_up, min=1e-8, max=5)
        # up_kl_loss = kl(Normal(qz_m_up, torch.sqrt(qz_v_up_clamped)), Normal(0, 1)).sum(dim=1)
        # up_kl_loss = kl(Normal(qz_m_up, torch.sqrt(qz_v_up+1e-8)), Normal(0, 1)).sum(dim=1)
        qz_v_up = torch.nn.functional.softplus(qz_v_up)
        up_kl_loss = 0.5 * (qz_m_up.pow(2) + qz_v_up - qz_v_up.log() - 1).sum(dim=1)

        # feature weight l1 loss
        feat_l1_loss_up = torch.sum(torch.abs(feat_w_up))
        feat_l1_loss_down = torch.sum(torch.abs(feat_w_down))

        # DPD loss
        # dpd_loss = F.binary_cross_entropy(org_prob.squeeze(), y, reduction='none')
        dpd_loss = F.binary_cross_entropy_with_logits(org_logit.squeeze(), y, reduction='none')

        # fidelity kl divergence loss
        epsilon = 1e-6
        alpha_probs = torch.clamp(torch.cat((alpha_prob, 1 - alpha_prob), dim=1), epsilon, 1 - epsilon)
        org_probs = torch.clamp(torch.cat((org_prob, 1 - org_prob), dim=1), epsilon, 1 - epsilon)
        fide_kl_loss = F.kl_div(torch.log(alpha_probs), org_probs, reduction='none').sum(dim=1)

        # softmax_org_probs = F.softmax(org_probs, dim=1)
        # log_softmax_alpha_probs = F.log_softmax(alpha_probs, dim=1)
        # fide_kl_loss = F.kl_div(log_softmax_alpha_probs, softmax_org_probs, reduction='none').sum(dim=1)

        # alpha_probs = torch.cat((alpha_prob, 1 - alpha_prob), dim=1)
        # org_probs = torch.cat((org_prob, 1 - org_prob), dim=1)
        # fide_kl_loss = F.kl_div(torch.log(alpha_probs + 1e-8), org_probs, reduction='none').sum(dim=1)

        # Save each loss to the dictionary to return
        loss_dict = dict(
            up_rec_loss=up_rec_loss,
            down_rec_loss=down_rec_loss,
            up_kl_loss=up_kl_loss,
            feat_l1_loss_up=feat_l1_loss_up,
            feat_l1_loss_down=feat_l1_loss_down,
            dpd_loss=dpd_loss,
            fide_kl_loss=fide_kl_loss,
        )
        return loss_dict

    @staticmethod
    def update_loss_weights(current_epoch, max_epochs, scheme=None):
        # Update loss weights based on current epoch and maximum number of epochs
        loss_weights = None
        if scheme is None:
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {'up_rec_loss': 2.0, 'down_rec_loss': 2.0, 'up_kl_loss': 0.01, 'feat_l1_loss_up': 0.5,
                                'feat_l1_loss_down': 1.0, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {'up_rec_loss': 1.0, 'down_rec_loss': 1.0, 'up_kl_loss': 0.10, 'feat_l1_loss_up': 0.5,
                                'feat_l1_loss_down': 0.5, 'dpd_loss': 0.0, 'fide_kl_loss': 0.01, 'causal_loss': 0.01}
            elif current_epoch < max_epochs * 0.7:
                # Third quarter of training: emphasize KL loss
                loss_weights = {'up_rec_loss': 0.5, 'down_rec_loss': 0.5, 'up_kl_loss': 0.10, 'feat_l1_loss_up': 0.1,
                                'feat_l1_loss_down': 0.1, 'dpd_loss': 2.0, 'fide_kl_loss': 0.01, 'causal_loss': 0.01}
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {'up_rec_loss': 0.2, 'down_rec_loss': 0.2, 'up_kl_loss': 0.01, 'feat_l1_loss_up': 0.01,
                                'feat_l1_loss_down': 0.01, 'dpd_loss': 2.0, 'fide_kl_loss': 2.0, 'causal_loss': 2.0}
        elif scheme == 'sc':
            if current_epoch < max_epochs * 0.10:
                # First quarter of training: emphasize reconstruction loss
                loss_weights = {'sub_rec_loss': 2.0, 'glo_rec_loss': 1.0, 'sub_kl_loss': 0.01, 'glo_kl_loss': 0.01,
                                'feat_l1_loss': 0.01, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            elif current_epoch < max_epochs * 0.40:
                # Second quarter of training: transition from reconstruction to KL loss
                loss_weights = {'sub_rec_loss': 1.0, 'glo_rec_loss': 0.5, 'sub_kl_loss': 0.2, 'glo_kl_loss': 0.1,
                                'feat_l1_loss': 0.00, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            elif current_epoch < max_epochs * 0.80:
                # Third quarter of training: emphasize KL loss
                loss_weights = {'sub_rec_loss': 0.5, 'glo_rec_loss': 0.25, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                                'feat_l1_loss': 0.00, 'dpd_loss': 2.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
            else:
                # Fourth quarter of training: transition from KL to causal loss
                loss_weights = {'sub_rec_loss': 0.2, 'glo_rec_loss': 0.1, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                                'feat_l1_loss': 0.00, 'dpd_loss': 2.0, 'fide_kl_loss': 1.0, 'causal_loss': 1.0}
        return loss_weights


# FVAE model
class FVAE(nn.Module):
    """
    Fractal Variational Autoencoder for Feature Selection.
    """

    def __init__(
            self,
            n_input: int,
            n_hidden: int = 128,
            n_latent: int = 20,
            n_causal: int = 5,
            n_controls: int = 10,
            scale: float = 1.0,
            n_layers_encoder: int = 1,
            n_layers_decoder: int = 1,
            n_layers_dpd: int = 1,
            dropout_rate_encoder: float = 0.1,
            dropout_rate_decoder: float = 0.0,
            dropout_rate_dpd: float = 0.1,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
            use_batch_norm_dpd: bool = True,
            dpd_linear: bool = True,
    ):
        super(FVAE, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_causal = n_causal
        self.n_controls = n_controls
        # self.warm_up = 0
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.feature_selector = FeatureSelector(self.n_input, n_controls=self.n_controls)
        self.encoder = Encoder(
            self.n_input,
            self.n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder = Decoder(
            self.n_latent,
            self.n_input,
            n_layers=n_layers_decoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )
        self.dpd_model = DynamicPhenotypeDescriptor(
            self.n_input,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_dpd,
            n_layers=n_layers_dpd,
            batch_norm=use_batch_norm_dpd,
            linear=dpd_linear,
        )

    def forward(self, x):
        """
        Forward pass through the whole network.
        """
        x_top, _ = self.feature_selector(x, keep_top=True, keep_not_top=False)
        latent_top = self.encoder(x_top)
        x_top_rec = self.decoder(latent_top['z'])
        x_down, _ = self.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
        org_dpd, org_prob = self.dpd_model(x_down)

        alpha_z = torch.zeros_like(latent_top['z'])
        alpha_z[:, :self.n_causal] = latent_top['z'][:, :self.n_causal]
        alpha_z[:, self.n_causal:] = latent_top['z'][:, self.n_causal:].mean(dim=0, keepdim=True)
        alpha_x_rec = self.decoder(alpha_z)
        alpha_x_down, _ = self.feature_selector(alpha_x_rec, keep_top=False, keep_not_top=True)
        alpha_dpd, alpha_prob = self.dpd_model(alpha_x_down)

        x_all, feat_w = self.feature_selector(x, keep_top=True, keep_not_top=True)
        # x_all, feat_w = self.feature_selector(x, keep_top=True, keep_not_top=False)
        latent_all = self.encoder(x_all * self.scale)
        x_all_rec = self.decoder(latent_all['z'])
        # print("scale", self.scale)
        return dict(
            latent_top=latent_top,
            latent_all=latent_all,
            x_top_rec=x_top_rec,
            x_all_rec=x_all_rec,
            feat_w=feat_w,
            org_dpd=org_dpd,
            org_prob=org_prob,
            alpha_dpd=alpha_dpd,
            alpha_prob=alpha_prob,
        )

    @staticmethod
    def compute_loss(model_outputs, x, y):
        # get model
        latent_top, latent_all, x_top_rec, x_all_rec, feat_w, org_dpd, org_prb, _, alpha_prob = model_outputs.values()
        qz_m_top, qz_v_top = latent_top['qz_m'], latent_top['qz_v']
        qz_m_all, qz_v_all = latent_all['qz_m'], latent_all['qz_v']

        # top feature reconstruction loss
        top_rec_loss = F.mse_loss(x_top_rec, x, reduction='none').mean(dim=1)

        # all feature reconstruction loss
        all_rec_loss = F.mse_loss(x_all_rec, x, reduction='none').mean(dim=1)

        # top latent kl divergence loss
        top_kl_loss = kl(Normal(qz_m_top, torch.sqrt(qz_v_top)), Normal(0, 1)).sum(dim=1)
        # all latent kl divergence loss
        all_kl_loss = kl(Normal(qz_m_all, torch.sqrt(qz_v_all)), Normal(0, 1)).sum(dim=1)

        # feature weight l1 loss
        feat_l1_loss = torch.sum(torch.abs(feat_w))

        # DPD binary classification loss
        dpd_loss = F.binary_cross_entropy(org_prb.squeeze(), y, reduction='none')
        # dpd_loss = F.binary_cross_entropy_with_logits(org_dpd.squeeze(), y, reduction='none')

        # fidelity kl divergence loss
        alpha_probs = torch.cat((alpha_prob, 1 - alpha_prob), dim=1)
        org_probs = torch.cat((org_prb, 1 - org_prb), dim=1)
        fide_kl_loss = F.kl_div(torch.log(alpha_probs + 1e-8), org_probs, reduction='none').sum(dim=1)

        # Save each loss to the dictionary to return
        loss_dict = dict(
            sub_rec_loss=top_rec_loss,
            glo_rec_loss=all_rec_loss,
            sub_kl_loss=top_kl_loss,
            glo_kl_loss=all_kl_loss,
            feat_l1_loss=feat_l1_loss,
            dpd_loss=dpd_loss,
            fide_kl_loss=fide_kl_loss,
        )

        return loss_dict

    @staticmethod
    def update_loss_weights(current_epoch, max_epochs, loss_weights):
        # Update loss weights based on current epoch and maximum number of epochs
        if current_epoch < max_epochs * 0.10:
            # First quarter of training: emphasize reconstruction loss
            loss_weights = {'sub_rec_loss': 2.0, 'glo_rec_loss': 1.0, 'sub_kl_loss': 0.01, 'glo_kl_loss': 0.01,
                            'feat_l1_loss': 1.0, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        elif current_epoch < max_epochs * 0.40:
            # Second quarter of training: transition from reconstruction to KL loss
            loss_weights = {'sub_rec_loss': 1.0, 'glo_rec_loss': 0.5, 'sub_kl_loss': 0.2, 'glo_kl_loss': 0.1,
                            'feat_l1_loss': 0.5, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        elif current_epoch < max_epochs * 0.70:
            # Third quarter of training: emphasize KL loss
            oss_weights = {'sub_rec_loss': 0.5, 'glo_rec_loss': 0.25, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                           'feat_l1_loss': 0.1, 'dpd_loss': 2.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        else:
            # Fourth quarter of training: transition from KL to causal loss
            loss_weights = {'sub_rec_loss': 0.2, 'glo_rec_loss': 0.1, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                            'feat_l1_loss': 0.01, 'dpd_loss': 2.0, 'fide_kl_loss': 2.0, 'causal_loss': 2.0}
        return loss_weights

    @staticmethod
    def update_loss_weights_sc(current_epoch, max_epochs, loss_weights):
        # Update loss weights based on current epoch and maximum number of epochs
        if current_epoch < max_epochs * 0.10:
            # First quarter of training: emphasize reconstruction loss
            loss_weights = {'sub_rec_loss': 2.0, 'glo_rec_loss': 1.0, 'sub_kl_loss': 0.01, 'glo_kl_loss': 0.01,
                            'feat_l1_loss': 0.01, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        elif current_epoch < max_epochs * 0.40:
            # Second quarter of training: transition from reconstruction to KL loss
            loss_weights = {'sub_rec_loss': 1.0, 'glo_rec_loss': 0.5, 'sub_kl_loss': 0.2, 'glo_kl_loss': 0.1,
                            'feat_l1_loss': 0.00, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        elif current_epoch < max_epochs * 0.70:
            # Third quarter of training: emphasize KL loss
            oss_weights = {'sub_rec_loss': 0.5, 'glo_rec_loss': 0.25, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                           'feat_l1_loss': 0.00, 'dpd_loss': 2.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        else:
            # Fourth quarter of training: transition from KL to causal loss
            loss_weights = {'sub_rec_loss': 0.2, 'glo_rec_loss': 0.1, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.05,
                            'feat_l1_loss': 0.00, 'dpd_loss': 2.0, 'fide_kl_loss': 1.0, 'causal_loss': 1.0}
        return loss_weights

    @staticmethod
    def update_loss_weights_sub(current_epoch, max_epochs, loss_weights):
        # Update loss weights based on current epoch and maximum number of epochs
        if current_epoch < max_epochs * 0.10:
            # First quarter of training: emphasize reconstruction loss
            loss_weights = {'sub_rec_loss': 2.0, 'glo_rec_loss': 0.0, 'sub_kl_loss': 0.01, 'glo_kl_loss': 0.00,
                            'feat_l1_loss': 1.0, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        elif current_epoch < max_epochs * 0.40:
            # Second quarter of training: transition from reconstruction to KL loss
            loss_weights = {'sub_rec_loss': 1.0, 'glo_rec_loss': 0.0, 'sub_kl_loss': 0.2, 'glo_kl_loss': 0.0,
                            'feat_l1_loss': 0.5, 'dpd_loss': 0.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        elif current_epoch < max_epochs * 0.70:
            # Third quarter of training: emphasize KL loss
            oss_weights = {'sub_rec_loss': 0.5, 'glo_rec_loss': 0.0, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.00,
                           'feat_l1_loss': 0.1, 'dpd_loss': 2.0, 'fide_kl_loss': 0.0, 'causal_loss': 0.0}
        else:
            # Fourth quarter of training: transition from KL to causal loss
            loss_weights = {'sub_rec_loss': 0.2, 'glo_rec_loss': 0.0, 'sub_kl_loss': 0.10, 'glo_kl_loss': 0.00,
                            'feat_l1_loss': 0.01, 'dpd_loss': 2.0, 'fide_kl_loss': 1.0, 'causal_loss': 1.0}
        return loss_weights
