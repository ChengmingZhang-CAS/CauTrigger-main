# CauTrigger (v0.0.1)

<div align="center">
  <img src="docs/logo.png" alt="CauTrigger logo" width="300" />
</div>

Deciphering and manipulating cell state transitions remain fundamental challenges in biology, as transcription factors (TFs) orchestrate these processes by regulating downstream effectors through hierarchical interactions. CauTrigger, a deep learning framework that uses causal information flow to identify TFs driving state transitions from gene expression profiles spanning two states. By incorporating hierarchical regulatory relationships, CauTrigger disentangles causal drivers from spurious associations and offers mechanistic insights into regulatory cascades. 

![CauTrigger Overview](docs/CauTrigger_overview.png)

Causal decoupling model constructed on a dual-flow variational autoencoder (DFVAE) framework to identify causal triggers influencing state transition. Triggers ($x^u$) are processed through a feature selection layer to separates causal triggers ($x^{u,c}$) and others ($x^{u,s}$), and then encoded them into latent space $z$ consists of causal ($z^c$) and spurious ($z^s$) components. This latent space is decoded to generate downstream conductors ($x^d$) and to predict the final cell state ($y$). The model strives to maximize the causal information flow, $I(z^c→y)$, from $z^c$ to $y$, thus delineating the causal path from $x^{u,c}$ to $y$ via $z^c$.


## Installation

```bash
git clone git@github.com:ChengmingZhang-CAS/CauTrigger-main.git
cd CauTrigger-main
conda create -n CauTrigger python==3.10
conda activate CauTrigger
pip install -r requirements.txt

# install torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

python setup.py install
```


## Tutorials
We provide two simulated toy AnnData objects in docs/toy_adata_(non)linear.h5ad, for simple tests of causal decomposition and trigger identification.
```python
import pandas as pd
import scanpy as sc
from CauTrigger.utils import set_seed
from CauTrigger.model import CauTrigger
set_seed(42)

adata1 = sc.read_h5ad('toy_adata_linear.h5ad')
# adata1
# AnnData object with n_obs × n_vars = 200 × 100
#     obs: 'labels'
#     var: 'feat_type', 'feat_label'
#     obsm: 'X_down'


model = CauTrigger(
    adata1,
    n_causal=2,  # causal latent dim
    n_latent=10,  # overall latent dim
    n_hidden=128,
    n_layers_encoder=0,
    n_layers_decoder=0,
    n_layers_dpd=0,
    dropout_rate_encoder=0.0,
    dropout_rate_decoder=0.0,
    dropout_rate_dpd=0.0,
    use_batch_norm='none',
    use_batch_norm_dpd=True,
    decoder_linear=True,
    dpd_linear=False,
    init_weight=init_weight,
    init_thresh=0.0,
    update_down_weight=False,
    attention=False,
    att_mean=False,
)
model.train(max_epochs=300, stage_training=True)
weight_df_weight = model.get_up_feature_weights(normalize=True, method="Model", sort_by_weight=False)
weight_df = pd.DataFrame({'weight_value': weight_df_weight[0]['weight'],})

# Compare the Causal information flow across different dimensions of the latent space
info_flow, info_flow_cat = model.compute_information_flow()
```

The following code blocks provide a simple tutorial for applying causal decomposition and performing downstream tasks on real-world datasets.
- Load package

  ```python
  from CauTrigger.utils import set_seed
  from CauTrigger.model import CauTrigger
  import anndata
  ```

- Prepare anndata

  ```python
  # read your expression profile and transform it to adata
  adata = ...
  
  # read your prior grn
  grn = ...
  grn_TF = ....
  grn_nonTF = ...
  
  # make adata for CauTrigger
  start_TF = adata[(adata.obs['state'] == 0), np.intersect1d(adata.var_names, grn_TF)]
  end_TF = adata[(adata.obs['state'] == 1), np.intersect1d(adata.var_names, grn_TF)]
  start_down = adata[(adata.obs['state'] == 0), np.intersect1d(adata.var_names, grn_nonTF)]
  end_down = adata[(adata.obs['state'] == 1), np.intersect1d(adata.var_names, grn_nonTF)]
  adata_ct = anndata.concat([start_TF.copy(), end_TF.copy()])
  adata_ct.obs['labels'] = np.repeat([0, 1], [start_TF.shape[0], end_TF.shape[0]])
  adata_ct.obsm['X_down'] = anndata.concat([start_down, end_down]).X.copy()
  ```

- Train model

  ```python
  set_seed(42)
  model = CauTrigger(adata_ct)
  model.train(max_epochs=200)
  ```

- Get the causal triggers

  ```python
  weight_df = model.get_up_feature_weights(normalize=True, method="Model", sort_by_weight=False)[0]
  model_res = pd.DataFrame({'weight_value': weight_df['weight'], }).sort_values('weight_value', ascending=False)
  print(model_res)
  ```
- In silico perturb and draw vector field map (borrowed from CellOracle)

  ```Python
  from CauTrigger.utils import pert_plot_up, pert_plot_down
  
  adata_TF = adata[:, adata.var_names.isin(grn_TF)].copy()
  adata_down = adata[:, adata.var_names.isin(grn_TF)].copy()
  state_pair = (0, 1)
  state_obs = 'labels'
  dot_size = 200
  
  # Single or multi genes to perturb 
  KO_Gene = ...  # e.g. ['Gene'] or ['Gene1', 'Gene2',...]
  
  # First to find the optimal 'min_mass' value
  pert_plot_up(adata_TF, adata_down, model, state_pair, KO_Gene, state_obs,run_suggest_mass_thresholds=True,dot_size=dot_size, scale=0.1, min_mass=0.008,embedding_name='X_umap')
  
  # Then draw the vector filed map
  pert_plot_up(adata_TF, adata_down, model, state_pair, KO_Gene, state_obs,run_suggest_mass_thresholds=False,fold=[40,50],dot_size=dot_size, , scale=8, min_mass=2,save_dir=output_path,embedding_name='X_umap')
  ```

- Calculate genetic interaction

  ```python
  from CauTrigger.utils import calculate_GIs
  from itertools import combinations
  
  # gene list to calculate GIs
  genes = ...  # e.g. model_res[0:10].index
  genescombs = list(combinations(genes, 2))
  # synergy_score_down = pd.DataFrame({'MAG':0},index=['+'.join(i) for i in genescombs])
  synergy_score_up = pd.DataFrame({'MAG':0},index=['+'.join(i) for i in genescombs])
  
  for id, pair in enumerate(genescombs):
      adata_pert = adata_TF.copy()
      adata_pert = adata_pert[adata_pert.obs[state_obs] == 0]
      adata_pert.X[:, adata_pert.var_names.get_loc(pair[0])] = 2 * adata_TF.X.max()
      model.eval()
      with torch.no_grad():
          model_output_pertA =model.module.forward(torch.Tensor(adata_pert.X).to('cuda:0'))
      adata_pert = adata_TF.copy()
      adata_pert = adata_pert[adata_pert.obs[state_obs] == 0]
      adata_pert.X[:, adata_pert.var_names.get_loc(pair[1])] = 2 * adata_TF.X.max()
      model.eval()
      with torch.no_grad():
          model_output_pertB = model.module.forward(torch.Tensor(adata_pert.X).to('cuda:0'))
      adata_pert = adata_TF.copy()
      adata_pert = adata_pert[adata_pert.obs[state_obs] == 0]
      adata_pert.X[:, adata_pert.var_names.get_loc(pair[0])] = 2 * adata_TF.X.max()
      adata_pert.X[:, adata_pert.var_names.get_loc(pair[1])] = 2 * adata_TF.X.max()
      model.eval()
      with torch.no_grad():
          model_output_pertAB = model.module.forward(torch.Tensor(adata_pert.X).to('cuda:0'))
      ctrl = np.squeeze(np.concatenate([adata_TF.X, adata_down.X], axis=1).mean(0))
      predA = np.squeeze(np.concatenate([model_output_pertA['x_up_rec1'].cpu().numpy(), model_output_pertA['x_down_rec_alpha'].cpu().numpy()], axis=1).mean(0))
      predB = np.squeeze(np.concatenate([model_output_pertB['x_up_rec1'].cpu().numpy(), model_output_pertB['x_down_rec_alpha'].cpu().numpy()], axis=1).mean(0))
      predAB = np.squeeze(np.concatenate([model_output_pertAB['x_up_rec1'].cpu().numpy(), model_output_pertAB['x_down_rec_alpha'].cpu().numpy()], axis=1).mean(0))
      results = calculate_GIs(predA - ctrl, predB - ctrl, predAB - ctrl)
      synergy_score_up.iloc[id, 0] = results['mag']
  
  synergy_score_up.sort_values(by='MAG', ascending=False)
  ```

  Similarly, you can perform a combined simulation perturbation after this step.

- Draw regulon

  ```python
  from CauTrigger.utils import draw_regulon
  
  RGMs = {}
  for i in adata.var_names[adata.var_names.isin(grn_TF)]:
      df = grn[grn[0] == i].iloc[:, 0:3]
      RGMs[i] = df[df[1].isin(adata.var_names[adata.var_names.isin(grn_nonTF)])]
      
  regulators = ...
  for regulator in regulators:
      draw_regulon(regulator, RGMs, output_path=os.path.join(output_path, f'regulons'))
  ```

  
For detailed documentation and tutorials, please visit our official CauTrigger tutorial website:

[CauTrigger Tutorials](https://cautrigger-tutorials.readthedocs.io/en/latest/)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use CauTrigger in your research, please cite the following paper:



## Contact

For questions or issues, please contact Chengming Zhang at zhangchengming@g.ecc.u-tokyo.ac.jp.
