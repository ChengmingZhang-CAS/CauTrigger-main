# Tutorial

- Load package

  ```python
  from CauTrigger.utils import set_seed
  from CauTrigger.model import CauTrigger
  import anndata
  ```

- Prepare anndata

  ```python
  # read your expression profile and transform it to adata.
  adata = ...
  sc.pp.neighbors(adata)
  sc.tl.umap(adata)
  sc.pl.umap(adata, color='labels')  # cell cluster
  
  # read your prior grn
  grn = ...  # The first column is TF, second is target, third is regulatory direction(optional)
  grn_TF = ....  # The TF list
  grn_nonTF = ...  # The non-TF list
  
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

  



