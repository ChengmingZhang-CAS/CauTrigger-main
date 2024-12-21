# Tutorial

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

  



