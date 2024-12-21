# CauTrigger

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

For detailed documentation and tutorials, please visit our official CauTrigger tutorial website:

[CauTrigger Tutorials](https://caufinder-tutorials.readthedocs.io/en/latest/index.html)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Citation

If you use CauTrigger in your research, please cite the following paper:



## Contact

For questions or issues, please contact Chengming Zhang at zhangchengming@g.ecc.u-tokyo.ac.jp.
