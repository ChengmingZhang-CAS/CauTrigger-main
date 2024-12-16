# CauTrigger

![CauTrigger logo](docs/logo.png)

Deciphering and manipulating cell state transitions remain fundamental challenges in biology, as transcription factors (TFs) orchestrate these processes by regulating downstream effectors through hierarchical interactions. However, existing methods often struggle to accurately identify causal drivers of state transitions and predict transcriptional responses to their perturbations. To address these challenges, we developed CauTrigger, a deep learning framework that uses causal information flow to identify TFs driving state transitions from gene expression profiles spanning two states. By incorporating hierarchical regulatory relationships, CauTrigger disentangles causal drivers from spurious associations and offers mechanistic insights into regulatory cascades. Additionally, it predicts transcriptional responses to single or combinatorial TF perturbations, enabling a deeper understanding of regulatory mechanisms. Benchmarking on diverse datasets demonstrates that CauTrigger outperforms traditional methods in driver TF identification and perturbation outcome prediction. Applied to single-cell and spatial transcriptomics datasets, CauTrigger not only validated known drivers but also uncovered novel insights through TF simulations, offering a versatile tool for investigating the causal mechanisms underlying cell state transitions.

![CauTrigger Overview](docs/CauTrigger_overview.png)

**Overview of the CauTrigger framework:**
a, Cell state transitions are regulated by hierarchical processes, where upstream causal factors (“triggers”) influence downstream regulatory factors (“conductors”), driving state changes. This hierarchical regulation is analogous to firing a handgun, where the trigger initiates downstream components (e.g., the hammer) to execute the final state transition. b, Iterative identification of causal triggers at each regulatory layer. This process segments regulatory components into hierarchical layers, systematically revealing upstream factors driving transitions. c, Structural causal model showing the decomposition of original triggers (TFs) x^u={x^(u,c),x^(u,s) } into their latent representations z={z^c,z^s }, with x^d denoting conductors (TGs) and y denoting the state. d, Causal decoupling model constructed on a dual-flow variational autoencoder (DFVAE) framework to identify causal triggers influencing state transition. Triggers (x^u) are processed through a feature selection layer to separates causal triggers (x^(u,c)) and others (x^(u,s)), and then encoded them into latent space z consists of causal (z^c) and spurious (z^s) components. This latent space is decoded to generate downstream conductors (x^d) and to predict the final cell state (y). The model strives to maximize the causal information flow, I(z^c→y), from z^c to y, thus delineating the causal path from x^(u,c) to y via z^c. e, Downstream applications of CauTrigger include in silico perturbation analysis, genetic interaction prediction, and vector field mapping for state transition simulation, enabling functional exploration of cellular processes.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ChengmingZhang-CAS/CauTrigger-main.git
   ```
   
2. Create and activate a Conda environment:
   ```bash
   conda env create -n ct_env -f environment.yml
   conda activate ct_env
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
