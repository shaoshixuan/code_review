# CMB
> Existing recommender systems tend to prioritize items closely aligned with users' historical interactions, inevitably trapping users in the dilemma of ``filter bubble''. Recent efforts are dedicated to improving the diversity of recommendations. However, they mainly suffer from two major issues: 1) a lack of explainability, making it difficult for the system designers to understand how diverse recommendations are generated, and 2) limitations to specific metrics, with difficulty in enhancing non-differentiable diversity metrics. To this end, we propose a \textbf{C}ounterfactual \textbf{M}ulti-player \textbf{B}andits (CMB) method to deliver explainable recommendation diversification across a wide range of diversity metrics. Leveraging a counterfactual framework, our method identifies the factors influencing diversity outcomes. Meanwhile, we adopt the multi-player bandits to optimize the counterfactual optimization objective, making it adaptable to both differentiable and non-differentiable diversity metrics. Extensive experiments conducted on three real-world datasets demonstrate the applicability, effectiveness, and explainability of the proposed CMB.

## Overall
Pytorch implementation for paper "Counterfactual Multi-player Bandits for Explainable Recommendation Diversification" published on ECML PKDD 2025.

## Requirements
- Python 3.9
- pytorch 1.13.0
- cuda 11

## Instruction
1. You may download Amazon Review dataset from https://nijianmo.github.io/amazon/index.html.

2. We provide an example on "CDs and Vinyl" datasets. The pre-processing data within subtopic information are already in the "dataset/CDs" folder.

3. To set the python path, under the project root folder, run:
    ```
    source setup.sh
    ```
4. To train the base recommender: run:
    ```
    python scripts/train_base_amazon.py
    ```
5. To run cmb method, run:
    ```
    python scripts/generate_div_amazon.py
    ```

## Citation
Please cite our paper if you think this code is useful:
```latex
@article{zhang2025cmb,
  title={Shapley Value-driven Data Pruning for Recommender Systems},
  author={Yansen Zhang and Bowei He and Xiaokun Zhang and Haolun Wu and Zexu Sun and Chen Ma},
  journal={ECML PKDD},
  year={2025}
}
```
