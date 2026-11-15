# MC-QRA: Quantitative Spatiotemporal Risk Assessment and Probabilistic Dependency Analysis of Maritime Cyber Attacks

<p align="center">
    <img src="https://img.shields.io/static/v1?label=Python&message=3.11&color=e83e8c"/>
</p>

Code for [MC-QRA: Quantitative Spatiotemporal Risk Assessment and Probabilistic Dependency Analysis of Maritime Cyber Attacks
](https://www.sciencedirect.com/science/article/pii/S095183202601625X)


## Abstract
The increasing digitalisation of maritime systems has raised concerns about cyber threats to shipping operations, port infrastructure and maritime safety. However, existing approaches often examine cyber-risk scenarios, geographical patterns and dependencies among risk factors separately. This study proposes Maritime Cyber-Quantitative Risk Assessment (MC-QRA), an integrated framework for scenario-based relative risk assessment, spatiotemporal analysis and probabilistic dependency modelling using reported maritime cyber incidents. The framework comprises four stages: latent class analysis (LCA) for scenario identification; frequency–severity scoring for relative risk assessment; kernel density estimation (KDE) and Getis–Ord Gi* analysis for spatiotemporal assessment; and mutual information (MI) and Tree-Augmented Naive Bayes (TAN) modelling for dependency analysis. The framework was applied to 444 incidents from the Maritime Cyber Attack Database (MCAD), identifying five distinct cyberattack scenarios. The results show an increase in reported incidents classified as medium or high risk after 2021, with risk-weighted spatial concentrations in Europe, North America and East Asia. Scenario-level risk decomposition indicates that ransomware-related incidents account for 52.2% of the summed relative risk scores, compared with 19.6% for spoofing-related incidents. Cyber threat type, exploited asset and target category provide the most information for distinguishing the assigned risk levels. The TAN model achieved a mean classification accuracy of 87.4% under stratified fivefold cross-validation, while identifying conditional probabilistic dependencies among incident characteristics. The resulting risk indices and spatial patterns describe relative differences within the reported MCAD sample rather than exposure-adjusted absolute risk. MC-QRA thus provides an integrated basis for comparing incident scenarios, examining spatiotemporal patterns and supporting preliminary maritime cyber-risk prioritisation.


## Quick Start
1. Install dependencies:
```shell
pip install -r requirements.txt
```

2. Run the analysis:
```shell
python main.py
```

### Cite MC-QRA
If you find this repo useful, please consider citing paper

```
@article{gong2026mc,
  title={MC-QRA: Quantitative Spatiotemporal Risk Assessment and Probabilistic Dependency Analysis of Maritime Cyber Attacks},
  author={Gong, Jincheng and Zhao, Yunfeng and Li, Huanhuan and Yang, Zaili},
  journal={Reliability Engineering \& System Safety},
  pages={113823},
  year={2026},
  publisher={Elsevier}
}
```
