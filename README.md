# TimeCIEL ⏱️☁️


[![HAL](https://img.shields.io/badge/HAL-Paper-blue.svg)](https://hal.science/hal-05053054v1/document)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

Official implementation of the paper: **"TimeCIEL: Contextual Interactive Ensemble Learning for Time Series Classification"** (Accepted at PAAMS 2025).

> **Authors:** Jordan Levy, Clément Blanco-Volle, Nicolas Verstaevel, Benoit Gaudou & Vincent Talon <br>
> **Institution:** IRIT, Université Toulouse Capitole

## Abstract

Multivariate time series classification is a challenging task where black box models achieve high performances. However, in real-world applications, interpretability is crucial for helping users understand the decision-making process of an algorithm, not just its performance. In this paper, we present a multi-agent ensemble learning approach for time series classification suited for online learning. Our approach relies on the organization of agents in the feature space at each time step. We demonstrate that our approach achieves performances comparable to state-of-the-art methods. Finally, we highlight its explainability and interpretability properties as a white-box model.

## Installation

To install the project dependencies:

```bash
pip install -r requirements.txt
```

To install the library directly from GitHub:
```bash
pip install git+[https://github.com/jordanlv/TimeCIEL.git](https://github.com/jordanlv/TimeCIEL.git)
```

## Citation
If you find this code useful, please cite our paper:
```
@inproceedings{levy2025timeciel,
  title={TimeCIEL: Contextual Interactive Ensemble Learning for Time Series Classification},
  author={Levy, Jordan and Blanco-Volle, Cl{\'e}ment and Verstaevel, Nicolas and Gaudou, Benoit and Talon, Vincent},
  booktitle={International Conference on Practical Applications of Agents and Multi-Agent Systems},
  pages={316--327},
  year={2025},
  organization={Springer}
}
```
