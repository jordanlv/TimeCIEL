# TimeCIEL ⏱️☁️

## Overview 🌍

TimeCIEL (Contextual Interactive Ensemble Learning) is a multiagent ensemble learning system designed for multivariate time series supervised learning tasks. It leverages multiple learning agents that collaborate to solve supervised learning tasks.

### Prerequisites

- **Python 3.x** installed on your machine
- **pip** (Python package installer)

## Installation 💾

To install the dependencies of the project:

```bash
pip install -r requirements.txt
```

To install the library:

```bash
pip install git+https://github.com/jordanlv/TimeCIEL.git
```

## Repository 🗂️

The repository is organized as follows:

```
.
├── benchmark/      # Codes of the benchmark
│   ├── <method>.py     # Benchmark for a specific method
│   └── <result_method>.txt     # Results of a benchmark
│
├── explainability/       # Explainability shows in the paper
│   ├── global.ipynb      # Global explainability
│   └── local.ipynb       # Local explainability
│
└── torch_mas/      # Core implementation of the multi-agent algorithms
    ├── batch/      # Implementation of batch mode
    │   ├── activation_function/        # Implementations of various activation functions
    │   │   └── <activation>.py     # Code for specific activation functions
    │   │
    │   ├── internal_model/     # Implementations of internal models
    │   │   └── <model>.py      # Code for specific types of internal models
    │   │
    │   └── trainer/        # Implementation of various trainer
    │       ├── <trainer>.py        # Code for specific trainers
    │       └── learning_rules.py       # Definitions of learning rules for trainers
    │
    ├── common/     # Utilities shared between batch and sequential modes
    │   ├── models/     # Utilities for machine learning models
    │   │   └── <model_utilities>.py        # Code for model utility functions, layers, etc.
    │   │
    │   └── orthotopes/     # Utilities for orthotope (n-dimensional rectangle) manipulation
    │       └── <orthotope_utilities>.py        # Code for orthotope operations and utilities
    │
    └── sequential/     # Implementation of sequential mode
        ├── activation_function/        # Implementations of various activation functions
        │   └── <activation>.py     # Code for specific activation functions
        │
        ├── internal_model/     # Implementations of internal models
        │   └── <model>.py      # Code for specific types of internal models
        │
        └── trainer/        # Implementation of various trainer
            └── <trainer>.py        # Code for specific trainers
```