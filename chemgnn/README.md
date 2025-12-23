# chemgnn

**chemgnn** is a reference implementation of an integrated chemical property
prediction framework combining traditional machine-learning models
(e.g., Random Forest) and graph neural networks (GNNs) for cyclic compounds.

This package accompanies the manuscript:

> *Advancing Chemical Safety Prediction: An Integrated GNN Framework with  
> DFT-augmented Cyclic Compound Solution*  
> submitted to *Journal of Cheminformatics*.

The original workflows were developed in Jupyter notebooks and have been
refactored into a reusable Python package to improve clarity,
reproducibility, and extensibility.

---

## Overview

The goal of this project is to provide a unified and reproducible framework
for predicting chemical properties of cyclic compounds using:

- **Descriptor-based models**  
  - RDKit molecular descriptors  
  - Random Forest regression

- **Graph-based models**  
  - Neural Fingerprint (NFP)-style GNNs  
  - TensorFlow / Keras backend

The package separates **data handling**, **feature generation**, **model
definitions**, **training**, and **evaluation**, enabling direct comparison
between classical ML and GNN-based approaches under a common structure.

---

## Package Structure

chemgnn/
├── chemgnn/
│   ├── __init__.py
│   │
│   ├── data.py
│   │   └── CSV loading and dataset splitting utilities
│   │
│   ├── features/
│   │   ├── rdkit_descriptors.py   # Descriptor computation for RF models
│   │   └── nfp_features.py        # Atom/bond featurizers for GNN models
│   │
│   ├── datasets/
│   │   ├── tabular.py             # Tabular dataset construction (RF)
│   │   └── graph_tf.py            # TensorFlow graph datasets (GNN)
│   │
│   ├── models/
│   │   ├── rf.py                  # Random Forest model definition
│   │   └── nfp_gnn.py              # NFP-based GNN architecture
│   │
│   ├── train/
│   │   ├── rf_train.py             # Training / CV / tuning for RF
│   │   └── gnn_train.py            # Training and evaluation for GNN
│   │
│   └── evaluate.py                # Common regression metrics (MAE, RMSE, R²)
│
└── examples/
    ├── RF_cyclic_HoC.ipynb         # Random Forest workflow example
    └── GNN_model.ipynb             # GNN workflow example


---

## Installation

This package is provided as a **reference implementation**.
Users are expected to configure their Python environment according to
their local system and requirements.

The code was primarily developed in Python 3.10 environments.
Some components (in particular, GNN-related dependencies) may require
adjustments to library versions depending on the local setup.

---

## Usage

### Random Forest models

A typical Random Forest workflow consists of:

1. Loading a CSV dataset
2. Computing RDKit descriptors
3. Splitting data into training and test sets
4. Training and evaluating a Random Forest regressor

A complete, reproducible example is provided in:

examples/RF_cyclic_HoC.ipynb


---

### Graph Neural Network models

The GNN workflow includes:

1. SMILES-based graph construction
2. Atom and bond featurization
3. TensorFlow dataset generation
4. Training and evaluation of an NFP-style GNN

A full example is provided in:

examples/GNN_model.ipynb


---

## Evaluation

Model performance is evaluated using standard regression metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Utility functions for consistent metric computation are provided in:

chemgnn/evaluate.py


---

## Notes on Reproducibility

- Dataset splits are controlled via explicit random seeds.
- Feature computation and model definitions are fully separated from
  notebook logic.
- The notebooks in `examples/` serve as **executable documentation**
  demonstrating end-to-end workflows.

---

## Citation

If you use this code in your research, please cite the corresponding
manuscript submitted to *Journal of Cheminformatics*.

---

## License

This project is intended for academic and research use.
Please refer to the repository for licensing details.

