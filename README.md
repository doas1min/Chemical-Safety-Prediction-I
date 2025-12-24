# chemgnn

**chemgnn** is a reference implementation of an integrated chemical property
prediction framework combining traditional machine-learning models
(e.g., Random Forest) and graph neural networks (GNNs) for cyclic compounds.

This repository accompanies the manuscript:

> *Advancing Chemical Safety Prediction: An Integrated GNN Framework with  
> DFT-augmented Cyclic Compound Solution*  
> submitted to *Journal of Cheminformatics*.

The original workflows were developed in Jupyter notebooks and have been
refactored into a structured Python codebase to improve clarity,
reproducibility, and reuse.

---

## Overview

This project provides a unified workflow for predicting chemical properties
of cyclic compounds using two complementary approaches:

- **Descriptor-based models**
  - RDKit molecular descriptors
  - Random Forest regression

- **Graph-based models**
  - Neural Fingerprint (NFP)-style graph neural networks
  - TensorFlow / Keras backend

Data handling, feature generation, model definitions, training, and evaluation
are separated into modular components to enable transparent comparison
between classical machine-learning and GNN-based methods.

---

## Installation

This codebase can be installed locally in **editable mode** to enable
direct imports (e.g., `import chemgnn`) from scripts or Jupyter notebooks.

From the project root directory:

```bash
pip install -e .
```

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
4. Training and evaluation of a message-passing graph neural network (GNN)

A full example is provided in:

examples/GNN_model.ipynb

This example uses the HoC dataset, but the same GNN pipeline applies to Vapor Pressure and Flashpoint.

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

This project is licensed under the MIT License.  
See the **[LICENSE](LICENSE)** file for details.
