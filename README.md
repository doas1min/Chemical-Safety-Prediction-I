# Integrated GNN Framework for Chemical Safety Property Prediction

This repository contains the code and data used in the paper:
**"Advancing Chemical Safety Prediction: An Integrated GNN Framework with DFT-augmented Cyclic Compound Solution" (2025)**

We provide:
- GNN (Message Passing Neural Network) implementation for predicting HoC, VP, and Flashpoint
- Random Forest model specialized for cyclic HoC compounds (with DFT-augmented data)
- Complete datasets for all experiments
- Scripts to reproduce the main results of the paper

---

## Repository Structure

```
data/ # CSV datasets for HoC, VP, Flashpoint
src/ # GNN and Random Forest implementation
README.md
LICENSE
```

---

## Dataset Description

All datasets used in the paper are included in the `data/` directory.

- **`HoC.csv`** — Full experimental Heat of Combustion dataset
- **`HoC_no_cyclic.csv`** — HoC dataset excluding the 12 cyclic compounds
- **`HoC_cyclic_DFT_aug.csv`** — Cyclic-only HoC dataset (12 experimental + 43 DFT-augmented = 55 total)
- **`VP.csv`** — Vapor Pressure dataset
- **`Flashpoint.csv`** — Flashpoint dataset

---

## Code Structure (`src/` directory)

The `src/` directory contains two types of modeling scripts:  
(1) GNN (MPNN) models and  
(2) Random Forest model for cyclic HoC.

### 🔹 GNN scripts (shared across HoC / VP / Flashpoint)

All three properties (HoC, VP, Flashpoint) use the **same GNN architecture and training pipeline**.  
Thus, we provide a single representative notebook:

- `GNN_model.ipynb` — Shared MPNN implementation used for HoC / VP / Flashpoint  
  (only the input dataset changes)

This notebook includes:
- RDKit graph construction  
- Message passing and global state module  
- Training / validation / test pipeline  
- Hyperparameter tuning  
- Prediction and visualization

**Note:**  
The GNN pipeline is fully reusable.  
To apply the model to a new dataset, simply:
- change the data file name/path, and  
- ensure key column names (e.g., `SMILES`, `target`) match the expected format.

No other code modifications are required.

### 🔹 Random Forest script (for cyclic HoC)
- `RF_cyclic_HoC.ipynb`  
  Random Forest regression for cyclic HoC compounds  
  (trained on `HoC_cyclic_DFT_aug.csv`)

This script includes:
- Descriptor generation (MolWt, TPSA, HBA/HBD, Kappa, etc.)
- Grid search and cross-validation  
- Model evaluation

---

## Reproducibility Notes

- All CSV files are included under `data/`.
- HoC / VP / Flashpoint share the same GNN architecture; only the dataset differs.
- Random seeds are fixed, but results may vary slightly depending on environment or library versions.

---

## License
This project is licensed under the MIT License.  
See the **[LICENSE](LICENSE)** file for details.
