# chemgnn/features/rdkit_descriptors.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import (
    CalcTPSA,
    CalcExactMolWt,
    CalcNumHBA,
    CalcNumHBD,
)
from rdkit.Chem.GraphDescriptors import (
    BalabanJ,
    BertzCT,
    Chi1,
    Kappa1,
)

# --------------------------------------------------
# Core feature computation
# --------------------------------------------------

def compute_rdkit_descriptors(
    df,
    smiles_col="SMILES",
    drop_invalid=True,
):
    """
    Compute RDKit descriptors used for RF models.

    Returns
    -------
    DataFrame with descriptor columns appended.
    """

    mols = []
    for s in df[smiles_col]:
        mol = Chem.MolFromSmiles(str(s))
        mols.append(mol)

    df = df.copy()
    df["_Mol"] = mols

    if drop_invalid:
        df = df[df["_Mol"].notna()].reset_index(drop=True)

    df["MolWt"] = df["_Mol"].apply(
        lambda m: CalcExactMolWt(Chem.AddHs(m))
    )
    df["tpsa"] = df["_Mol"].apply(
        lambda m: CalcTPSA(Chem.AddHs(m))
    )
    df["HBA"] = df["_Mol"].apply(
        lambda m: CalcNumHBA(Chem.AddHs(m))
    )
    df["HBO"] = df["_Mol"].apply(
        lambda m: CalcNumHBD(Chem.AddHs(m))
    )
    df["Kappa"] = df["_Mol"].apply(
        lambda m: Kappa1(Chem.AddHs(m))
    )
    df["bertz_ct"] = df["_Mol"].apply(
        lambda m: BertzCT(Chem.AddHs(m))
    )
    df["chi1"] = df["_Mol"].apply(
        lambda m: Chi1(Chem.AddHs(m))
    )
    df["Balaban"] = df["_Mol"].apply(
        lambda m: BalabanJ(Chem.AddHs(m))
    )
    df["Os"] = df["_Mol"].apply(
        lambda m: sum(a.GetSymbol() == "O" for a in m.GetAtoms())
    )

    df = df.drop(columns=["_Mol"])

    return df


# --------------------------------------------------
# Helper
# --------------------------------------------------

def rf_feature_columns():
    """Return the canonical RF feature column names."""
    return [
        "MolWt",
        "tpsa",
        "HBA",
        "HBO",
        "Kappa",
        "bertz_ct",
        "chi1",
        "Balaban",
        "Os",
    ]
