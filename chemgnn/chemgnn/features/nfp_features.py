# chemgnn/features/nfp_features.py
from collections import namedtuple
from nfp.preprocessing.features import get_ring_size

atom_type = namedtuple(
    "Atom", ["symbol", "aromatic", "ring_size", "degree", "totalHs"]
)
bond_type = namedtuple(
    "Bond", ["bond_type", "degree", "ring_size"]
)

def atom_featurizer(atom):
    return atom_type(
        atom.GetSymbol(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True),
    )

def bond_featurizer(bond, flipped=False):
    if not flipped:
        atoms = f"{bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()}"
    else:
        atoms = f"{bond.GetEndAtom().GetSymbol()}-{bond.GetBeginAtom().GetSymbol()}"

    btype = str(bond.GetBondType())
    ring = get_ring_size(bond, max_size=6) if bond.IsInRing() else None
    return bond_type(atoms, btype, ring)
