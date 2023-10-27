"""Constants for ChemicalX."""

from torchdrug.data import Molecule
from torchdrug.data.feature import atom_default

__all__ = [
    "TORCHDRUG_NODE_FEATURES",
]

#: The default number of node features on a molecule in torchdrug
# torchdrug < 1.3.* 
#TORCHDRUG_NODE_FEATURES = len(atom_default(Molecule.dummy_atom))
# torchdrug >= 1.3.* 
# test
TORCHDRUG_NODE_FEATURES = len(atom_default(Molecule.dummy_mol.GetAtomWithIdx(0)))