"""Atom count descriptors implemented with direct RDKit atom access.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np
from rdkit.Chem import Mol

FEATURE_NAMES = [
    "nH",
    "nB",
    "nC",
    "nN",
    "nO",
    "nS",
    "nP",
    "nF",
    "nCl",
    "nBr",
    "nI",
    "nX",
]

_ELEMENTS = {
    "nB": 5,
    "nC": 6,
    "nN": 7,
    "nO": 8,
    "nS": 16,
    "nP": 15,
    "nF": 9,
    "nCl": 17,
    "nBr": 35,
    "nI": 53,
}
_HALOGENS = {9, 17, 35, 53}


def calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute the remaining Mordred atom count descriptors.

    Hydrogen counts use RDKit's total hydrogen count on each atom, so implicit
    hydrogens are included without adding explicit hydrogen atoms to the
    molecule. Heavy-element counts are taken from the molecule's atom list.
    """
    atoms = mol.GetAtoms()
    atomic_nums = [atom.GetAtomicNum() for atom in atoms]
    values = [sum(atom.GetTotalNumHs() for atom in atoms)]
    values.extend(
        sum(atomic_num == _ELEMENTS[name] for atomic_num in atomic_nums)
        for name in FEATURE_NAMES[1:-1]
    )
    values.append(sum(atomic_num in _HALOGENS for atomic_num in atomic_nums))

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
