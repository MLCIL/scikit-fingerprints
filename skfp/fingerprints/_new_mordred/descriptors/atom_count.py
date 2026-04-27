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
    "nH": 1,
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


def calc(mol_with_hydrogens: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute the remaining Mordred atom count descriptors.

    Element counts use the explicit-hydrogen molecule to match Mordred
    `AtomCount` semantics for `nH`.
    """
    atomic_nums = [atom.GetAtomicNum() for atom in mol_with_hydrogens.GetAtoms()]
    values = [
        sum(atomic_num == _ELEMENTS[name] for atomic_num in atomic_nums)
        for name in FEATURE_NAMES[:-1]
    ]
    values.append(sum(atomic_num in _HALOGENS for atomic_num in atomic_nums))

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
