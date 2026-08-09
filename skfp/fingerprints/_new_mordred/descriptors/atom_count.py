from collections import Counter

import numpy as np
from rdkit.Chem import Mol, rdMolDescriptors

from skfp.fingerprints._new_mordred.utils.periodic_table import HALOGEN_ATOMIC_NUMS

"""
Atom count descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "nAtom",
    "nHeavyAtom",
    "nSpiro",
    "nBridgehead",
    "nHetero",
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

_ELEMENT_ATOMIC_NUMBERS = [
    5,  # B
    6,  # C
    7,  # N
    8,  # O
    16,  # S
    15,  # P
    9,  # F
    17,  # Cl
    35,  # Br
    53,  # I
]


def calc(mol: Mol) -> np.ndarray:
    """
    Count atoms by common element and structural category.
    """
    atoms = mol.GetAtoms()
    atomic_number_counts = Counter(atom.GetAtomicNum() for atom in atoms)
    values = [
        rdMolDescriptors.CalcNumAtoms(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        sum(atom.GetTotalNumHs() for atom in atoms),
    ]
    values.extend(
        atomic_number_counts[element_atomic_number]
        for element_atomic_number in _ELEMENT_ATOMIC_NUMBERS
    )
    values.append(
        sum(
            atomic_number_count
            for atomic_number, atomic_number_count in atomic_number_counts.items()
            if atomic_number in HALOGEN_ATOMIC_NUMS
        )
    )

    return np.asarray(values, dtype=np.float32)
