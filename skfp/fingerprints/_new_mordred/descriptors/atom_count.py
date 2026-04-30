"""
Atom count descriptors implemented with direct RDKit atom access.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np
from rdkit.Chem import Mol, rdMolDescriptors

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
_HALOGENS = {9, 17, 35, 53}


def calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred atom count descriptors.
    """
    atoms = mol.GetAtoms()
    atomic_numbers = [atom.GetAtomicNum() for atom in atoms]
    values = [
        rdMolDescriptors.CalcNumAtoms(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        sum(atom.GetTotalNumHs() for atom in atoms),
    ]
    values.extend(
        sum(atomic_number == element_atomic_number for atomic_number in atomic_numbers)
        for element_atomic_number in _ELEMENT_ATOMIC_NUMBERS
    )
    values.append(sum(atomic_number in _HALOGENS for atomic_number in atomic_numbers))

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
