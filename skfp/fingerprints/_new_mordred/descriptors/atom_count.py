import numpy as np
from rdkit.Chem import Mol, rdMolDescriptors

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
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

_ELEMENT_ATOMIC_NUMBERS = np.array(
    [
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
)
_HALOGEN_ATOMIC_NUMS = np.array(sorted(HALOGEN_ATOMIC_NUMS))


def calc(mol: Mol, props: AtomicProperties) -> tuple[np.ndarray, list[str]]:
    """
    Count atoms by common element and structural category.
    """
    # histogram over atomic numbers, padded so that every element of interest and
    # every halogen can be indexed without a bounds check
    max_atomic_num = max(_ELEMENT_ATOMIC_NUMBERS.max(), _HALOGEN_ATOMIC_NUMS.max())
    counts = np.bincount(props.atomic_nums, minlength=max_atomic_num + 1)

    values = [
        rdMolDescriptors.CalcNumAtoms(mol),
        rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        props.total_num_hs.sum(),
        *counts[_ELEMENT_ATOMIC_NUMBERS],
        counts[_HALOGEN_ATOMIC_NUMS].sum(),
    ]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
