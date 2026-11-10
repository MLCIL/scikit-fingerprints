import numpy as np
from rdkit.Chem import Mol, rdMolDescriptors

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.molecular_properties import (
    MolecularProperties,
)
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
# the histogram over atomic numbers is padded up to this, so that every element of
# interest and every halogen can be indexed without a bounds check
_MAX_ATOMIC_NUM = max(_ELEMENT_ATOMIC_NUMBERS.max(), _HALOGEN_ATOMIC_NUMS.max())


def calc(
    mol_regular: Mol,
    props_regular: AtomicProperties,
    mol_properties: MolecularProperties,
) -> np.ndarray:
    """
    Count atoms by common element and structural category.
    """
    counts = np.bincount(props_regular.atomic_nums, minlength=_MAX_ATOMIC_NUM + 1)

    values = [
        # hydrogens included, whether they are explicit atoms or not
        mol_properties.num_atoms,
        # every atom left, hydrogens that RemoveHs keeps (e.g. isotopes) included
        props_regular.num_atoms,
        rdMolDescriptors.CalcNumSpiroAtoms(mol_regular),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol_regular),
        # anything but carbon and hydrogen, as RDKit counts heteroatoms
        np.count_nonzero(~props_regular.is_carbon & ~props_regular.is_hydrogen),
        # hydrogens stored on atoms, plus those that RemoveHs keeps as atoms
        props_regular.total_num_hs.sum() + np.count_nonzero(props_regular.is_hydrogen),
        *counts[_ELEMENT_ATOMIC_NUMBERS],
        counts[_HALOGEN_ATOMIC_NUMS].sum(),
    ]

    return np.asarray(values, dtype=np.float32)
