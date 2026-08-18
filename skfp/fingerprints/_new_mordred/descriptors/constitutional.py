import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    PROPERTY_FUNCS,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "SZ",
    "Sm",
    "Sv",
    "Sse",
    "Spe",
    "Sare",
    "Sp",
    "Si",
    "MZ",
    "Mm",
    "Mv",
    "Mse",
    "Mpe",
    "Mare",
    "Mp",
    "Mi",
]

_NUM_ELEMENTS = 119


def _get_normalized_property_table() -> np.ndarray:
    """
    Precompute each atomic property divided by its value for carbon.
    """
    atoms = tuple(Atom(atomic_num) for atomic_num in range(_NUM_ELEMENTS))
    property_values = np.asarray(
        [
            [property_func(atom) for atom in atoms]
            for property_func in PROPERTY_FUNCS.values()
        ],
        dtype=np.float64,
    )
    # Keep shape as (num_properties, 1) so broadcasting divides each row by carbon
    # while preserving the atom axis.
    carbon_values = property_values[:, 6].reshape(-1, 1)
    return property_values / carbon_values


_CARBON_NORMALIZED_PROPERTIES = _get_normalized_property_table()


def calc(mol_hydrogens: Mol) -> np.ndarray:
    """
    Compute the Mordred constitutional descriptors.

    For each atomic property, the property values of all atoms, including explicit
    hydrogens, are normalized by the corresponding value for carbon. The `S*`
    descriptors are their sums and the `M*` descriptors are their means.
    """
    atomic_numbers = np.fromiter(
        (atom.GetAtomicNum() for atom in mol_hydrogens.GetAtoms()),
        dtype=np.intp,
        count=mol_hydrogens.GetNumAtoms(),
    )
    sums = _CARBON_NORMALIZED_PROPERTIES[:, atomic_numbers].sum(axis=1)
    means = sums / atomic_numbers.size
    return np.concatenate((sums, means)).astype(np.float32)
