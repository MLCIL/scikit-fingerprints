import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    CARBON_ELEMENT_PROPERTIES,
    AtomicProperties,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    # sums, one per element property, in ELEMENT_PROPERTY_TABLES order
    "SZ",
    "Sm",
    "Sv",
    "Sse",
    "Spe",
    "Sare",
    "Sp",
    "Si",
    # the same properties again, as means
    "MZ",
    "Mm",
    "Mv",
    "Mse",
    "Mpe",
    "Mare",
    "Mp",
    "Mi",
]


def calc(props_hydrogens: AtomicProperties) -> np.ndarray:
    """
    Constitutional descriptors: the sum (``S*``) and mean (``M*``) over the atoms of
    every element property, each normalized by the property's value for carbon.

    Hydrogens count as atoms here, so this works on the hydrogen-explicit molecule.
    Normalizing the sums rather than the individual atoms divides eight times instead
    of once per atom, which the sum of a quotient permits.
    """
    sums = props_hydrogens.element_properties.sum(axis=1) / CARBON_ELEMENT_PROPERTIES
    return np.concatenate((sums, sums / props_hydrogens.num_atoms)).astype(np.float32)
