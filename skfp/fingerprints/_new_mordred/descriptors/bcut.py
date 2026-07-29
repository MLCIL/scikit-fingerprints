import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    WEIGHTING_PROPERTY_NAMES,
    AtomicProperties,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


FEATURE_NAMES = [
    f"BCUT_{prop}_{kind}_eigval"
    for prop in WEIGHTING_PROPERTY_NAMES
    for kind in ["smallest", "largest"]
]


def calc(props: AtomicProperties, n_frags: int) -> np.ndarray:
    """
    BCUT descriptors.

    Burden eigenvalues: the off-diagonal entries of the Burden matrix encode the
    bonding pattern, while the diagonal carries an atomic property. The smallest
    and largest eigenvalues are taken for each property.

    Requires a connected molecule (single fragment).
    """
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    burden_matrix = _get_burden_matrix(props)

    values = []
    for prop_name in WEIGHTING_PROPERTY_NAMES:
        prop_vals = props.get(prop_name)
        if not np.isfinite(prop_vals).all():
            # undefined property, e.g. Gasteiger charge of a metal
            values.extend([np.nan, np.nan])
            continue

        np.fill_diagonal(burden_matrix, prop_vals)
        eigvals = np.linalg.eigvalsh(burden_matrix)  # ascending order
        values.extend([eigvals[0], eigvals[-1]])

    return np.asarray(values, dtype=np.float32)


def _get_burden_matrix(props: AtomicProperties) -> np.ndarray:
    """
    Burden matrix with an as-yet unset diagonal.

    Off-diagonal entries are 0.001 for atom pairs that are not bonded, and the
    bond order divided by ten for bonded pairs, with 0.01 added when either atom
    is terminal.
    """
    matrix = np.full((props.num_atoms, props.num_atoms), 0.001)

    begins = props.bond_begin_idxs
    ends = props.bond_end_idxs
    degrees = props.degrees
    weights = props.bond_orders / 10.0
    weights = weights + 0.01 * ((degrees[begins] == 1) | (degrees[ends] == 1))

    matrix[begins, ends] = weights
    matrix[ends, begins] = weights

    return matrix
