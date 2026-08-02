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

    # the properties differ only along the diagonal, so the matrices are stacked and
    # decomposed in one call; an undefined property, such as the Gasteiger charge of
    # a metal, has no matrix of its own and stays NaN
    prop_vals = np.stack([props.get(name) for name in WEIGHTING_PROPERTY_NAMES])
    is_defined = np.isfinite(prop_vals).all(axis=1)

    matrices = np.repeat(
        _get_burden_matrix(props)[np.newaxis], is_defined.sum(), axis=0
    )
    diagonal = np.arange(props.num_atoms)
    matrices[:, diagonal, diagonal] = prop_vals[is_defined]

    values = np.full((len(prop_vals), 2), np.nan)
    eigvals = np.linalg.eigvalsh(matrices)  # ascending order
    values[is_defined] = eigvals[:, [0, -1]]

    return values.ravel().astype(np.float32)


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
