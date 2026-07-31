import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    WEIGHTING_PROPERTY_NAMES,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

MAX_DISTANCE = 8

_PROP_NAMES_NO_CHARGE = [
    name for name in WEIGHTING_PROPERTY_NAMES if name != "gasteiger_charge"
]

FEATURE_NAMES = [
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["autocorr", "autocorr_avg"]
        for prop in _PROP_NAMES_NO_CHARGE
        for dist in range(MAX_DISTANCE + 1)
    ],
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["autocorr_centered", "autocorr_avg_centered"]
        for prop in WEIGHTING_PROPERTY_NAMES
        for dist in range(MAX_DISTANCE + 1)
    ],
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["Moreau_autocorr", "Geary_autocorr"]
        for prop in WEIGHTING_PROPERTY_NAMES
        for dist in range(1, MAX_DISTANCE + 1)
    ],
]


def calc(
    atomic_props_hydrogens: AtomicProperties, distance_matrix_hydrogens: DistanceMatrix
) -> np.ndarray:
    """
    Autocorrelation descriptors.

    Quantifies correlation of atomic properties of atoms with the shortest
    path of length k between them. This is realized as a weighted sum
    of shortest path lengths.

    Following the original Mordred implementation, this function uses hydrogen-explicit
    molecule and distance matrix.
    """
    num_atoms = atomic_props_hydrogens.num_atoms

    # one-hot stack of distance masks for d = 1...8, shape (8, n, n)
    dist_masks = np.stack(
        [
            (distance_matrix_hydrogens.matrix == dist)
            for dist in range(1, MAX_DISTANCE + 1)
        ],
        axis=0,
    ).astype(np.float64)

    # number of atoms exactly d bonds away from each atom, shape (8, n)
    neighbor_counts = dist_masks.sum(axis=2)

    # number of unordered atom pairs at each distance, shape (8,)
    pair_counts = 0.5 * neighbor_counts.sum(axis=1)

    # every weighting property at once, shape (n_props, n)
    props = np.stack(
        [atomic_props_hydrogens.get(name) for name in WEIGHTING_PROPERTY_NAMES]
    )
    ats, aats, atsc, aatsc, mats, gats = _get_autocorrelations(
        props, dist_masks, neighbor_counts, pair_counts, num_atoms
    )

    # plain (uncentered) ATS and AATS do not use signed partial charge
    is_charge = np.array(
        [name == "gasteiger_charge" for name in WEIGHTING_PROPERTY_NAMES]
    )
    return np.concatenate(
        [
            ats[~is_charge].ravel(),
            aats[~is_charge].ravel(),
            atsc.ravel(),
            aatsc.ravel(),
            mats.ravel(),
            gats.ravel(),
        ],
        dtype=np.float32,
    )


@np.errstate(divide="ignore", invalid="ignore")
def _get_autocorrelations(
    props: np.ndarray,
    dist_masks: np.ndarray,
    neighbor_counts: np.ndarray,
    pair_counts: np.ndarray,
    num_atoms: int,
) -> tuple[np.ndarray, ...]:
    """
    Calculate every autocorrelation descriptor family, for every atomic property.

    All families are functions of the same two quantities, computed here for all
    properties and distances at once: the quadratic form ``p^T M_d p`` and the
    weighted square sum ``sum_i deg_d(i) p_i^2``, where ``M_d`` is the distance-d
    mask and ``deg_d`` the number of atoms d bonds away.

    Every returned array is indexed by property and then by distance, which is also
    the order the feature names are in.
    """
    # M_d @ p for every property and distance, shape (n_props, 8, n)
    weighted = np.einsum("dij,pj->pdi", dist_masks, props)
    # p^T M_d p, shape (n_props, 8)
    quadratic_form = np.einsum("pdi,pi->pd", weighted, props)

    # ATS: sum over unordered atom pairs at distance d of p_i * p_j
    # masks are symmetric with zero diagonal, so we divide by 2
    square_sums = np.sum(props**2, axis=1)
    ats = np.column_stack([square_sums, 0.5 * quadratic_form])
    aats = _per_pair_average(ats, pair_counts, num_atoms)

    # ATSC: like above, but on mean-centered properties
    # note that centering commutes with the product, so a mask applied to a
    # centered property is the same as the mask applied to the property
    # minus its mean times the neighbor counts
    means = props.mean(axis=1, keepdims=True)
    props_centered = props - means
    weighted_centered = weighted - means[:, :, np.newaxis] * neighbor_counts
    centered_square_sums = np.sum(props_centered**2, axis=1)
    atsc = np.column_stack(
        [
            centered_square_sums,
            0.5 * np.einsum("pdi,pi->pd", weighted_centered, props_centered),
        ]
    )
    aatsc = _per_pair_average(atsc, pair_counts, num_atoms)

    # MATS (Moran coefficient): the centered per-pair average, normalized by
    # property variance around its mean
    variation = centered_square_sums[:, np.newaxis]
    mats = np.where(variation != 0, num_atoms * aatsc[:, 1:] / variation, np.nan)
    mats = np.where(pair_counts != 0, mats, np.nan)

    # GATS (Geary coefficient): mean squared difference between paired atoms,
    # normalized by the property variance
    # note: expanding (p_i - p_j)^2 over the mask gives:
    # 2 * sum_i deg_d(i) p_i^2 - 2 * (p^T M_d p)
    # so no pairwise difference matrix has to be formed explicitly, we can use the
    # quadratic form from above
    sum_squared_diff = 2.0 * (props**2 @ neighbor_counts.T) - 2.0 * quadratic_form
    # make sure we get non-negative value (could happen due to float arithmetic)
    sum_squared_diff = np.maximum(sum_squared_diff, 0.0)
    mean_squared_diff = np.where(
        pair_counts != 0, sum_squared_diff / (4 * pair_counts), np.nan
    )
    props_var = props.var(axis=1, ddof=1)[:, np.newaxis]
    gats = np.where(props_var != 0, mean_squared_diff / props_var, np.nan)
    gats = np.where(pair_counts != 0, gats, np.nan)

    return ats, aats, atsc, aatsc, mats, gats


def _per_pair_average(
    values: np.ndarray, pair_counts: np.ndarray, num_atoms: int
) -> np.ndarray:
    """
    Average ATS-like values over the number of contributing atom pairs.

    Distance 0 pairs an atom with itself, so it is averaged over the atom count,
    while the remaining distances are averaged over their pair count and are NaN
    where no such pair exists.
    """
    averaged = np.where(pair_counts != 0, values[:, 1:] / pair_counts, np.nan)
    return np.column_stack([values[:, 0] / num_atoms, averaged])
