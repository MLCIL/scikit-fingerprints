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
) -> tuple[np.ndarray, list[str]]:
    """
    Autocorrelation descriptors.

    Quantifies correlation of atomic properties of atoms with the shortest
    path of length k between them. This is realized as a weighted sum
    of shortest path lengths.

    Following the original Mordred implementation, this function uses hydrogen-explicit
    molecule and distance matrix.
    """
    # one-hot stack of distance masks for d = 1...8, shape (8, n, n)
    dist_stack = np.stack(
        [
            (distance_matrix_hydrogens.matrix == dist)
            for dist in range(1, MAX_DISTANCE + 1)
        ],
        axis=0,
    ).astype(np.float64)

    # number of atoms exactly d bonds away from each atom, shape (8, n)
    neighbor_counts = dist_stack.sum(axis=2)

    # number of unordered atom pairs at each distance, shape (8,)
    pair_counts = 0.5 * neighbor_counts.sum(axis=1)

    ats: list[float] = []
    aats: list[float] = []
    atsc: list[float] = []
    aatsc: list[float] = []
    mats: list[float] = []
    gats: list[float] = []

    for prop_name in WEIGHTING_PROPERTY_NAMES:
        props = atomic_props_hydrogens.get(prop_name)
        prop_ats, prop_aats, prop_atsc, prop_aatsc, prop_mats, prop_gats = (
            _get_autocorrelations(props, dist_stack, neighbor_counts, pair_counts)
        )

        # plain (uncentered) ATS and AATS do not use signed partial charge
        if prop_name != "gasteiger_charge":
            ats.extend(prop_ats)
            aats.extend(prop_aats)

        atsc.extend(prop_atsc)
        aatsc.extend(prop_aatsc)
        mats.extend(prop_mats)
        gats.extend(prop_gats)

    all_values = np.concatenate([ats, aats, atsc, aatsc, mats, gats], dtype=np.float32)
    return all_values, FEATURE_NAMES


@np.errstate(divide="ignore", invalid="ignore")
def _get_autocorrelations(
    props: np.ndarray,
    dist_stack: np.ndarray,
    neighbor_counts: np.ndarray,
    pair_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate all autocorrelation descriptors for a single atomic property.

    Every family is derived from one bulk product of the distance masks with the
    property vector, since they are all functions of the same two per-distance
    quantities: the quadratic form ``p^T M_d p`` and the weighted square sum
    ``sum_i deg_d(i) p_i^2``, where ``M_d`` is the distance-d mask and ``deg_d``
    the number of atoms d bonds away.
    """
    num_atoms = len(props)

    # (8, n, n) @ (n,) -> (8, n), so M_d @ p for every distance at once
    weighted = dist_stack @ props
    quadratic_form = (weighted * props).sum(axis=1)  # p^T M_d p, shape (8,)

    # ATS: sum over unordered atom pairs at distance d of p_i * p_j
    # mask is symmetric with zero diagonal, so we divide by 2
    ats = np.concatenate([[np.sum(props**2)], 0.5 * quadratic_form])
    aats = _per_pair_average(ats, pair_counts, num_atoms)

    # ATSC: like above, but on mean-centered property
    # note that centering commutes with the product, so mask applied to
    # centered property is the same as mask applied to the property
    # minus mean times neighbor counts
    mean = props.mean()
    props_centered = props - mean
    weighted_centered = weighted - mean * neighbor_counts
    sum_squared_centered = np.sum(props_centered**2)
    atsc = np.concatenate(
        [
            [sum_squared_centered],
            0.5 * (weighted_centered * props_centered).sum(axis=1),
        ]
    )
    aatsc = _per_pair_average(atsc, pair_counts, num_atoms)

    # MATS (Moran coefficient): the centered per-pair average, normalized by
    # property variance around its mean
    if sum_squared_centered != 0:
        mats = num_atoms * aatsc[1:] / sum_squared_centered
    else:
        mats = np.full(len(pair_counts), np.nan)
    mats = np.where(pair_counts != 0, mats, np.nan)

    # GATS (Geary coefficient): mean squared difference between paired atoms,
    # normalized by the property variance
    # note: expanding (p_i - p_j)^2 over the mask gives:
    # 2 * sum_i deg_d(i) p_i^2 - 2 * (p^T M_d p)
    # so no pairwise difference matrix has to be formed explicitly, we can use the
    # quadratic form from above
    sum_squared_diff = np.maximum(
        2.0 * (neighbor_counts @ props**2) - 2.0 * quadratic_form, 0.0
    )
    # make sure we get non-negative value (could happen due to float arithmetic)
    sum_squared_diff = np.maximum(sum_squared_diff, 0)
    mean_squared_diff = np.where(
        pair_counts != 0, sum_squared_diff / (4 * pair_counts), np.nan
    )
    props_var = np.var(props, ddof=1)
    if props_var != 0:
        gats = mean_squared_diff / props_var
    else:
        gats = np.full(len(pair_counts), np.nan)
    gats = np.where(pair_counts != 0, gats, np.nan)

    return ats, aats, atsc, aatsc, mats, gats


def _per_pair_average(
    values: np.ndarray, pair_counts: np.ndarray, num_atoms: int
) -> np.ndarray:
    """
    Average an ATS-like array over the number of contributing atom pairs.

    Distance 0 pairs an atom with itself, so it is averaged over the atom count,
    while the remaining distances are averaged over their pair count and are NaN
    where no such pair exists.
    """
    averaged = np.where(pair_counts != 0, values[1:] / pair_counts, np.nan)
    return np.concatenate([[values[0] / num_atoms], averaged])
