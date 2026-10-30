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

# plain (uncentered) ATS and AATS do not use signed partial charge
_IS_CHARGE = np.array(
    [name == "Gasteiger_partial_charge" for name in WEIGHTING_PROPERTY_NAMES]
)
_PROP_NAMES_NO_CHARGE = [
    name
    for name, is_charge in zip(WEIGHTING_PROPERTY_NAMES, _IS_CHARGE, strict=True)
    if not is_charge
]

# e.g. "Moreau-Broto autocorrelation atomic number lag 1" (Mordred's ATS1Z), "average
# centered Moreau-Broto autocorrelation mass lag 2" (AATSC2m), "Moran autocorrelation
# intrinsic state lag 7" (MATS7s)
FEATURE_NAMES = [
    *[
        f"{desc} {prop.replace('_', ' ')} lag {dist}"
        for desc in [
            "Moreau-Broto autocorrelation",
            "average Moreau-Broto autocorrelation",
        ]
        for prop in _PROP_NAMES_NO_CHARGE
        for dist in range(MAX_DISTANCE + 1)
    ],
    *[
        f"{desc} {prop.replace('_', ' ')} lag {dist}"
        for desc in [
            "centered Moreau-Broto autocorrelation",
            "average centered Moreau-Broto autocorrelation",
        ]
        for prop in WEIGHTING_PROPERTY_NAMES
        for dist in range(MAX_DISTANCE + 1)
    ],
    *[
        f"{desc} {prop.replace('_', ' ')} lag {dist}"
        for desc in ["Moran autocorrelation", "Geary autocorrelation"]
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
    path of length d between them. This is realized as a sum, over the atom
    pairs at distance d, of the products of their properties.

    Following the original Mordred implementation, this function uses the atomic
    properties and distance matrix of the hydrogen-explicit molecule.
    """
    num_atoms = atomic_props_hydrogens.num_atoms

    # one-hot stack of distance masks for d = 1...8, shape (8, n, n)
    dist_masks = np.stack(
        [
            (distance_matrix_hydrogens.matrix == dist)
            for dist in range(1, MAX_DISTANCE + 1)
        ],
        axis=0,
    )

    # number of atoms exactly d bonds away from each atom, shape (8, n)
    neighbor_counts = dist_masks.sum(axis=2)

    # number of unordered atom pairs at each distance, shape (8,)
    pair_counts = 0.5 * neighbor_counts.sum(axis=1)

    # every weighting property at once, shape (n_props, n)
    props = atomic_props_hydrogens.weighting_properties
    ats, aats, atsc, aatsc, mats, gats = _get_autocorrelations(
        props, dist_masks, neighbor_counts, pair_counts, num_atoms
    )

    return np.concatenate(
        [
            ats[~_IS_CHARGE].ravel(),
            aats[~_IS_CHARGE].ravel(),
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
    weighted square sum ``sum_i deg_d(i) p_i^2``. Here ``p`` is the column vector of
    one property over the atoms, ``M_d`` the symmetric 0/1 matrix of atom pairs at
    distance d, and ``deg_d = M_d 1`` the number of atoms d bonds away from each atom.

    Every returned array is indexed by property and then by distance, which is also
    the order the feature names are in.
    """
    # centering first, so that what vanishes for a property equal on every atom
    # comes out as an exact zero, not residue the guards below would not recognize
    means = props.mean(axis=1, keepdims=True)
    props_centered = props - means

    # row vector p_c^T M_d for every distance and property, shape (8, n_props, n)
    # (equal to (M_d p_c)^T, as the masks are symmetric)
    weighted_centered = props_centered @ dist_masks

    # p_c^T M_d p_c, summed over atoms and transposed to shape (n_props, 8)
    centered_quadratic_form = np.sum(weighted_centered * props_centered, axis=2).T

    # summing the same row vector over the atoms instead gives deg_d . p_c
    degree_dot_centered = weighted_centered.sum(axis=2).T

    # as p = p_c + mean * 1, the uncentered form follows without multiplying the
    # masks again: p^T M_d p = p_c^T M_d p_c + 2 mean (deg_d . p_c) + mean^2 1^T M_d 1,
    # the last term over ordered pairs. It never vanishes, so the sum is exact enough
    quadratic_form = (
        centered_quadratic_form
        + 2.0 * means * degree_dot_centered
        + means**2 * 2.0 * pair_counts
    )

    # ATS: sum over unordered atom pairs at distance d of p_i * p_j
    # masks are symmetric with zero diagonal, so we divide by 2
    square_sums = np.sum(props**2, axis=1)
    ats = np.column_stack([square_sums, 0.5 * quadratic_form])
    aats = _per_pair_average(ats, pair_counts, num_atoms)

    # ATSC: like above, but on mean-centered properties
    centered_square_sums = np.sum(props_centered**2, axis=1)
    atsc = np.column_stack([centered_square_sums, 0.5 * centered_quadratic_form])
    aatsc = _per_pair_average(atsc, pair_counts, num_atoms)

    # MATS (Moran coefficient): the centered per-pair average, normalized by
    # property variance around its mean
    variation = centered_square_sums[:, np.newaxis]
    mats = np.where(variation != 0, num_atoms * aatsc[:, 1:] / variation, np.nan)

    # GATS (Geary coefficient): mean squared difference between paired atoms,
    # normalized by the property sample variance
    # summing (p_i - p_j)^2 over the mask gives 2 sum_i deg_d(i) p_i^2 - 2 p^T M_d p,
    # so no pairwise difference matrix is needed; the mean cancels out of the
    # differences, and centered the two terms cancel exactly for a constant property
    sum_squared_diff = (
        2.0 * (props_centered**2 @ neighbor_counts.T) - 2.0 * centered_quadratic_form
    )
    # make sure we get non-negative value (could happen due to float arithmetic)
    sum_squared_diff = np.maximum(sum_squared_diff, 0.0)
    # Geary divides the sum over unordered pairs by 2 * pair_counts, and the sum
    # over ordered pairs above is twice that, hence 4 * pair_counts
    mean_squared_diff = np.where(
        pair_counts != 0, sum_squared_diff / (4 * pair_counts), np.nan
    )
    # sample variance, from the centered square sums above, NaN for a single atom
    # (unlike np.var(ddof=1), the 0 / 0 here stays under np.errstate, without warning)
    props_var = variation / (num_atoms - 1)
    gats = np.where(props_var != 0, mean_squared_diff / props_var, np.nan)

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
