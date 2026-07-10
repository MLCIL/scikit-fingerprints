import numpy as np
from rdkit.Chem import Mol

from skfp.descriptors import atomic_partial_charges
from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    PROPERTY_FUNCS,
    get_intrinsic_state,
    get_sigma_electrons,
    get_valence_electrons,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# atomic properties used to weight distance matrices
_PROPS = {
    **PROPERTY_FUNCS,
    "valence_electrons": get_valence_electrons,
    "sigma_electrons": get_sigma_electrons,  # http://dx.doi.org/10.1002%2Fjps.2600721016
    "intrinsic_state": get_intrinsic_state,  # http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html, p.283
}


FEATURE_NAMES = [
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["autocorr", "autocorr_avg"]
        for prop in _PROPS
        for dist in range(9)
    ],
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["autocorr_centered", "autocorr_avg_centered"]
        for prop in [*_PROPS, "gasteiger_charge"]
        for dist in range(9)
    ],
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["Moreau_autocorr", "Geary_autocorr"]
        for prop in [*_PROPS, "gasteiger_charge"]
        for dist in range(1, 9)
    ],
]


def calc(
    mol_hydrogens: Mol, distance_matrix_hydrogens: DistanceMatrix
) -> tuple[np.ndarray, list[str]]:
    """
    Autocorrelation descriptors.

    Quantifies correlation of atomic properties of atoms with the shortest
    path of length k between them. This is realized as a weighted sum
    of shortest path lengths.

    Following the original Mordred implementation, this function uses hydrogen-explicit
    molecule and distance matrix.
    """
    atoms = list(mol_hydrogens.GetAtoms())
    atomic_props = {}
    for name, func in _PROPS.items():
        atomic_props[name] = np.array([func(atom) for atom in atoms], dtype=np.float64)

    atomic_props["gasteiger_charge"] = atomic_partial_charges(
        mol_hydrogens, partial_charge_model="Gasteiger", charge_errors="ignore"
    )

    # one-hot stack of distance masks for d = 1...8, shape (8, n, n)
    # allows bulk tensor computation
    dist_stack = np.stack(
        [(distance_matrix_hydrogens.matrix == d) for d in range(1, 9)], axis=0
    )

    # number of unordered pairs at each distance (0.5 * count of True), shape (8,)
    pair_counts = 0.5 * dist_stack.sum(axis=(1, 2))

    ats, aats = _calc_ats_aats(atomic_props, dist_stack, pair_counts)
    atsc, aatsc, mats = _calc_atsc_aatsc_mats(atomic_props, dist_stack, pair_counts)
    gats = _calc_gats(atomic_props, dist_stack, pair_counts)

    values = np.concatenate([ats, aats, atsc, aatsc, mats, gats], dtype=np.float32)
    return values, FEATURE_NAMES


@np.errstate(divide="ignore", invalid="ignore")
def _calc_ats_aats(
    atomic_props: dict[str, np.ndarray],
    dist_stack: np.ndarray,
    pair_counts: np.ndarray,
) -> tuple[list[float], list[float]]:
    """
    Autocorrelation of Topological Structure (ATS) descriptors.

    Moreau-Broto autocorrelation descriptors, based on weighted atomic
    correlations at a given distance.
    """
    ats_values = []
    aats_values = []

    for prop_name in _PROPS:
        props = atomic_props[prop_name]

        # distance 0 has separate formula
        ats_0 = np.sum(props**2)
        ats_values.append(ats_0)
        aats_values.append(ats_0 / len(props))

        ats_d = _weighted_sums(props, dist_stack)  # shape (8,)
        aats_d = np.where(pair_counts != 0, ats_d / pair_counts, np.nan)

        ats_values.extend(ats_d.tolist())
        aats_values.extend(aats_d.tolist())

    return ats_values, aats_values


@np.errstate(divide="ignore", invalid="ignore")
def _calc_atsc_aatsc_mats(
    atomic_props: dict[str, np.ndarray],
    dist_stack: np.ndarray,
    pair_counts: np.ndarray,
) -> tuple[list[float], list[float], list[float]]:
    """
    ATS centered descriptors, Moran coefficient descriptors (MATS).
    """
    atsc_values = []
    aatsc_values = []
    mats_values = []

    for prop_name in [*_PROPS, "gasteiger_charge"]:
        props = atomic_props[prop_name]
        props_centered = props - np.mean(props)
        sum_squared_props_vec_c = np.sum(props_centered**2)

        # distance 0 has separate formula
        atsc_0 = sum_squared_props_vec_c
        atsc_values.append(atsc_0)
        aatsc_values.append(atsc_0 / len(props))

        atsc_d = _weighted_sums(props_centered, dist_stack)  # shape (8,)
        aatsc_d = np.where(pair_counts != 0, atsc_d / pair_counts, np.nan)

        if sum_squared_props_vec_c != 0:
            mats_d = len(props) * aatsc_d / sum_squared_props_vec_c
        else:
            mats_d = np.full_like(aatsc_d, np.nan)

        mats_d = np.where(pair_counts != 0, mats_d, np.nan)

        atsc_values.extend(atsc_d.tolist())
        aatsc_values.extend(aatsc_d.tolist())
        mats_values.extend(mats_d.tolist())

    return atsc_values, aatsc_values, mats_values


@np.errstate(divide="ignore", invalid="ignore")
def _calc_gats(
    atomic_props: dict[str, np.ndarray],
    dist_stack: np.ndarray,
    pair_counts: np.ndarray,
) -> list[float]:
    """
    Geary coefficient descriptors.
    """
    gats_values = []

    for prop_name in [*_PROPS, "gasteiger_charge"]:
        props = atomic_props[prop_name]
        props_var = np.var(props, ddof=1)

        # squared pairwise differences, formed once and reused per distance
        pairs_sq_diff = (props[:, np.newaxis] - props) ** 2  # shape (n, n)
        sum_sq_diff = np.tensordot(
            dist_stack, pairs_sq_diff, axes=([1, 2], [0, 1])
        )  # shape (8,)

        denom = 4 * pair_counts
        mean_squared_diff = np.where(denom != 0, sum_sq_diff / denom, np.nan)
        if props_var != 0:
            gats_d = mean_squared_diff / props_var
        else:
            gats_d = np.full_like(mean_squared_diff, np.nan)

        gats_d = np.where(pair_counts != 0, gats_d, np.nan)

        gats_values.extend(gats_d.tolist())

    return gats_values


def _weighted_sums(props: np.ndarray, dist_stack: np.ndarray) -> np.ndarray:
    """
    For each distance d (1..8), compute 0.5 * props @ mask_d @ props.

    props_i * props_j is computed as an outer product, giving inter-atomic
    property correlations, with distance mask to apply to relevant entries.
    """
    # (n,) outer product (n,) -> (n, n)
    props_corr_matrix = np.multiply.outer(props, props)

    # dist_stack (distance masks tensor) has shape (8, n, n)
    # (8, n, n) tensor dot product (n, n) -> (8,)
    return 0.5 * np.tensordot(dist_stack, props_corr_matrix, axes=([1, 2], [0, 1]))
