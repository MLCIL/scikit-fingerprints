import numpy as np
from rdkit.Chem import Mol

from skfp.descriptors import atomic_partial_charges
from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_allred_rochow_electronegativity,
    get_atomic_number,
    get_intrinsic_state,
    get_ionization_potential,
    get_mass,
    get_pauling_electronegativity,
    get_polarizability,
    get_sanderson_electronegativity,
    get_sigma_electrons,
    get_valence_electrons,
    get_van_der_waals_volume,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# atomic properties used to weight distance matrices
_PROPS_NAMES = [
    "atomic_number",
    "mass",
    "van_der_Waals_volume",
    "Sanderson_electronegativity",
    "Pauling_electronegativity",
    "Allred_Rochow_electronegativity",
    "polarizability",
    "ionization_potential",
    "valence_electrons",  # http://dx.doi.org/10.1002%2Fjps.2600721016
    "sigma_electrons",
    "intrinsic_state",  # http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html, p.283
]
_PROPS_FUNCS = [
    get_atomic_number,
    get_mass,
    get_van_der_waals_volume,
    get_sanderson_electronegativity,
    get_pauling_electronegativity,
    get_allred_rochow_electronegativity,
    get_polarizability,
    get_ionization_potential,
    get_valence_electrons,
    get_sigma_electrons,
    get_intrinsic_state,
]


FEATURE_NAMES = [
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["autocorr", "autocorr_avg"]
        for prop in _PROPS_NAMES
        for dist in range(9)
    ],
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["autocorr_centered", "autocorr_avg_centered"]
        for prop in [*_PROPS_NAMES, "gasteiger_charge"]
        for dist in range(9)
    ],
    *[
        f"{desc}_{prop}_lag_{dist}"
        for desc in ["Moreau_autocorr", "Geary_autocorr"]
        for prop in [*_PROPS_NAMES, "gasteiger_charge"]
        for dist in range(1, 9)
    ],
]


def calc(mol: Mol, distance_matrix: DistanceMatrix) -> tuple[np.ndarray, list[str]]:
    """
    Autocorrelation descriptors.

    Quantifies correlation of atomic properties of atoms with the shortest
    path of length k between them. This is realized as a weighted sum
    of shortest path lengths.
    """
    atomic_props = {}
    for name, func in zip(_PROPS_NAMES, _PROPS_FUNCS, strict=True):
        props = np.array([func(atom) for atom in mol.GetAtoms()])
        atomic_props[name] = props

    atomic_props["gasteiger_charge"] = atomic_partial_charges(
        mol, partial_charge_model="Gasteiger", charge_errors="ignore"
    )

    dist_masks = [distance_matrix.matrix == d for d in range(1, 9)]

    ats, aats = _calc_ats_aats(atomic_props, dist_masks)
    atsc, aatsc, mats = _calc_atsc_aatsc_mats(atomic_props, dist_masks)
    gats = _calc_gats(atomic_props, dist_masks)

    values = np.concatenate([ats, aats, atsc, aatsc, mats, gats], dtype=np.float32)
    return values, FEATURE_NAMES


@np.errstate(divide="raise")
def _calc_ats_aats(
    atomic_props: dict[str, np.ndarray], dist_masks: list[np.ndarray]
) -> tuple[list[float], list[float]]:
    """
    Autocorrelation of Topological Structure (ATS) descriptors.

    Moreau-Broto autocorrelation descriptors, based on weighted atomic
    correlations at a given distance.
    """
    ats_values = []
    aats_values = []

    for prop_name in _PROPS_NAMES:
        props = atomic_props[prop_name]

        # distance 0 has separate formula
        ats_0 = np.sum(props**2)
        ats_values.append(ats_0)

        aats_0 = ats_0 / len(props)
        aats_values.append(aats_0)

        for dists_eq_d in dist_masks:
            ats_d = 0.5 * props.dot(dists_eq_d).dot(props)
            ats_values.append(ats_d)

            try:
                aats_d = ats_d / (0.5 * np.sum(dists_eq_d))
            except FloatingPointError:
                aats_d = np.nan

            aats_values.append(aats_d)

    return ats_values, aats_values


@np.errstate(divide="raise")
def _calc_atsc_aatsc_mats(
    atomic_props: dict[str, np.ndarray], dist_masks: list[np.ndarray]
) -> tuple[list[float], list[float], list[float]]:
    """
    ATS centered descriptors, Moran coefficient descriptors (MATS).
    """
    atsc_values = []
    aatsc_values = []
    mats_values = []

    for prop_name in [*_PROPS_NAMES, "gasteiger_charge"]:
        props = atomic_props[prop_name]
        props_centered = props - np.mean(props)

        # sum of squared properties
        sum_squared_props_vec_c = np.sum(props_centered**2)

        # distance 0 has separate formula
        atsc_0 = sum_squared_props_vec_c
        atsc_values.append(atsc_0)

        aatsc_0 = atsc_0 / len(props)
        aatsc_values.append(aatsc_0)

        for dists_eq_d in dist_masks:
            atsc_d = 0.5 * props_centered.dot(dists_eq_d).dot(props_centered)
            atsc_values.append(atsc_d)

            try:
                aatsc_d = atsc_d / (0.5 * np.sum(dists_eq_d))
                mats_d = len(props) * aatsc_d / sum_squared_props_vec_c
            except FloatingPointError:
                aatsc_d = np.nan
                mats_d = np.nan

            aatsc_values.append(aatsc_d)
            mats_values.append(mats_d)

    return atsc_values, aatsc_values, mats_values


@np.errstate(divide="raise")
def _calc_gats(
    atomic_props: dict[str, np.ndarray], dist_masks: list[np.ndarray]
) -> list[float]:
    """
    Geary coefficient descriptors.
    """
    gats_values = []

    for prop_name in [*_PROPS_NAMES, "gasteiger_charge"]:
        props = atomic_props[prop_name]
        props_var = np.var(props, ddof=1)

        pairs_sq_diff = (props[:, np.newaxis] - props) ** 2

        for dists_eq_d in dist_masks:
            n_pairs = 0.5 * np.sum(dists_eq_d)

            try:
                mean_squared_diff = np.sum(dists_eq_d * pairs_sq_diff) / (4 * n_pairs)
                gats_d = mean_squared_diff / props_var
            except FloatingPointError:
                gats_d = np.nan

            gats_values.append(gats_d)

    return gats_values
