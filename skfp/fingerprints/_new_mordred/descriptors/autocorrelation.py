import numpy as np
from rdkit.Chem import Mol

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
_PROPS_COMMON_NAMES = [
    "atomic_number",
    "mass",
    "van_der_Waals_volume",
    "Sanderson_electronegativity",
    "Pauling_electronegativity",
    "Allred_Rochow_electronegativity",
    "polarizability",
    "ionization_potential",
]

_PROPS_COMMON_FUNCS = [
    get_atomic_number,
    get_mass,
    get_van_der_waals_volume,
    get_sanderson_electronegativity,
    get_pauling_electronegativity,
    get_allred_rochow_electronegativity,
    get_polarizability,
    get_ionization_potential,
]

# valence-related properties, used only by some descriptors
_PROPS_VALENCE_NAMES = [
    "valence_electrons",  # http://dx.doi.org/10.1002%2Fjps.2600721016
    "sigma_electrons",
    "intrinsic_state",  # http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html, p.283
]

_PROPS_VALENCE_FUNCS = [
    get_valence_electrons,
    get_sigma_electrons,
    get_intrinsic_state,
]

# Gasteiger charges, used only by some descriptors
_PROP_NAMES_GASTEIGER = ["gasteiger_charge"]


FEATURE_NAMES = [
    *[
        f"ATS_{prop}_lag_{dist}"
        for prop in _PROPS_COMMON_NAMES + _PROPS_VALENCE_NAMES
        for dist in range(9)
    ],
]


def calc(
    mol_regular: Mol, distance_matrix_regular: DistanceMatrix
) -> tuple[np.ndarray, list[str]]:
    """
    Autocorrelation descriptors.

    Quantifies correlation of atomic properties of atoms with the shortest
    path of length ``k`` between them. This is realized as a weighted sum
    of shortest path lengths.
    """
    values = np.concatenate(
        [_calc_ats(mol_regular, distance_matrix_regular)], dtype=np.float32
    )
    return values, FEATURE_NAMES


def _calc_ats(mol: Mol, distance_matrix: DistanceMatrix) -> list[float]:
    """
    Autocorrelation of Topological Structure (ATS) descriptors.

    Also known as Moreau-Broto correlation.
    """
    # TODO: this probably can be rewritten as matrix multiplications
    values = []
    for prop_func in _PROPS_COMMON_FUNCS + _PROPS_VALENCE_FUNCS:
        props_vec = np.array([prop_func(atom) for atom in mol.GetAtoms()])

        # distance 0 has separate formula
        val = np.sum(np.square(props_vec))
        values.append(val)

        for distance in range(1, 9):
            dists_mask = distance_matrix.matrix == distance
            val = 0.5 * props_vec.dot(dists_mask).dot(props_vec)
            values.append(val)

    return values
