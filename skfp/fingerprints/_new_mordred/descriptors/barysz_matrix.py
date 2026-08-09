import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom
from scipy.sparse.csgraph import floyd_warshall

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    PROPERTY_FUNCS,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_ATTR_NAMES = [
    "SpAbs",
    "SpMax",
    "SpDiam",
    "SpAD",
    "SpMAD",
    "LogEE",
    "SM1",
    "VE1",
    "VE2",
    "VE3",
    "VR1",
    "VR2",
    "VR3",
]

FEATURE_NAMES = [f"{attr}_Dz{prop}" for prop in PROPERTY_FUNCS for attr in _ATTR_NAMES]


def calc(mol_regular: Mol, props_regular: AtomicProperties, n_frags: int) -> np.ndarray:
    """
    Barysz matrix spectral descriptors.

    Constructs a weighted distance matrix where bond weights are inversely
    proportional to atomic properties and bond order, normalized by the
    corresponding carbon-carbon value. Spectral attributes of the resulting
    matrix are computed for each atomic property.

    Requires a connected molecule (single fragment).
    """
    if n_frags != 1:
        values_nan = np.full(
            len(PROPERTY_FUNCS) * len(_ATTR_NAMES), np.nan, dtype=np.float32
        )

        return values_nan

    values: list = []
    for prop_func in PROPERTY_FUNCS.values():
        matrix = _barysz_matrix(mol_regular, prop_func)
        if matrix is None:
            values.extend([np.nan] * len(_ATTR_NAMES))
        else:
            values.extend(
                _barysz_matrix_attribute_values(props_regular, n_frags, matrix)
            )

    return np.asarray(values, dtype=np.float32)


@np.errstate(divide="ignore", invalid="ignore")
def _barysz_matrix(mol: Mol, prop_func) -> np.ndarray | None:
    carbon_value = prop_func(Atom(6))  # Carbon

    property_values = np.asarray(
        [prop_func(atom) for atom in mol.GetAtoms()], dtype=np.float32
    )
    if np.any(~np.isfinite(property_values)):
        return None

    n_atoms = mol.GetNumAtoms()
    matrix = np.full((n_atoms, n_atoms), np.inf, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)

    bonds = mol.GetBonds()
    if bonds:
        i_arr = np.array([b.GetBeginAtomIdx() for b in bonds])
        j_arr = np.array([b.GetEndAtomIdx() for b in bonds])
        bo_arr = np.array([b.GetBondTypeAsDouble() for b in bonds])
        weights = carbon_value**2 / (
            property_values[i_arr] * property_values[j_arr] * bo_arr
        )
        if not np.all(np.isfinite(weights)):
            return None
        matrix[i_arr, j_arr] = weights
        matrix[j_arr, i_arr] = weights

    matrix = floyd_warshall(matrix, directed=False)
    diagonal = 1.0 - carbon_value / property_values
    if np.any(~np.isfinite(diagonal)):
        return None

    np.fill_diagonal(matrix, diagonal)
    return matrix


def _barysz_matrix_attribute_values(
    props: AtomicProperties, n_frags: int, matrix: np.ndarray
) -> list[float | np.floating]:
    attrs = MatrixAttributes(matrix, props, hermitian=True, n_frags=n_frags)
    return [
        attrs.graph_energy,
        attrs.leading_eigenvalue,
        attrs.spectral_diameter,
        attrs.sp_ad,
        attrs.sp_mad,
        attrs.log_ee,
        attrs.sm1,
        attrs.ve1,
        attrs.ve2,
        attrs.ve3,
        attrs.vr1,
        attrs.vr2,
        attrs.vr3,
    ]
