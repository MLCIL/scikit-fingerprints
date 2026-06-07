import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom
from scipy.sparse.csgraph import floyd_warshall

from skfp.fingerprints._new_mordred.utils.atomic_properties import PROPERTY_FUNCS
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

BARYSZ_PROPERTIES = ["Z", "m", "v", "se", "pe", "are", "p", "i"]

BARYSZ_ATTRIBUTES = [
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

_CARBON = Atom(6)

FEATURE_NAMES = [
    f"{attr}_Dz{prop}" for prop in BARYSZ_PROPERTIES for attr in BARYSZ_ATTRIBUTES
]


def calc(mol_regular: Mol, n_frags: int) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred Barysz matrix spectral descriptors.
    """
    return _barysz_values(mol_regular, n_frags), FEATURE_NAMES


def _property_values(mol: Mol, prop: str) -> np.ndarray:
    prop_func = PROPERTY_FUNCS[prop]
    return np.asarray([prop_func(atom) for atom in mol.GetAtoms()], dtype=float)


def _barysz_matrix(mol: Mol, prop: str) -> np.ndarray | None:
    property_values = _property_values(mol, prop)
    if np.any(~np.isfinite(property_values)):
        return None

    prop_func = PROPERTY_FUNCS[prop]
    carbon_value = prop_func(_CARBON)
    n_atoms = mol.GetNumAtoms()

    matrix = np.full((n_atoms, n_atoms), np.inf, dtype=float)
    np.fill_diagonal(matrix, 0.0)

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_order = bond.GetBondTypeAsDouble()
        denominator = property_values[i] * property_values[j] * bond_order
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = float(np.divide(carbon_value * carbon_value, denominator))
        if not np.isfinite(weight):
            return None

        matrix[i, j] = weight
        matrix[j, i] = weight

    matrix = floyd_warshall(matrix, directed=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        diagonal = 1.0 - carbon_value / property_values
    if np.any(~np.isfinite(diagonal)):
        return None

    np.fill_diagonal(matrix, diagonal)
    return matrix


def _barysz_matrix_attribute_values(
    mol: Mol, matrix: np.ndarray, n_frags: int
) -> list[float | np.floating]:
    attrs = MatrixAttributes(matrix, mol, hermitian=True, n_frags=n_frags)
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


def _barysz_values(mol: Mol, n_frags: int) -> np.ndarray:
    if n_frags != 1:
        return np.full(
            len(BARYSZ_PROPERTIES) * len(BARYSZ_ATTRIBUTES), np.nan, dtype=np.float32
        )

    values: list[float | np.floating] = []
    for prop in BARYSZ_PROPERTIES:
        matrix = _barysz_matrix(mol, prop)
        if matrix is None:
            values.extend([np.nan] * len(BARYSZ_ATTRIBUTES))
        else:
            values.extend(_barysz_matrix_attribute_values(mol, matrix, n_frags))

    return np.asarray(values, dtype=np.float32)
