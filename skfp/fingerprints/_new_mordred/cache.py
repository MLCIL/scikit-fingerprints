"""
Per-molecule dependency cache for New Mordred descriptors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from rdkit.Chem import GetMolFrags, Mol, rdPartialCharges
from rdkit.Chem.rdchem import Atom
from scipy.sparse.csgraph import floyd_warshall

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_allred_rocow_en,
    get_gasteiger_charge,
    get_intrinsic_state,
    get_ionization_potential,
    get_mass,
    get_pauling_en,
    get_polarizability,
    get_sanderson_en,
    get_sigma_electrons,
    get_valence_electrons,
    get_vdw_volume,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

ADJACENCY_MATRIX_FEATURE_NAMES = [
    "SpAbs_A",
    "SpMax_A",
    "SpDiam_A",
    "SpAD_A",
    "SpMAD_A",
    "LogEE_A",
    "VE1_A",
    "VE2_A",
    "VE3_A",
    "VR1_A",
    "VR2_A",
    "VR3_A",
]
AROMATIC_FEATURE_NAMES = ["nAromAtom", "nAromBond"]
AUTOCORRELATION_MAX_DISTANCE = 8
AUTOCORRELATION_ATS_PROPERTIES = [
    "dv",
    "d",
    "s",
    "Z",
    "m",
    "v",
    "se",
    "pe",
    "are",
    "p",
    "i",
]
AUTOCORRELATION_ALL_PROPERTIES = ["c", *AUTOCORRELATION_ATS_PROPERTIES]
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


def _get_atomic_number(atom: Atom) -> int:
    return atom.GetAtomicNum()


_AUTOCORRELATION_PROPERTY_FUNCS: dict[str, Callable[[Atom], float]] = {
    "c": get_gasteiger_charge,
    "dv": get_valence_electrons,
    "d": get_sigma_electrons,
    "s": get_intrinsic_state,
    "Z": _get_atomic_number,
    "m": get_mass,
    "v": get_vdw_volume,
    "se": get_sanderson_en,
    "pe": get_pauling_en,
    "are": get_allred_rocow_en,
    "p": get_polarizability,
    "i": get_ionization_potential,
}


def _adjacency_matrix_values(
    mol: Mol, n_frags: int, adjacency_matrix: AdjacencyMatrix
) -> np.ndarray:
    if n_frags != 1:
        return np.full(len(ADJACENCY_MATRIX_FEATURE_NAMES), np.nan, dtype=np.float32)

    attrs = MatrixAttributes(
        adjacency_matrix.order(),
        mol,
        hermitian=adjacency_matrix.hermitian,
        n_frags=n_frags,
    )
    return np.asarray(
        [
            attrs.graph_energy,
            attrs.leading_eigenvalue,
            attrs.spectral_diameter,
            attrs.sp_ad,
            attrs.sp_mad,
            attrs.log_ee,
            attrs.ve1,
            attrs.ve2,
            attrs.ve3,
            attrs.vr1,
            attrs.vr2,
            attrs.vr3,
        ],
        dtype=np.float32,
    )


def _aromatic_values(mol: Mol) -> np.ndarray:
    return np.asarray(
        [
            sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()),
            sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic()),
        ],
        dtype=np.float32,
    )


def _autocorrelation_property_vector(mol: Mol, prop: str) -> np.ndarray:
    return np.asarray(
        [_AUTOCORRELATION_PROPERTY_FUNCS[prop](atom) for atom in mol.GetAtoms()]
    )


def _autocorrelation_gmats(
    distance_matrix: DistanceMatrix,
) -> list[np.ndarray]:
    return [
        distance_matrix.matrix == order
        for order in range(AUTOCORRELATION_MAX_DISTANCE + 1)
    ]


def _autocorrelation_gsums(gmats: list[np.ndarray]) -> list[float]:
    return [
        float(gmat.sum() if order == 0 else 0.5 * gmat.sum())
        for order, gmat in enumerate(gmats)
    ]


def _autocorrelation_weights(mol: Mol) -> dict[str, np.ndarray]:
    return {
        prop: _autocorrelation_property_vector(mol, prop)
        for prop in AUTOCORRELATION_ALL_PROPERTIES
    }


def _centered_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {prop: values - values.mean() for prop, values in weights.items()}


def _property_values(mol: Mol, prop: str) -> np.ndarray:
    prop_func = _AUTOCORRELATION_PROPERTY_FUNCS[prop]
    return np.asarray([prop_func(atom) for atom in mol.GetAtoms()], dtype=float)


def _barysz_matrix(mol: Mol, prop: str) -> np.ndarray | None:
    property_values = _property_values(mol, prop)
    if np.any(~np.isfinite(property_values)):
        return None

    prop_func = _AUTOCORRELATION_PROPERTY_FUNCS[prop]
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
) -> list[float]:
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

    values: list[float] = []
    for prop in BARYSZ_PROPERTIES:
        matrix = _barysz_matrix(mol, prop)
        if matrix is None:
            values.extend([np.nan] * len(BARYSZ_ATTRIBUTES))
        else:
            values.extend(_barysz_matrix_attribute_values(mol, matrix, n_frags))

    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class MordredMolCache:
    """
    Eager per-molecule dependencies shared by New Mordred descriptors.
    """

    original_mol: Mol
    use_3d: bool
    n_frags: int
    mol_regular: Mol
    mol_kekulized: Mol
    distance_matrix_regular: DistanceMatrix
    adjacency_matrix_regular: AdjacencyMatrix
    adjacency_matrix_values: np.ndarray
    aromatic_values: np.ndarray
    autocorrelation_gmats: list[np.ndarray]
    autocorrelation_gsums: list[float]
    autocorrelation_weights: dict[str, np.ndarray]
    autocorrelation_centered_weights: dict[str, np.ndarray]
    barysz_values: np.ndarray
    mol_with_hydrogens: Mol | None

    @classmethod
    def from_mol(cls, mol: Mol, use_3D: bool) -> MordredMolCache:
        mol_regular = preprocess_mol(mol)
        mol_kekulized = preprocess_mol(mol, kekulize=True)
        mol_with_hydrogens = (
            preprocess_mol(mol, explicit_hydrogens=True) if use_3D else None
        )
        n_frags = len(GetMolFrags(mol))
        distance_matrix_regular = DistanceMatrix(mol_regular)
        adjacency_matrix_regular = AdjacencyMatrix(mol_regular)
        autocorrelation_gmats = _autocorrelation_gmats(distance_matrix_regular)
        rdPartialCharges.ComputeGasteigerCharges(mol_regular)
        autocorrelation_weights = _autocorrelation_weights(mol_regular)
        return cls(
            original_mol=mol,
            use_3d=use_3D,
            n_frags=n_frags,
            mol_regular=mol_regular,
            mol_kekulized=mol_kekulized,
            distance_matrix_regular=distance_matrix_regular,
            adjacency_matrix_regular=adjacency_matrix_regular,
            adjacency_matrix_values=_adjacency_matrix_values(
                mol_regular, n_frags, adjacency_matrix_regular
            ),
            aromatic_values=_aromatic_values(mol_regular),
            autocorrelation_gmats=autocorrelation_gmats,
            autocorrelation_gsums=_autocorrelation_gsums(autocorrelation_gmats),
            autocorrelation_weights=autocorrelation_weights,
            autocorrelation_centered_weights=_centered_weights(
                autocorrelation_weights
            ),
            barysz_values=_barysz_values(mol_regular, n_frags),
            mol_with_hydrogens=mol_with_hydrogens,
        )
