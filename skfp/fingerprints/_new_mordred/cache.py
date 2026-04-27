"""
Per-molecule dependency cache for New Mordred descriptors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import GetMolFrags, Mol, rdPartialCharges
from rdkit.Chem.rdchem import Atom, Bond, BondType
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
BCUT_PROPERTIES = ["c", "dv", "d", "s", "Z", "m", "v", "se", "pe", "are", "p", "i"]
CHI_FEATURE_NAMES = [
    "Xch-3d",
    "Xch-4d",
    "Xch-5d",
    "Xch-6d",
    "Xch-7d",
    "Xch-3dv",
    "Xch-4dv",
    "Xch-5dv",
    "Xch-6dv",
    "Xch-7dv",
    "Xc-3d",
    "Xc-4d",
    "Xc-5d",
    "Xc-6d",
    "Xc-3dv",
    "Xc-4dv",
    "Xc-5dv",
    "Xc-6dv",
    "Xpc-4d",
    "Xpc-5d",
    "Xpc-6d",
    "Xpc-4dv",
    "Xpc-5dv",
    "Xpc-6dv",
    "Xp-0d",
    "Xp-1d",
    "Xp-2d",
    "Xp-3d",
    "Xp-4d",
    "Xp-5d",
    "Xp-6d",
    "Xp-7d",
    "AXp-0d",
    "AXp-1d",
    "AXp-2d",
    "AXp-3d",
    "AXp-4d",
    "AXp-5d",
    "AXp-6d",
    "AXp-7d",
    "Xp-0dv",
    "Xp-1dv",
    "Xp-2dv",
    "Xp-3dv",
    "Xp-4dv",
    "Xp-5dv",
    "Xp-6dv",
    "Xp-7dv",
    "AXp-0dv",
    "AXp-1dv",
    "AXp-2dv",
    "AXp-3dv",
    "AXp-4dv",
    "AXp-5dv",
    "AXp-6dv",
    "AXp-7dv",
]
_CHI_TYPES = ("chain", "path", "path_cluster", "cluster")
_CHI_PREFIX_TO_TYPE = {
    "Xch": "chain",
    "Xp": "path",
    "AXp": "path",
    "Xpc": "path_cluster",
    "Xc": "cluster",
}
CONSTITUTIONAL_PROPERTIES = ["Z", "m", "v", "se", "pe", "are", "p", "i"]
CONSTITUTIONAL_FEATURE_NAMES = [
    *[f"S{prop}" for prop in CONSTITUTIONAL_PROPERTIES],
    *[f"M{prop}" for prop in CONSTITUTIONAL_PROPERTIES],
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
    return {
        prop: values - values.mean() if len(values) else values.copy()
        for prop, values in weights.items()
    }


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


def _bcut_pair(mol: Mol, property_values: np.ndarray) -> list[float]:
    if np.any(~np.isfinite(property_values)):
        return [np.nan, np.nan]

    n_atoms = mol.GetNumAtoms()
    burden_matrix = np.full((n_atoms, n_atoms), 0.001, dtype=float)
    np.fill_diagonal(burden_matrix, property_values)

    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        i = begin_atom.GetIdx()
        j = end_atom.GetIdx()
        weight = bond.GetBondTypeAsDouble() / 10.0
        if begin_atom.GetDegree() == 1 or end_atom.GetDegree() == 1:
            weight += 0.01
        burden_matrix[i, j] = weight
        burden_matrix[j, i] = weight

    eigenvalues = np.linalg.eigvalsh(burden_matrix)
    return [float(eigenvalues[-1]), float(eigenvalues[0])]


def _bcut_values(mol: Mol, n_frags: int) -> np.ndarray:
    if n_frags != 1:
        return np.full(len(BCUT_PROPERTIES) * 2, np.nan, dtype=np.float32)

    values: list[float] = []
    for prop in BCUT_PROPERTIES:
        values.extend(_bcut_pair(mol, _property_values(mol, prop)))

    return np.asarray(values, dtype=np.float32)


def _is_aromatic_bond(bond: Bond) -> bool:
    return bond.GetIsAromatic() or bond.GetBondType() == BondType.AROMATIC


def _bond_count_values(mol_regular: Mol, mol_kekulized: Mol) -> np.ndarray:
    bonds_regular = mol_regular.GetBonds()
    bonds_kekulized = mol_kekulized.GetBonds()

    n_bonds = mol_regular.GetNumBonds()
    n_bonds_s = 0
    n_bonds_d = 0
    n_bonds_t = 0
    n_bonds_a = 0
    n_bonds_m = 0

    for bond in bonds_regular:
        bond_type = bond.GetBondType()
        is_aromatic = _is_aromatic_bond(bond)

        n_bonds_s += bond_type == BondType.SINGLE
        n_bonds_d += bond_type == BondType.DOUBLE
        n_bonds_t += bond_type == BondType.TRIPLE
        n_bonds_a += is_aromatic
        n_bonds_m += is_aromatic or bond_type != BondType.SINGLE

    n_bonds_ks = 0
    n_bonds_kd = 0
    for bond in bonds_kekulized:
        bond_type = bond.GetBondType()
        n_bonds_ks += bond_type == BondType.SINGLE
        n_bonds_kd += bond_type == BondType.DOUBLE

    return np.asarray(
        [
            n_bonds,
            n_bonds,
            n_bonds_s,
            n_bonds_d,
            n_bonds_t,
            n_bonds_a,
            n_bonds_m,
            n_bonds_ks,
            n_bonds_kd,
        ],
        dtype=np.float32,
    )


def _chi_values(mol: Mol) -> np.ndarray:
    properties = {
        "d": np.asarray(
            [get_sigma_electrons(atom) for atom in mol.GetAtoms()],
            dtype=float,
        ),
        "dv": np.asarray(
            [get_valence_electrons(atom) for atom in mol.GetAtoms()],
            dtype=float,
        ),
    }
    subgraphs_by_order = {order: _chi_subgraphs(mol, order) for order in range(1, 8)}

    values = []
    for name in CHI_FEATURE_NAMES:
        chi_type, order, prop, averaged = _parse_chi_feature_name(name)
        if order == 0:
            node_sets = [[atom.GetIdx()] for atom in mol.GetAtoms()]
        else:
            node_sets = subgraphs_by_order[order][chi_type]
        values.append(_chi_value(node_sets, properties[prop], averaged))

    return np.asarray(values, dtype=np.float32)


def _chi_subgraphs(mol: Mol, order: int) -> dict[str, list[list[int]]]:
    classified: dict[str, list[list[int]]] = {chi_type: [] for chi_type in _CHI_TYPES}
    bond_endpoints = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    ]

    for bond_idxs in Chem.FindAllSubgraphsOfLengthN(mol, order):
        neighbors = _subgraph_neighbors(bond_idxs, bond_endpoints)
        chi_type = _classify_neighbors(neighbors)
        classified[chi_type].append(list(neighbors.keys()))

    return classified


def _subgraph_neighbors(
    bond_idxs: Sequence[int], bond_endpoints: Sequence[tuple[int, int]]
) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for bond_idx in bond_idxs:
        begin_idx, end_idx = bond_endpoints[bond_idx]
        neighbors[begin_idx].add(end_idx)
        neighbors[end_idx].add(begin_idx)
    return neighbors


def _classify_neighbors(neighbors: dict[int, set[int]]) -> str:
    visited: set[int] = set()
    visited_edges: set[tuple[int, int]] = set()
    degrees: set[int] = set()
    is_chain = _depth_first_search(
        next(iter(neighbors.keys())),
        neighbors,
        visited,
        visited_edges,
        degrees,
    )

    if is_chain:
        return "chain"
    if not degrees - {1, 2}:
        return "path"
    if 2 in degrees:
        return "path_cluster"
    return "cluster"


def _depth_first_search(
    node: int,
    neighbors: dict[int, set[int]],
    visited: set[int],
    visited_edges: set[tuple[int, int]],
    degrees: set[int],
) -> bool:
    visited.add(node)
    degrees.add(len(neighbors[node]))
    saw_cycle = False

    for neighbor in neighbors[node]:
        edge = (neighbor, node) if node > neighbor else (node, neighbor)

        if neighbor not in visited:
            visited_edges.add(edge)
            if _depth_first_search(
                neighbor, neighbors, visited, visited_edges, degrees
            ):
                saw_cycle = True
        elif edge not in visited_edges:
            visited_edges.add(edge)
            saw_cycle = True

    return saw_cycle


def _parse_chi_feature_name(name: str) -> tuple[str, int, str, bool]:
    prefix, order_and_prop = name.split("-", maxsplit=1)
    averaged = prefix.startswith("A")
    chi_type = _CHI_PREFIX_TO_TYPE[prefix]
    order = int(order_and_prop[0])
    prop = order_and_prop[1:]
    return chi_type, order, prop, averaged


def _chi_value(
    node_sets: Sequence[Sequence[int] | set[int]],
    prop_values: np.ndarray,
    averaged: bool,
) -> float:
    if averaged and len(node_sets) == 0:
        return np.nan

    value = 0.0
    for nodes in node_sets:
        product = 1.0
        for node in nodes:
            product *= prop_values[node]

        if product <= 0:
            return np.nan

        value += product**-0.5

    if averaged:
        value /= len(node_sets)

    return value


def _constitutional_values(mol: Mol) -> np.ndarray:
    sums: list[float] = []
    for prop in CONSTITUTIONAL_PROPERTIES:
        carbon_value = _AUTOCORRELATION_PROPERTY_FUNCS[prop](_CARBON)
        sums.append(float(np.sum(_property_values(mol, prop) / carbon_value)))

    n_atoms = mol.GetNumAtoms()
    means = (
        [np.nan] * len(sums) if n_atoms == 0 else [value / n_atoms for value in sums]
    )

    return np.asarray([*sums, *means], dtype=np.float32)


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
    bcut_values: np.ndarray
    bond_count_values: np.ndarray
    chi_values: np.ndarray
    constitutional_values: np.ndarray
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
            bcut_values=_bcut_values(mol_regular, n_frags),
            bond_count_values=_bond_count_values(mol_regular, mol_kekulized),
            chi_values=_chi_values(mol_regular),
            constitutional_values=_constitutional_values(mol_regular),
            mol_with_hydrogens=mol_with_hydrogens,
        )
