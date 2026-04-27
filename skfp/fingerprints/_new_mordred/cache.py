"""
Per-molecule dependency cache for New Mordred descriptors.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import groupby
from math import log, sqrt
from time import monotonic
from typing import Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    EState,
    Get3DDistanceMatrix,
    GetMolFrags,
    Mol,
    MolFromSmarts,
    rdMolDescriptors,
    rdPartialCharges,
)
from rdkit.Chem.rdchem import Atom, Bond, BondType
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import cdist

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_allred_rocow_en,
    get_core_count,
    get_eta_beta_delta,
    get_eta_beta_non_sigma,
    get_eta_beta_sigma,
    get_eta_epsilon,
    get_gasteiger_charge,
    get_intrinsic_state,
    get_ionization_potential,
    get_mass,
    get_mc_gowan_volume,
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
from skfp.fingerprints._new_mordred.utils.periodic_table import MORDRED_VDW_RADII

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
CPSA_FEATURE_NAMES = [
    "PNSA1",
    "PNSA2",
    "PNSA3",
    "PNSA4",
    "PNSA5",
    "PPSA1",
    "PPSA2",
    "PPSA3",
    "PPSA4",
    "PPSA5",
    "DPSA1",
    "DPSA2",
    "DPSA3",
    "DPSA4",
    "DPSA5",
    "FNSA1",
    "FNSA2",
    "FNSA3",
    "FNSA4",
    "FNSA5",
    "FPSA1",
    "FPSA2",
    "FPSA3",
    "FPSA4",
    "FPSA5",
    "WNSA1",
    "WNSA2",
    "WNSA3",
    "WNSA4",
    "WNSA5",
    "WPSA1",
    "WPSA2",
    "WPSA3",
    "WPSA4",
    "WPSA5",
    "RNCG",
    "RPCG",
    "RNCS",
    "RPCS",
    "TASA",
    "TPSA",
    "RASA",
    "RPSA",
]
CPSA_FEATURE_NAMES_2D = ["RNCG", "RPCG"]
CPSA_FEATURE_NAMES_3D = [
    name for name in CPSA_FEATURE_NAMES if name not in CPSA_FEATURE_NAMES_2D
]
DETOUR_MATRIX_FEATURE_NAMES = [
    "SpAbs_Dt",
    "SpMax_Dt",
    "SpDiam_Dt",
    "SpAD_Dt",
    "SpMAD_Dt",
    "LogEE_Dt",
    "SM1_Dt",
    "VE1_Dt",
    "VE2_Dt",
    "VE3_Dt",
    "VR1_Dt",
    "VR2_Dt",
    "VR3_Dt",
    "DetourIndex",
]
DISTANCE_MATRIX_FEATURE_NAMES = [
    "SpAbs_D",
    "SpMax_D",
    "SpDiam_D",
    "SpAD_D",
    "SpMAD_D",
    "LogEE_D",
    "VE1_D",
    "VE2_D",
    "VE3_D",
    "VR1_D",
    "VR2_D",
    "VR3_D",
]
ECCENTRIC_CONNECTIVITY_INDEX_FEATURE_NAMES = ["ECIndex"]
ESTATE_ATOM_TYPES = [
    "sLi",
    "ssBe",
    "ssssBe",
    "ssBH",
    "sssB",
    "ssssB",
    "sCH3",
    "dCH2",
    "ssCH2",
    "tCH",
    "dsCH",
    "aaCH",
    "sssCH",
    "ddC",
    "tsC",
    "dssC",
    "aasC",
    "aaaC",
    "ssssC",
    "sNH3",
    "sNH2",
    "ssNH2",
    "dNH",
    "ssNH",
    "aaNH",
    "tN",
    "sssNH",
    "dsN",
    "aaN",
    "sssN",
    "ddsN",
    "aasN",
    "ssssN",
    "sOH",
    "dO",
    "ssO",
    "aaO",
    "sF",
    "sSiH3",
    "ssSiH2",
    "sssSiH",
    "ssssSi",
    "sPH2",
    "ssPH",
    "sssP",
    "dsssP",
    "sssssP",
    "sSH",
    "dS",
    "ssS",
    "aaS",
    "dssS",
    "ddssS",
    "sCl",
    "sGeH3",
    "ssGeH2",
    "sssGeH",
    "ssssGe",
    "sAsH2",
    "ssAsH",
    "sssAs",
    "sssdAs",
    "sssssAs",
    "sSeH",
    "dSe",
    "ssSe",
    "aaSe",
    "dssSe",
    "ddssSe",
    "sBr",
    "sSnH3",
    "ssSnH2",
    "sssSnH",
    "ssssSn",
    "sI",
    "sPbH3",
    "ssPbH2",
    "sssPbH",
    "ssssPb",
]
ESTATE_FEATURE_NAMES = [
    f"{prefix}{atom_type}"
    for prefix in ("N", "S", "MAX", "MIN")
    for atom_type in ESTATE_ATOM_TYPES
]
_ESTATE_ATOM_TYPE_TO_IDX = {
    atom_type: idx for idx, atom_type in enumerate(ESTATE_ATOM_TYPES)
}
EXTENDED_TOPOCHEMICAL_ATOM_FEATURE_NAMES = [
    "ETA_alpha",
    "AETA_alpha",
    "ETA_shape_p",
    "ETA_shape_y",
    "ETA_shape_x",
    "ETA_beta",
    "AETA_beta",
    "ETA_beta_s",
    "AETA_beta_s",
    "ETA_beta_ns",
    "AETA_beta_ns",
    "ETA_beta_ns_d",
    "AETA_beta_ns_d",
    "ETA_eta",
    "AETA_eta",
    "ETA_eta_L",
    "AETA_eta_L",
    "ETA_eta_R",
    "AETA_eta_R",
    "ETA_eta_RL",
    "AETA_eta_RL",
    "ETA_eta_F",
    "AETA_eta_F",
    "ETA_eta_FL",
    "AETA_eta_FL",
    "ETA_eta_B",
    "AETA_eta_B",
    "ETA_eta_BR",
    "AETA_eta_BR",
    "ETA_dAlpha_A",
    "ETA_dAlpha_B",
    "ETA_epsilon_1",
    "ETA_epsilon_2",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
    "ETA_dBeta",
    "AETA_dBeta",
    "ETA_psi_1",
    "ETA_dPsi_A",
    "ETA_dPsi_B",
]
FRAGMENT_COMPLEXITY_FEATURE_NAMES = ["fragCpx"]
FRAMEWORK_FEATURE_NAMES = ["fMF"]
GEOMETRICAL_INDEX_FEATURE_NAMES = [
    "GeomDiameter",
    "GeomRadius",
    "GeomShapeIndex",
    "GeomPetitjeanIndex",
]
GRAVITATIONAL_INDEX_FEATURE_NAMES = ["GRAV", "GRAVH", "GRAVp", "GRAVHp"]
_INFORMATION_CONTENT_ORDERS = range(6)
_INFORMATION_CONTENT_PREFIXES = ("IC", "TIC", "SIC", "BIC", "CIC", "MIC", "ZMIC")
INFORMATION_CONTENT_FEATURE_NAMES = [
    f"{prefix}{order}"
    for prefix in _INFORMATION_CONTENT_PREFIXES
    for order in _INFORMATION_CONTENT_ORDERS
]
KAPPA_SHAPE_INDEX_FEATURE_NAMES = ["Kier1", "Kier2", "Kier3"]
LIPINSKI_FEATURE_NAMES = ["Lipinski", "GhoseFilter"]
LOGS_FEATURE_NAMES = ["FilterItLogS"]
_LOGS_SMARTS_CONTRIBUTIONS = [
    ("[NH0;X3;v3]", 0.71535),
    ("[NH2;X3;v3]", 0.41056),
    ("[nH0;X3]", 0.82535),
    ("[OH0;X2;v2]", 0.31464),
    ("[OH0;X1;v2]", 0.14787),
    ("[OH1;X2;v2]", 0.62998),
    ("[CH2;!R]", -0.35634),
    ("[CH3;!R]", -0.33888),
    ("[CH0;R]", -0.21912),
    ("[CH2;R]", -0.23057),
    ("[ch0]", -0.37570),
    ("[ch1]", -0.22435),
    ("F", -0.21728),
    ("Cl", -0.49721),
    ("Br", -0.57982),
    ("I", -0.51547),
]
_LOGS_PATTERN_CONTRIBUTIONS = [
    (MolFromSmarts(smarts), contribution)
    for smarts, contribution in _LOGS_SMARTS_CONTRIBUTIONS
]
MCGOWAN_VOLUME_FEATURE_NAMES = ["VMcGowan"]
MOLECULAR_DISTANCE_EDGE_FEATURE_NAMES = [
    "MDEC-11",
    "MDEC-12",
    "MDEC-13",
    "MDEC-14",
    "MDEC-22",
    "MDEC-23",
    "MDEC-24",
    "MDEC-33",
    "MDEC-34",
    "MDEC-44",
    "MDEO-11",
    "MDEO-12",
    "MDEO-22",
    "MDEN-11",
    "MDEN-12",
    "MDEN-13",
    "MDEN-22",
    "MDEN-23",
    "MDEN-33",
]
_MOLECULAR_DISTANCE_EDGE_FEATURES = [
    (6, 1, 1),
    (6, 1, 2),
    (6, 1, 3),
    (6, 1, 4),
    (6, 2, 2),
    (6, 2, 3),
    (6, 2, 4),
    (6, 3, 3),
    (6, 3, 4),
    (6, 4, 4),
    (8, 1, 1),
    (8, 1, 2),
    (8, 2, 2),
    (7, 1, 1),
    (7, 1, 2),
    (7, 1, 3),
    (7, 2, 2),
    (7, 2, 3),
    (7, 3, 3),
]
MOLECULAR_ID_FEATURE_NAMES = [
    "MID",
    "AMID",
    "MID_h",
    "AMID_h",
    "MID_C",
    "AMID_C",
    "MID_N",
    "AMID_N",
    "MID_O",
    "AMID_O",
    "MID_X",
    "AMID_X",
]

PATH_COUNT_FEATURE_NAMES = [
    "MPC2",
    "MPC3",
    "MPC4",
    "MPC5",
    "MPC6",
    "MPC7",
    "MPC8",
    "MPC9",
    "MPC10",
    "TMPC10",
    "piPC1",
    "piPC2",
    "piPC3",
    "piPC4",
    "piPC5",
    "piPC6",
    "piPC7",
    "piPC8",
    "piPC9",
    "piPC10",
    "TpiPC10",
]
_MOLECULAR_ID_EPS = 1e-10
_MOLECULAR_ID_WEIGHT_LIMIT = int(1.0 / (_MOLECULAR_ID_EPS**2))
_MOLECULAR_ID_HALOGENS = {9, 17, 35, 53, 85, 117}
MORSE_MAX_DISTANCE = 32
MORSE_PROPERTIES = ["", "m", "v", "se", "p"]
MORSE_FEATURE_NAMES = [
    f"Mor{distance:02d}{prop}"
    for prop in MORSE_PROPERTIES
    for distance in range(1, MORSE_MAX_DISTANCE + 1)
]
_CARBON = Atom(6)
_FrameworkNode = tuple[str, int]
_SPHERE_MESH_CACHE: dict[int, np.ndarray] = {}


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


def _distance_matrix_values(
    mol: Mol, n_frags: int, distance_matrix: DistanceMatrix
) -> np.ndarray:
    if n_frags != 1:
        return np.full(len(DISTANCE_MATRIX_FEATURE_NAMES), np.nan, dtype=np.float32)

    attrs = MatrixAttributes(
        distance_matrix.matrix,
        mol,
        hermitian=distance_matrix.hermitian,
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


def _eccentric_connectivity_index_values(
    distance_matrix: DistanceMatrix, adjacency_matrix: AdjacencyMatrix
) -> np.ndarray:
    if distance_matrix.matrix.size == 0:
        return np.asarray([np.nan], dtype=np.float32)

    eccentricity = distance_matrix.matrix.max(axis=0)
    vertex_degree = adjacency_matrix.degree
    value = int((eccentricity.astype("int") * vertex_degree).sum())
    return np.asarray([value], dtype=np.float32)


def _estate_values(mol: Mol) -> np.ndarray:
    atom_types_by_atom = EState.TypeAtoms(mol)
    estate_indices = EState.EStateIndices(mol)

    n_atom_types = len(ESTATE_ATOM_TYPES)
    counts = np.zeros(n_atom_types, dtype=np.float64)
    sums = np.zeros(n_atom_types, dtype=np.float64)
    max_values = np.full(n_atom_types, -np.inf, dtype=np.float64)
    min_values = np.full(n_atom_types, np.inf, dtype=np.float64)
    has_value = np.zeros(n_atom_types, dtype=bool)

    for atom_types, estate_idx in zip(atom_types_by_atom, estate_indices, strict=True):
        estate_value = float(estate_idx)
        for atom_type in atom_types:
            type_idx = _ESTATE_ATOM_TYPE_TO_IDX[atom_type]
            counts[type_idx] += 1
            sums[type_idx] += estate_value
            if has_value[type_idx]:
                max_values[type_idx] = max(max_values[type_idx], estate_value)
                min_values[type_idx] = min(min_values[type_idx], estate_value)
            else:
                max_values[type_idx] = estate_value
                min_values[type_idx] = estate_value
                has_value[type_idx] = True

    max_values[~has_value] = np.nan
    min_values[~has_value] = np.nan

    return np.concatenate((counts, sums, max_values, min_values)).astype(
        np.float32, copy=False
    )


@dataclass(frozen=True, slots=True)
class _EtaData:
    mol: Mol
    n_atoms: int
    alpha: np.ndarray
    epsilon: np.ndarray
    beta_sigma: np.ndarray
    beta_non_sigma: np.ndarray
    beta_delta: np.ndarray
    gamma: np.ndarray


def _eta_data(mol: Mol) -> _EtaData:
    alpha = np.asarray([get_core_count(atom) for atom in mol.GetAtoms()], dtype=float)
    epsilon = np.asarray(
        [get_eta_epsilon(atom) for atom in mol.GetAtoms()], dtype=float
    )
    beta_sigma = np.asarray(
        [get_eta_beta_sigma(atom) for atom in mol.GetAtoms()], dtype=float
    )
    beta_non_sigma = np.asarray(
        [get_eta_beta_non_sigma(atom) for atom in mol.GetAtoms()], dtype=float
    )
    beta_delta = np.asarray(
        [get_eta_beta_delta(atom) for atom in mol.GetAtoms()], dtype=float
    )
    beta = beta_sigma + beta_non_sigma + beta_delta
    gamma = np.divide(
        alpha,
        beta,
        out=np.full(mol.GetNumAtoms(), np.nan, dtype=float),
        where=beta != 0,
    )
    return _EtaData(
        mol=mol,
        n_atoms=mol.GetNumAtoms(),
        alpha=alpha,
        epsilon=epsilon,
        beta_sigma=beta_sigma,
        beta_non_sigma=beta_non_sigma,
        beta_delta=beta_delta,
        gamma=gamma,
    )


def _alter_molecule(mol: Mol, saturated: bool = False) -> Mol:
    new = Chem.RWMol(Chem.Mol())
    atom_idxs: dict[int, int] = {}

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if saturated:
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
        else:
            new_atom = Chem.Atom(6)
        atom_idxs[atom.GetIdx()] = new.AddAtom(new_atom)

    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if not saturated and (begin.GetDegree() > 4 or end.GetDegree() > 4):
            raise ValueError("bond degree greater than 4")
        begin_idx = atom_idxs.get(begin.GetIdx())
        end_idx = atom_idxs.get(end.GetIdx())
        if begin_idx is None or end_idx is None:
            continue
        bond_type = (
            bond.GetBondType()
            if saturated and (begin.GetAtomicNum() != 6 or end.GetAtomicNum() != 6)
            else Chem.BondType.SINGLE
        )
        new.AddBond(begin_idx, end_idx, bond_type)

    new_mol = Chem.Mol(new)
    if Chem.SanitizeMol(new_mol, catchErrors=True) != 0:
        raise ValueError("cannot sanitize altered molecule")
    Chem.Kekulize(new_mol)
    return new_mol


def _safe_eta_data(mol: Mol) -> _EtaData | None:
    try:
        return _eta_data(mol)
    except (ValueError, ZeroDivisionError):
        return None


def _safe_altered_data(mol: Mol, saturated: bool) -> _EtaData | None:
    try:
        return _eta_data(_alter_molecule(mol, saturated=saturated))
    except (ValueError, ZeroDivisionError):
        return None


def _eta_safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def _eta_alpha(data: _EtaData) -> float:
    return float(data.alpha.sum())


def _eta_average(value: float, data: _EtaData | None) -> float:
    if data is None:
        return np.nan
    return _eta_safe_divide(value, data.n_atoms)


def _eta_positive_delta(value: float, data: _EtaData | None) -> float:
    if data is None:
        return np.nan
    return max(_eta_safe_divide(value, data.n_atoms), 0.0)


def _eta_shape_index(data: _EtaData | None, degree: int) -> float:
    if data is None:
        return np.nan
    alpha = _eta_alpha(data)
    return float(
        _eta_safe_divide(
            sum(
                data.alpha[atom.GetIdx()]
                for atom in data.mol.GetAtoms()
                if atom.GetDegree() == degree
            ),
            alpha,
        )
    )


def _eta_beta_sigma(data: _EtaData) -> float:
    return float(data.beta_sigma.sum() / 2.0)


def _eta_beta_non_sigma_delta(data: _EtaData) -> float:
    return float(data.beta_delta.sum())


def _eta_beta_non_sigma(data: _EtaData) -> float:
    return float(data.beta_non_sigma.sum() / 2.0 + _eta_beta_non_sigma_delta(data))


def _eta_value(data: _EtaData, local: bool = False) -> float:
    distances = Chem.GetDistanceMatrix(data.mol, force=True)
    value = 0.0
    for i, row in enumerate(distances):
        for j, distance in enumerate(row):
            if i >= j or (local and distance != 1) or (not local and distance == 0):
                continue
            value += np.sqrt(data.gamma[i] * data.gamma[j] / distance**2)
    return float(value)


def _eta_branching(data: _EtaData, eta_reference_local: float, ring: bool) -> float:
    if data.n_atoms <= 1:
        return np.nan
    eta_normal_local = (
        1.0 if data.n_atoms == 2 else np.sqrt(2.0) + 0.5 * (data.n_atoms - 3)
    )
    ring_count = len(Chem.GetSymmSSSR(data.mol)) if ring else 0
    return float(eta_normal_local - eta_reference_local + 0.086 * ring_count)


def _eta_epsilon(data: _EtaData) -> float:
    if data.n_atoms == 0:
        return np.nan
    return float(data.epsilon.mean())


def _eta_calculate_values(
    data: _EtaData | None,
    reference_data: _EtaData | None,
    saturated_data: _EtaData | None,
) -> list[float]:
    if data is None:
        alpha = beta_sigma = beta_non_sigma = beta_non_sigma_delta = np.nan
        beta = eta = eta_local = eta_branching = eta_branching_ring = np.nan
        epsilon_1 = epsilon_2 = epsilon_5 = delta_beta = psi = np.nan
    else:
        alpha = _eta_alpha(data)
        beta_sigma = _eta_beta_sigma(data)
        beta_non_sigma = _eta_beta_non_sigma(data)
        beta_non_sigma_delta = _eta_beta_non_sigma_delta(data)
        beta = beta_sigma + beta_non_sigma
        eta = _eta_value(data)
        eta_local = _eta_value(data, local=True)
        epsilon_1 = _eta_epsilon(data)
        epsilon_2 = epsilon_1
        epsilon_5 = epsilon_2
        delta_beta = beta_non_sigma - beta_sigma
        psi = _eta_safe_divide(alpha, data.n_atoms * epsilon_2)

    alpha_reference = (
        _eta_alpha(reference_data) if reference_data is not None else np.nan
    )
    eta_reference = _eta_value(reference_data) if reference_data is not None else np.nan
    eta_reference_local = (
        _eta_value(reference_data, local=True) if reference_data is not None else np.nan
    )
    eta_functionality = eta_reference - eta
    eta_functionality_local = eta_reference_local - eta_local
    if data is not None:
        eta_branching = _eta_branching(data, eta_reference_local, ring=False)
        eta_branching_ring = _eta_branching(data, eta_reference_local, ring=True)

    epsilon_3 = _eta_epsilon(reference_data) if reference_data is not None else np.nan
    epsilon_4 = _eta_epsilon(saturated_data) if saturated_data is not None else np.nan

    return [
        alpha,
        _eta_average(alpha, data),
        _eta_shape_index(data, 1),
        _eta_shape_index(data, 3),
        _eta_shape_index(data, 4),
        beta,
        _eta_average(beta, data),
        beta_sigma,
        _eta_average(beta_sigma, data),
        beta_non_sigma,
        _eta_average(beta_non_sigma, data),
        beta_non_sigma_delta,
        _eta_average(beta_non_sigma_delta, data),
        eta,
        _eta_average(eta, data),
        eta_local,
        _eta_average(eta_local, data),
        eta_reference,
        _eta_average(eta_reference, reference_data),
        eta_reference_local,
        _eta_average(eta_reference_local, reference_data),
        eta_functionality,
        _eta_average(eta_functionality, data),
        eta_functionality_local,
        _eta_average(eta_functionality_local, data),
        eta_branching,
        _eta_average(eta_branching, data),
        eta_branching_ring,
        _eta_average(eta_branching_ring, data),
        _eta_positive_delta(alpha - alpha_reference, data),
        _eta_positive_delta(alpha_reference - alpha, data),
        epsilon_1,
        epsilon_2,
        epsilon_3,
        epsilon_4,
        epsilon_5,
        epsilon_1 - epsilon_3,
        epsilon_1 - epsilon_4,
        epsilon_3 - epsilon_4,
        epsilon_2 - epsilon_5,
        delta_beta,
        _eta_average(delta_beta, data),
        psi,
        max(0.714 - psi, 0.0),
        max(psi - 0.714, 0.0),
    ]


def _extended_topochemical_atom_values(mol: Mol, n_frags: int) -> np.ndarray:
    if n_frags != 1:
        return np.full(
            len(EXTENDED_TOPOCHEMICAL_ATOM_FEATURE_NAMES), np.nan, dtype=np.float32
        )
    data = _safe_eta_data(mol)
    reference_data = _safe_altered_data(mol, saturated=False)
    saturated_data = _safe_altered_data(mol, saturated=True)
    values = _eta_calculate_values(data, reference_data, saturated_data)
    return np.asarray(values, dtype=np.float32)


def _fragment_complexity_values(mol: Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    n_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 6)
    value = abs(n_bonds**2 - n_atoms**2 + n_atoms) + n_hetero / 100
    return np.asarray([value], dtype=np.float32)


def _framework_shortest_path(
    graph: dict[_FrameworkNode, list[_FrameworkNode]],
    source: _FrameworkNode,
    target: _FrameworkNode,
) -> list[_FrameworkNode]:
    queue = deque([(source, [source])])
    visited = {source}

    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    return []


def _framework_values(mol: Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return np.asarray([np.nan], dtype=np.float32)

    rings = [frozenset(ring) for ring in Chem.GetSymmSSSR(mol)]
    ring_by_atom = {
        atom_idx: ("R", ring_idx)
        for ring_idx, ring in enumerate(rings)
        for atom_idx in ring
    }
    ring_nodes = list(set(ring_by_atom.values()))

    graph: dict[_FrameworkNode, list[_FrameworkNode]] = {}
    for bond in mol.GetBonds():
        begin = ring_by_atom.get(bond.GetBeginAtomIdx(), ("A", bond.GetBeginAtomIdx()))
        end = ring_by_atom.get(bond.GetEndAtomIdx(), ("A", bond.GetEndAtomIdx()))
        graph.setdefault(begin, []).append(end)
        graph.setdefault(end, []).append(begin)

    linkers: set[int] = set()
    for i, source in enumerate(ring_nodes):
        for target in ring_nodes[i + 1 :]:
            path = _framework_shortest_path(graph, source, target)
            linkers.update(atom_idx for node_type, atom_idx in path if node_type == "A")

    ring_atoms = {atom_idx for ring in rings for atom_idx in ring}
    value = (len(linkers) + len(ring_atoms)) / n_atoms
    return np.asarray([value], dtype=np.float32)


def _geometrical_index_values(mol: Mol | None) -> np.ndarray:
    if mol is None:
        return np.full(len(GEOMETRICAL_INDEX_FEATURE_NAMES), np.nan, dtype=np.float32)

    try:
        conformer = mol.GetConformer()
    except ValueError:
        return np.full(len(GEOMETRICAL_INDEX_FEATURE_NAMES), np.nan, dtype=np.float32)
    if not conformer.Is3D():
        return np.full(len(GEOMETRICAL_INDEX_FEATURE_NAMES), np.nan, dtype=np.float32)

    distances = Get3DDistanceMatrix(mol)
    eccentricities = np.max(distances, axis=0)
    radius = float(np.min(eccentricities))
    diameter = float(np.max(distances))

    shape_index = np.nan if radius == 0.0 else (diameter - radius) / radius
    petitjean_index = np.nan if diameter == 0.0 else (diameter - radius) / diameter
    return np.asarray(
        [diameter, radius, shape_index, petitjean_index], dtype=np.float32
    )


def _has_3d_conformer(mol: Mol | None) -> bool:
    if mol is None:
        return False
    try:
        conformer = mol.GetConformer()
    except ValueError:
        return False
    return conformer.Is3D()


def _gravitational_index_value(
    mass_products: np.ndarray,
    inverse_squared_distances: np.ndarray,
    adjacency: np.ndarray | None = None,
) -> float:
    values = mass_products * inverse_squared_distances
    if adjacency is not None:
        values = values * adjacency
    return float(0.5 * np.sum(values))


def _gravitational_pair_values(mol: Mol) -> list[float]:
    masses = np.asarray([atom.GetMass() for atom in mol.GetAtoms()], dtype=float)
    mass_products = masses[:, np.newaxis] * masses
    np.fill_diagonal(mass_products, 0.0)

    distances = Get3DDistanceMatrix(mol).astype(float)
    if np.any(distances[~np.eye(len(distances), dtype=bool)] == 0.0):
        return [np.nan, np.nan]
    np.fill_diagonal(distances, 1.0)

    inverse_squared_distances = distances**-2
    adjacency = Chem.GetAdjacencyMatrix(mol, useBO=False, force=True)
    return [
        _gravitational_index_value(mass_products, inverse_squared_distances),
        _gravitational_index_value(mass_products, inverse_squared_distances, adjacency),
    ]


def _gravitational_index_values(
    mol_regular: Mol, mol_with_hydrogens: Mol | None
) -> np.ndarray:
    heavy_values = (
        _gravitational_pair_values(mol_regular)
        if _has_3d_conformer(mol_regular)
        else [np.nan, np.nan]
    )
    hydrogen_values = (
        _gravitational_pair_values(mol_with_hydrogens)
        if _has_3d_conformer(mol_with_hydrogens)
        else [np.nan, np.nan]
    )
    return np.asarray(
        [heavy_values[0], hydrogen_values[0], heavy_values[1], hydrogen_values[1]],
        dtype=np.float32,
    )


class _InformationContentBFSTree:
    __slots__ = ("atoms", "bonds", "tree", "visited")

    def __init__(self, mol: Mol):
        self.tree: dict[int, Any] = {}
        self.visited: set[int] = set()

        self.bonds: dict[tuple[int, int], BondType] = {}
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            self.bonds[begin_idx, end_idx] = bond_type
            self.bonds[end_idx, begin_idx] = bond_type

        self.atoms = [
            (
                atom.GetAtomicNum(),
                atom.GetDegree(),
                tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors()),
            )
            for atom in mol.GetAtoms()
        ]

    def reset(self, atom_idx: int) -> None:
        self.tree.clear()
        self.visited.clear()
        self.tree[atom_idx] = {}
        self.visited.add(atom_idx)

    def expand(self) -> None:
        self._expand(self.tree)

    def _expand(self, tree: dict[int, Any]) -> None:
        for src, dst in list(tree.items()):
            self.visited.add(src)
            if not dst:
                tree[src] = {
                    neighbor_idx: {}
                    for neighbor_idx in self.atoms[src][2]
                    if neighbor_idx not in self.visited
                }
            else:
                self._expand(dst)

    def _code(self, tree: dict[int, Any], before: int | None, trail: tuple):
        if not tree:
            yield trail
            return
        for src, dst in tree.items():
            atom_code = self.atoms[src][:2]
            if before is None:
                next_trail = (*trail, atom_code)
            else:
                next_trail = (*trail, self.bonds[before, src], atom_code)
            yield from self._code(dst, src, next_trail)

    def get_code(self, atom_idx: int, order: int) -> tuple:
        self.reset(atom_idx)
        for _ in range(order):
            self.expand()
        return tuple(sorted(self._code(self.tree, None, ())))


def _information_content_values(mol: Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return np.full(len(INFORMATION_CONTENT_FEATURE_NAMES), np.nan, dtype=np.float32)

    ag_values = _information_ag_values_by_order(mol)
    ic_values = [_information_shannon_entropy(counts) for _, counts in ag_values]
    bond_order_sum = sum(bond.GetBondTypeAsDouble() for bond in mol.GetBonds())
    log2_atoms = np.log2(n_atoms)
    log2_bond_order_sum = np.log2(bond_order_sum) if bond_order_sum > 0 else np.nan

    tic_values = [n_atoms * ic_value for ic_value in ic_values]
    sic_values = [_information_safe_divide(v, log2_atoms) for v in ic_values]
    bic_values = [_information_safe_divide(v, log2_bond_order_sum) for v in ic_values]
    cic_values = [log2_atoms - ic_value for ic_value in ic_values]
    mic_values = [
        _information_shannon_entropy(counts, _information_atom_masses(mol, ids))
        for ids, counts in ag_values
    ]
    zmic_values = [
        _information_shannon_entropy(
            counts, counts * _information_atomic_numbers(mol, ids)
        )
        for ids, counts in ag_values
    ]

    return np.asarray(
        [
            *ic_values,
            *tic_values,
            *sic_values,
            *bic_values,
            *cic_values,
            *mic_values,
            *zmic_values,
        ],
        dtype=np.float32,
    )


def _information_ag_values_by_order(mol: Mol) -> list[tuple[np.ndarray, np.ndarray]]:
    tree = _InformationContentBFSTree(mol)
    atom_codes_by_order: list[list[Any]] = [_information_atom_codes_order_0(mol)]
    atom_codes_by_order.extend(
        [tree.get_code(atom_idx, order) for atom_idx in range(mol.GetNumAtoms())]
        for order in range(1, 6)
    )
    return [_information_ag_values(atom_codes) for atom_codes in atom_codes_by_order]


def _information_atom_codes_order_0(mol: Mol) -> list[Any]:
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


def _information_ag_values(atom_codes: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    representative_ids = {code: atom_idx for atom_idx, code in enumerate(atom_codes)}
    grouped_codes = [
        (code, sum(1 for _ in group)) for code, group in groupby(sorted(atom_codes))
    ]
    n_groups = len(grouped_codes)
    ids = np.fromiter(
        (representative_ids[code] for code, _ in grouped_codes),
        dtype=int,
        count=n_groups,
    )
    counts = np.fromiter(
        (count for _, count in grouped_codes),
        dtype=float,
        count=n_groups,
    )
    return ids, counts


def _information_shannon_entropy(counts: np.ndarray, weights=1) -> float:
    total = np.sum(counts)
    if total <= 0:
        return np.nan
    probabilities = counts / total
    entropy_terms = probabilities * np.log2(probabilities)
    return float(-np.sum(weights * entropy_terms))


def _information_safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def _information_atom_masses(mol: Mol, atom_ids: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (mol.GetAtomWithIdx(int(atom_id)).GetMass() for atom_id in atom_ids),
        dtype=float,
        count=len(atom_ids),
    )


def _information_atomic_numbers(mol: Mol, atom_ids: np.ndarray) -> np.ndarray:
    return np.fromiter(
        (mol.GetAtomWithIdx(int(atom_id)).GetAtomicNum() for atom_id in atom_ids),
        dtype=float,
        count=len(atom_ids),
    )


def _kappa_shape_index_values(mol: Mol) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    values = [_kappa_shape_index(mol, n_atoms, order) for order in range(1, 4)]
    return np.asarray(values, dtype=np.float32)


def _kappa_shape_index(mol: Mol, n_atoms: int, order: int) -> float:
    n_paths = len(_chi_subgraphs(mol, order)["path"])
    if n_paths == 0:
        return np.nan

    p_min = n_atoms - order
    if order == 1:
        p_max = 0.5 * n_atoms * (n_atoms - 1)
        return 2 * p_max * p_min / n_paths**2
    if order == 2:
        p_max = 0.5 * (n_atoms - 1) * (n_atoms - 2)
        return 2 * p_max * p_min / n_paths**2
    if n_atoms % 2 == 0:
        p_max = 0.25 * (n_atoms - 2) ** 2
    else:
        p_max = 0.25 * (n_atoms - 1) * (n_atoms - 3)
    return 4 * p_max * p_min / n_paths**2


def _lipinski_values(mol: Mol) -> np.ndarray:
    exact_mol_wt = Descriptors.ExactMolWt(mol)
    log_p = Crippen.MolLogP(mol)
    mol_mr = Crippen.MolMR(mol)

    lipinski = (
        rdMolDescriptors.CalcNumHBD(mol) <= 5
        and rdMolDescriptors.CalcNumHBA(mol) <= 10
        and exact_mol_wt <= 500
        and log_p <= 5
    )
    ghose_filter = (
        160 <= exact_mol_wt <= 480
        and 20 <= mol.GetNumAtoms() <= 70
        and -0.4 <= log_p <= 5.6
        and 40 <= mol_mr <= 130
    )
    return np.asarray([lipinski, ghose_filter], dtype=np.float32)


def _logs_values(mol: Mol) -> np.ndarray:
    value = 0.89823 - 0.10369 * np.sqrt(Descriptors.MolWt(mol))
    for pattern, contribution in _LOGS_PATTERN_CONTRIBUTIONS:
        value += contribution * len(mol.GetSubstructMatches(pattern))
    return np.asarray([value], dtype=np.float32)


def _mcgowan_volume_values(mol: Mol) -> np.ndarray:
    try:
        volume = sum(get_mc_gowan_volume(atom) for atom in mol.GetAtoms())
    except (IndexError, KeyError):
        volume = np.nan
    value = volume - 6.56 * mol.GetNumBonds()
    return np.asarray([value], dtype=np.float32)


def _molecular_distance_edge_value(
    atom_nums: np.ndarray,
    distance_matrix: np.ndarray,
    degrees: np.ndarray,
    atomic_num: int,
    a: int,
    b: int,
) -> np.float32:
    matching_atoms = np.where(atom_nums == atomic_num)[0]
    distances = [
        distance_matrix[i, j]
        for idx, i in enumerate(matching_atoms)
        for j in matching_atoms[idx + 1 :]
        if (degrees[i] == a and degrees[j] == b)
        or (degrees[i] == b and degrees[j] == a)
    ]
    if not distances:
        return np.float32(np.nan)

    distances_arr = np.asarray(distances, dtype=np.float64)
    n = len(distances_arr)
    dx = np.exp(np.sum(np.log(distances_arr)) / (2 * n))
    return np.float32(n / (dx**2))


def _molecular_distance_edge_values(
    mol: Mol, distance_matrix: DistanceMatrix, adjacency_matrix: AdjacencyMatrix
) -> np.ndarray:
    atom_nums = np.fromiter(
        (atom.GetAtomicNum() for atom in mol.GetAtoms()),
        dtype=np.int32,
        count=mol.GetNumAtoms(),
    )
    values = [
        _molecular_distance_edge_value(
            atom_nums,
            distance_matrix.matrix,
            adjacency_matrix.degree,
            atomic_num,
            a,
            b,
        )
        for atomic_num, a, b in _MOLECULAR_DISTANCE_EDGE_FEATURES
    ]
    return np.asarray(values, dtype=np.float32)


def _molecular_id_adjacency_list(mol: Mol) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        begin_idx = begin_atom.GetIdx()
        end_idx = end_atom.GetIdx()
        weight = begin_atom.GetDegree() * end_atom.GetDegree()
        adjacency[begin_idx].append((end_idx, weight))
        adjacency[end_idx].append((begin_idx, weight))
    return adjacency


def _molecular_atomic_id(start: int, adjacency: list[list[tuple[int, int]]]) -> float:
    path_sum = 0.0
    visited = {start}

    def dfs(atom_idx: int, cumulative_weight: int) -> None:
        nonlocal path_sum
        for neighbor_idx, edge_weight in adjacency[atom_idx]:
            if neighbor_idx in visited:
                continue
            visited.add(neighbor_idx)
            next_weight = cumulative_weight * edge_weight
            path_sum += 1.0 / sqrt(next_weight)
            if next_weight < _MOLECULAR_ID_WEIGHT_LIMIT:
                dfs(neighbor_idx, next_weight)
            visited.remove(neighbor_idx)

    dfs(start, 1)
    return 1.0 + path_sum / 2.0


def _molecular_id_values(mol: Mol, n_frags: int) -> np.ndarray:
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0 or n_frags > 1:
        return np.full(len(MOLECULAR_ID_FEATURE_NAMES), np.nan, dtype=np.float32)

    atom_nums = np.fromiter(
        (atom.GetAtomicNum() for atom in mol.GetAtoms()),
        dtype=np.int32,
        count=n_atoms,
    )
    adjacency = _molecular_id_adjacency_list(mol)
    atomic_ids = np.asarray(
        [_molecular_atomic_id(i, adjacency) for i in range(n_atoms)]
    )

    total = np.sum(atomic_ids)
    hetero = np.sum(atomic_ids[(atom_nums != 1) & (atom_nums != 6)])
    carbon = np.sum(atomic_ids[atom_nums == 6])
    nitrogen = np.sum(atomic_ids[atom_nums == 7])
    oxygen = np.sum(atomic_ids[atom_nums == 8])
    halogen = np.sum(atomic_ids[np.isin(atom_nums, list(_MOLECULAR_ID_HALOGENS))])

    return np.asarray(
        [
            total,
            total / n_atoms,
            hetero,
            hetero / n_atoms,
            carbon,
            carbon / n_atoms,
            nitrogen,
            nitrogen / n_atoms,
            oxygen,
            oxygen / n_atoms,
            halogen,
            halogen / n_atoms,
        ],
        dtype=np.float32,
    )


def _path_count_bond_ids_to_atom_ids(
    bond_ids: tuple[int, ...], bond_atoms: list[tuple[int, int]]
) -> list[int] | tuple[int, int]:
    it = iter(bond_ids)

    try:
        a0f, a0t = bond_atoms[next(it)]
    except StopIteration:
        return []

    try:
        a1f, a1t = bond_atoms[next(it)]
    except StopIteration:
        return a0f, a0t

    if a0f in [a1f, a1t]:
        path = [a0t, a0f]
        current = a1f if a0f == a1t else a1t
    else:
        path = [a0f, a0t]
        current = a1f if a0t == a1t else a1t

    for bond_id in it:
        anf, ant = bond_atoms[bond_id]

        path.append(current)

        if anf == current:
            current = ant
        else:
            current = anf

    path.append(current)
    return path


def _path_counts(mol: Mol, order: int) -> tuple[int, float]:
    length = 0
    pi_count = 0.0
    bond_atoms = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    ]

    for path in Chem.FindAllPathsOfLengthN(mol, order):
        atom_ids = set()
        previous = None
        weight = 1.0

        for atom_id in _path_count_bond_ids_to_atom_ids(path, bond_atoms):
            if atom_id in atom_ids:
                break

            atom_ids.add(atom_id)

            if previous is not None:
                bond = mol.GetBondBetweenAtoms(previous, atom_id)
                weight *= bond.GetBondTypeAsDouble()

            previous = atom_id
        else:
            length += 1
            pi_count += weight

    return length, pi_count


def _path_count_values(mol: Mol) -> np.ndarray:
    raw_counts = {order: _path_counts(mol, order) for order in range(1, 11)}
    atom_count = mol.GetNumAtoms()

    values: list[float] = []
    values.extend(raw_counts[order][0] for order in range(2, 11))
    values.append(atom_count + sum(raw_counts[order][0] for order in range(1, 11)))
    values.extend(log(raw_counts[order][1] + 1) for order in range(1, 11))
    values.append(
        log(atom_count + sum(raw_counts[order][1] for order in range(1, 11)) + 1)
    )

    return np.asarray(values, dtype=np.float32)


def _morse_atomic_property_vector(mol: Mol, prop: str) -> np.ndarray:
    prop_func = _AUTOCORRELATION_PROPERTY_FUNCS[prop]
    carbon_value = prop_func(_CARBON)
    values = np.asarray([prop_func(atom) for atom in mol.GetAtoms()], dtype=float)
    return values / carbon_value


def _morse_group_values(
    pair_distances: np.ndarray, weight_products: np.ndarray
) -> list[float]:
    if np.any(~np.isfinite(weight_products)):
        return [np.nan] * MORSE_MAX_DISTANCE

    values = [float(np.sum(weight_products))]
    for distance in range(2, MORSE_MAX_DISTANCE + 1):
        scaled_distances = (distance - 1) * pair_distances
        with np.errstate(divide="ignore", invalid="ignore"):
            kernel = np.sin(scaled_distances) / scaled_distances
        values.append(float(np.sum(weight_products * kernel)))
    return values


def _morse_values(mol: Mol | None) -> np.ndarray:
    if mol is None or mol.GetNumAtoms() <= 1 or not _has_3d_conformer(mol):
        return np.full(len(MORSE_FEATURE_NAMES), np.nan, dtype=np.float32)

    distances = Get3DDistanceMatrix(mol).astype(float)
    triu_idxs = np.triu_indices_from(distances, k=1)
    pair_distances = distances[triu_idxs]

    values: list[float] = []
    values.extend(_morse_group_values(pair_distances, np.ones_like(pair_distances)))
    for prop in MORSE_PROPERTIES[1:]:
        prop_weights = _morse_atomic_property_vector(mol, prop)
        weight_products = prop_weights[triu_idxs[0]] * prop_weights[triu_idxs[1]]
        values.extend(_morse_group_values(pair_distances, weight_products))

    return np.asarray(values, dtype=np.float32)


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


class _SphereMesh:
    def __init__(self, level: int = 5):
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        self.vertices = np.asarray(
            [
                (-1, golden_ratio, 0),
                (1, golden_ratio, 0),
                (-1, -golden_ratio, 0),
                (1, -golden_ratio, 0),
                (0, -1, golden_ratio),
                (0, 1, golden_ratio),
                (0, -1, -golden_ratio),
                (0, 1, -golden_ratio),
                (golden_ratio, 0, -1),
                (golden_ratio, 0, 1),
                (-golden_ratio, 0, -1),
                (-golden_ratio, 0, 1),
            ],
            dtype=float,
        )
        self.faces = np.asarray(
            [
                (0, 11, 5),
                (0, 5, 1),
                (0, 1, 7),
                (0, 7, 10),
                (0, 10, 11),
                (1, 5, 9),
                (5, 11, 4),
                (11, 10, 2),
                (10, 7, 6),
                (7, 1, 8),
                (3, 9, 4),
                (3, 4, 2),
                (3, 2, 6),
                (3, 6, 8),
                (3, 8, 9),
                (4, 9, 5),
                (2, 4, 11),
                (6, 2, 10),
                (8, 6, 7),
                (9, 8, 1),
            ],
            dtype=int,
        )

        self._normalize(0)
        self._level = 1
        self._subdivide_to(level)

    def _normalize(self, begin: int) -> None:
        self.vertices[begin:] /= np.sqrt((self.vertices[begin:] ** 2).sum(axis=1))[
            :, np.newaxis
        ]

    def _subdivide(self) -> None:
        self._level += 1

        n_vertices = len(self.vertices)
        n_faces = len(self.faces)

        a = self.faces[:, 0]
        b = self.faces[:, 1]
        c = self.faces[:, 2]

        av = self.vertices[a]
        bv = self.vertices[b]
        cv = self.vertices[c]

        self.vertices = np.r_[self.vertices, av + bv, bv + cv, av + cv]
        self._normalize(n_vertices)

        ab = np.arange(len(self.faces)) + n_vertices
        bc = ab + n_faces
        ac = ab + 2 * n_faces

        self.faces = (
            np.concatenate([a, b, c, ab, ab, ab, ac, ac, ac, bc, bc, bc])
            .reshape(3, -1)
            .T
        )

    def _subdivide_to(self, level: int) -> None:
        for _ in range(level - self._level):
            self._subdivide()


def _atomic_surface_areas(
    mol: Mol, conformer: int = -1, solvent_radius: float = 1.4, level: int = 5
) -> np.ndarray:
    radii = np.asarray(
        [
            MORDRED_VDW_RADII[atom.GetAtomicNum()] + solvent_radius
            for atom in mol.GetAtoms()
        ],
        dtype=float,
    )
    radii_squared = radii**2

    conf = mol.GetConformer(conformer)
    coords = np.asarray(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=float,
    )

    cutoff_distances = radii[:, np.newaxis] + radii
    distances = cdist(coords, coords)
    if level not in _SPHERE_MESH_CACHE:
        _SPHERE_MESH_CACHE[level] = _SphereMesh(level).vertices.T
    mesh = _SPHERE_MESH_CACHE[level]

    surface_areas: list[float] = []
    for atom_idx, radius in enumerate(radii):
        surface_area = 4.0 * np.pi * radii_squared[atom_idx]
        neighbor_idxs = np.flatnonzero(
            cutoff_distances[atom_idx] >= distances[atom_idx]
        )
        neighbor_idxs = [idx for idx in neighbor_idxs if idx != atom_idx]
        neighbor_idxs.sort(key=lambda idx: distances[atom_idx, idx])

        if not neighbor_idxs:
            surface_areas.append(float(surface_area))
            continue

        sphere = mesh * radius + coords[atom_idx, np.newaxis].T
        n_points = sphere.shape[1]

        for neighbor_idx in neighbor_idxs:
            neighbor_coords = coords[neighbor_idx, np.newaxis].T
            squared_distances = (sphere - neighbor_coords) ** 2
            mask = (
                squared_distances[0] + squared_distances[1] + squared_distances[2]
            ) > radii_squared[neighbor_idx]
            sphere = np.compress(mask, sphere, axis=1)

        surface_areas.append(float(surface_area * sphere.shape[1] / n_points))

    return np.asarray(surface_areas, dtype=float)


def _gasteiger_charges(mol: Mol) -> np.ndarray:
    return np.asarray(
        [
            atom.GetDoubleProp("_GasteigerCharge")
            + (
                atom.GetDoubleProp("_GasteigerHCharge")
                if atom.HasProp("_GasteigerHCharge")
                else 0.0
            )
            for atom in mol.GetAtoms()
        ],
        dtype=float,
    )


def _relative_charge(charges: np.ndarray, mask: np.ndarray) -> float:
    matching_charges = charges[mask]
    if len(matching_charges) == 0:
        return 0.0

    qmax = matching_charges[np.argmax(np.abs(matching_charges))]
    return float(qmax / np.sum(matching_charges))


def _partial_surface_areas(
    surface_areas: np.ndarray, charges: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    charge_sum = np.sum(charges[mask])
    mask_count = np.sum(mask)
    n_atoms = len(charges)

    values = [
        np.sum(surface_areas[mask]),
        np.sum(charge_sum * surface_areas[mask]),
        np.sum(charges[mask] * surface_areas[mask]),
        np.sum((charge_sum / n_atoms) * surface_areas[mask]),
        np.nan
        if mask_count == 0
        else np.sum((charge_sum / mask_count) * surface_areas[mask]),
    ]
    return np.asarray(values, dtype=float)


def _relative_charge_surface_area(
    surface_areas: np.ndarray,
    charges: np.ndarray,
    mask: np.ndarray,
    relative_charge: float,
) -> float:
    matching_charges = charges[mask]
    if len(matching_charges) == 0:
        return 0.0

    surface_area_max = surface_areas[mask][np.argmax(np.abs(matching_charges))]
    return float(surface_area_max / relative_charge)


def _cpsa_2d_values(mol: Mol) -> np.ndarray:
    charges = _gasteiger_charges(mol)
    negative_mask = charges < 0.0
    positive_mask = charges > 0.0

    values = [
        _relative_charge(charges, negative_mask),
        _relative_charge(charges, positive_mask),
    ]
    return np.asarray(values, dtype=np.float32)


def _cpsa_3d_values(mol: Mol | None) -> np.ndarray:
    if mol is None:
        return np.full(len(CPSA_FEATURE_NAMES_3D), np.nan, dtype=np.float32)

    rdPartialCharges.ComputeGasteigerCharges(mol)
    charges = _gasteiger_charges(mol)
    try:
        conformer = mol.GetConformer()
    except ValueError:
        return np.full(len(CPSA_FEATURE_NAMES_3D), np.nan, dtype=np.float32)
    if not conformer.Is3D():
        return np.full(len(CPSA_FEATURE_NAMES_3D), np.nan, dtype=np.float32)

    try:
        surface_areas = _atomic_surface_areas(mol)
    except ValueError:
        return np.full(len(CPSA_FEATURE_NAMES_3D), np.nan, dtype=np.float32)

    total_surface_area = np.sum(surface_areas)
    negative_mask = charges < 0.0
    positive_mask = charges > 0.0

    pnsa = _partial_surface_areas(surface_areas, charges, negative_mask)
    ppsa = _partial_surface_areas(surface_areas, charges, positive_mask)
    dpsa = ppsa - pnsa
    fnsa = pnsa / total_surface_area
    fpsa = ppsa / total_surface_area
    wnsa = pnsa * total_surface_area / 1000.0
    wpsa = ppsa * total_surface_area / 1000.0

    rncg = _relative_charge(charges, negative_mask)
    rpcg = _relative_charge(charges, positive_mask)
    rncs = _relative_charge_surface_area(surface_areas, charges, negative_mask, rncg)
    rpcs = _relative_charge_surface_area(surface_areas, charges, positive_mask, rpcg)

    hydrophobic_mask = np.abs(charges) < 0.2
    polar_mask = np.abs(charges) >= 0.2
    tasa = np.sum(surface_areas[hydrophobic_mask])
    tpsa = np.sum(surface_areas[polar_mask])
    rasa = tasa / total_surface_area
    rpsa = tpsa / total_surface_area

    values = [
        *pnsa,
        *ppsa,
        *dpsa,
        *fnsa,
        *fpsa,
        *wnsa,
        *wpsa,
        rncs,
        rpcs,
        tasa,
        tpsa,
        rasa,
        rpsa,
    ]
    return np.asarray(values, dtype=np.float32)


class _DetourTimeoutError(TimeoutError):
    """Raised when detour matrix enumeration exceeds the configured timeout."""


class _LongestSimplePath:
    """Compute longest simple-path distances in a biconnected component."""

    def __init__(self, graph: nx.Graph, timeout_at: float | None):
        self._graph = graph
        self._timeout_at = timeout_at
        self._neighbors = {node: list(graph[node]) for node in graph.nodes()}
        self._start: int
        self._result: dict[int, float]
        self._visited: set[int]
        self._distance: float

    def _raise_if_timed_out(self) -> None:
        if self._timeout_at is not None and monotonic() > self._timeout_at:
            raise _DetourTimeoutError

    def _search(self, node: int) -> None:
        self._raise_if_timed_out()
        self._visited.add(node)
        for neighbor in self._neighbors[node]:
            if neighbor in self._visited:
                continue

            self._visited.add(neighbor)
            self._distance += 1.0

            distance = self._distance
            self._result[neighbor] = max(self._result[neighbor], distance)

            self._search(neighbor)

            self._visited.remove(neighbor)
            self._distance -= 1.0

    def _from_node(self, node: int) -> dict[int, float]:
        self._start = node
        self._result = dict.fromkeys(self._graph.nodes(), 0.0)
        self._visited = set()
        self._distance = 0.0
        self._search(node)
        return self._result

    def __call__(self) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for source in self._graph.nodes():
            for target, distance in self._from_node(source).items():
                key = (source, target) if source <= target else (target, source)
                result[key] = distance

        return result


class _DetourMatrixBuilder:
    """Merge biconnected-component detours through articulation points."""

    def __init__(self, graph: nx.Graph, timeout: float | None):
        self._graph = graph
        self._timeout_at = None if timeout is None else monotonic() + timeout
        self._n_nodes = graph.number_of_nodes()
        self._queue: list[tuple[set[int], dict[tuple[int, int], float]]] = []
        self._nodes: set[int]
        self._distances: dict[tuple[int, int], float]

    def _merge(self) -> None:
        for queue_idx in range(1, len(self._queue) + 1):
            new_nodes, new_distances = self._queue[-queue_idx]
            common_nodes = new_nodes & self._nodes
            if not common_nodes:
                continue
            if len(common_nodes) > 1:
                raise ValueError("bug: multiple common nodes")

            common_node = common_nodes.pop()
            self._queue.pop(-queue_idx)
            self._nodes.update(new_nodes)
            break
        else:
            raise ValueError("bug: disconnected biconnected components")

        merged: dict[tuple[int, int], float] = {}
        for i in self._nodes:
            for j in self._nodes:
                if i > j:
                    continue
                merged[(i, j)] = self._merged_distance(
                    i,
                    j,
                    common_node,
                    new_distances,
                )

        self._distances = merged

    def _merged_distance(
        self,
        i: int,
        j: int,
        common_node: int,
        new_distances: dict[tuple[int, int], float],
    ) -> float:
        key = (i, j)
        if key in self._distances:
            return self._distances[key]
        if key in new_distances:
            return new_distances[key]
        if i == j == common_node:
            return max(new_distances[key], self._distances[key])

        i_common = (i, common_node) if i <= common_node else (common_node, i)
        j_common = (j, common_node) if j <= common_node else (common_node, j)

        if i_common in self._distances and j_common in new_distances:
            return self._distances[i_common] + new_distances[j_common]
        if j_common in self._distances and i_common in new_distances:
            return self._distances[j_common] + new_distances[i_common]

        raise ValueError("bug: unknown detour distance")

    def _add_biconnected_component(self, component: set[int]) -> None:
        subgraph = self._graph.subgraph(component)
        distances = _LongestSimplePath(subgraph, self._timeout_at)()
        nodes: set[int] = set()
        for i, j in distances:
            nodes.add(i)
            nodes.add(j)

        self._queue.append((nodes, distances))

    def __call__(self) -> np.ndarray:
        if self._n_nodes == 1:
            return np.array([[0.0]], dtype=float)

        for component in nx.biconnected_components(self._graph):
            self._add_biconnected_component(component)

        self._nodes, self._distances = self._queue.pop()
        while self._queue:
            self._merge()

        matrix = np.empty((self._n_nodes, self._n_nodes), dtype=float)
        for i in range(self._n_nodes):
            for j in range(i, self._n_nodes):
                distance = self._distances[(i, j)]
                matrix[i, j] = distance
                matrix[j, i] = distance

        return matrix


def _detour_matrix(mol: Mol, timeout: float | None) -> np.ndarray:
    graph = nx.Graph()
    graph.add_nodes_from(atom.GetIdx() for atom in mol.GetAtoms())
    graph.add_edges_from(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()
    )

    return _DetourMatrixBuilder(graph, timeout)()


def _detour_matrix_values(
    mol: Mol, n_frags: int, timeout: float | None = 60.0
) -> np.ndarray:
    if n_frags != 1:
        return np.full(len(DETOUR_MATRIX_FEATURE_NAMES), np.nan, dtype=np.float32)

    try:
        matrix = _detour_matrix(mol, timeout)
    except _DetourTimeoutError:
        return np.full(len(DETOUR_MATRIX_FEATURE_NAMES), np.nan, dtype=np.float32)

    attrs = MatrixAttributes(matrix, mol, hermitian=True, n_frags=n_frags)
    values = [
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
        int(0.5 * matrix.sum()),
    ]

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
    distance_matrix_values: np.ndarray
    eccentric_connectivity_index_values: np.ndarray
    estate_values: np.ndarray
    extended_topochemical_atom_values: np.ndarray
    fragment_complexity_values: np.ndarray
    framework_values: np.ndarray
    geometrical_index_values: np.ndarray
    gravitational_index_values: np.ndarray
    information_content_values: np.ndarray
    kappa_shape_index_values: np.ndarray
    lipinski_values: np.ndarray
    logs_values: np.ndarray
    mcgowan_volume_values: np.ndarray
    molecular_distance_edge_values: np.ndarray
    molecular_id_values: np.ndarray
    path_count_values: np.ndarray
    morse_values: np.ndarray
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
    cpsa_2d_values: np.ndarray
    cpsa_3d_values: np.ndarray
    detour_matrix_values: np.ndarray
    mol_with_hydrogens: Mol | None

    @classmethod
    def from_mol(
        cls, mol: Mol, use_3D: bool, detour_timeout: float | None = 60.0
    ) -> MordredMolCache:
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
            distance_matrix_values=_distance_matrix_values(
                mol_regular, n_frags, distance_matrix_regular
            ),
            eccentric_connectivity_index_values=_eccentric_connectivity_index_values(
                distance_matrix_regular, adjacency_matrix_regular
            ),
            estate_values=_estate_values(mol_regular),
            extended_topochemical_atom_values=_extended_topochemical_atom_values(
                mol_kekulized, n_frags
            ),
            fragment_complexity_values=_fragment_complexity_values(mol_regular),
            framework_values=_framework_values(mol_regular),
            geometrical_index_values=_geometrical_index_values(mol_with_hydrogens),
            gravitational_index_values=_gravitational_index_values(
                mol_regular, mol_with_hydrogens
            ),
            information_content_values=_information_content_values(mol_kekulized),
            kappa_shape_index_values=_kappa_shape_index_values(mol_regular),
            lipinski_values=_lipinski_values(mol_regular),
            logs_values=_logs_values(mol_regular),
            mcgowan_volume_values=_mcgowan_volume_values(mol_regular),
            molecular_distance_edge_values=_molecular_distance_edge_values(
                mol_regular, distance_matrix_regular, adjacency_matrix_regular
            ),
            molecular_id_values=_molecular_id_values(mol_regular, n_frags),
            path_count_values=_path_count_values(mol_regular),
            morse_values=_morse_values(mol_with_hydrogens),
            aromatic_values=_aromatic_values(mol_regular),
            autocorrelation_gmats=autocorrelation_gmats,
            autocorrelation_gsums=_autocorrelation_gsums(autocorrelation_gmats),
            autocorrelation_weights=autocorrelation_weights,
            autocorrelation_centered_weights=_centered_weights(autocorrelation_weights),
            barysz_values=_barysz_values(mol_regular, n_frags),
            bcut_values=_bcut_values(mol_regular, n_frags),
            bond_count_values=_bond_count_values(mol_regular, mol_kekulized),
            chi_values=_chi_values(mol_regular),
            constitutional_values=_constitutional_values(mol_regular),
            cpsa_2d_values=_cpsa_2d_values(mol_regular),
            cpsa_3d_values=_cpsa_3d_values(mol_with_hydrogens),
            detour_matrix_values=_detour_matrix_values(
                mol_regular, n_frags, timeout=detour_timeout
            ),
            mol_with_hydrogens=mol_with_hydrogens,
        )
