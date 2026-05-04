"""
Per-molecule dependency cache for New Mordred descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit.Chem import GetMolFrags, Mol

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
    mol_with_hydrogens: Mol | None

    @classmethod
    def from_mol(cls, mol: Mol, use_3D: bool) -> MordredMolCache:
        mol_regular = preprocess_mol(mol)
        n_frags = len(GetMolFrags(mol))
        adjacency_matrix_regular = AdjacencyMatrix(mol_regular)
        return cls(
            original_mol=mol,
            use_3d=use_3D,
            n_frags=n_frags,
            mol_regular=mol_regular,
            mol_kekulized=preprocess_mol(mol, kekulize=True),
            distance_matrix_regular=DistanceMatrix(mol_regular),
            adjacency_matrix_regular=adjacency_matrix_regular,
            adjacency_matrix_values=_adjacency_matrix_values(
                mol_regular, n_frags, adjacency_matrix_regular
            ),
            aromatic_values=_aromatic_values(mol_regular),
            mol_with_hydrogens=(
                preprocess_mol(mol, explicit_hydrogens=True) if use_3D else None
            ),
        )
