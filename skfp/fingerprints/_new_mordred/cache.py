"""
Per-molecule dependency cache for New Mordred descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit.Chem import GetMolFrags, Mol

from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol


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
    mol_with_hydrogens: Mol | None

    @classmethod
    def from_mol(cls, mol: Mol, use_3D: bool) -> MordredMolCache:
        mol_regular = preprocess_mol(mol)
        return cls(
            original_mol=mol,
            use_3d=use_3D,
            n_frags=len(GetMolFrags(mol)),
            mol_regular=mol_regular,
            mol_kekulized=preprocess_mol(mol, kekulize=True),
            distance_matrix_regular=DistanceMatrix(mol_regular),
            adjacency_matrix_regular=AdjacencyMatrix(mol_regular),
            mol_with_hydrogens=(
                preprocess_mol(mol, explicit_hydrogens=True) if use_3D else None
            ),
        )
