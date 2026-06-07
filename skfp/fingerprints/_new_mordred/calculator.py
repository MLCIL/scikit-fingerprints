import numpy as np
from rdkit.Chem import GetMolFrags, Mol

from skfp.fingerprints._new_mordred.descriptors import (
    abc_index,
    acid_base,
    adjacency_matrix,
    autocorrelation,
    barysz_matrix,
    walk_count,
    wiener_index,
    zagreb_index,
)
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from tests.fingerprints._new_mordred import estate

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_FEATURE_NAME_TO_IDX_2D = {name: i for i, name in enumerate(FEATURE_NAMES_2D)}
_FEATURE_NAME_TO_IDX_ALL = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}


def get_feature_names(use_3d: bool) -> np.ndarray:
    return (
        np.asarray(ALL_FEATURE_NAMES, dtype=object)
        if use_3d
        else np.asarray(FEATURE_NAMES_2D, dtype=object)
    )


def compute(mol: Mol, use_3D: bool) -> np.ndarray:
    n_features = len(ALL_FEATURE_NAMES) if use_3D else len(FEATURE_NAMES_2D)
    idx_map = _FEATURE_NAME_TO_IDX_ALL if use_3D else _FEATURE_NAME_TO_IDX_2D
    result = np.full(n_features, np.nan, dtype=np.float32)

    # dependencies
    n_frags = len(GetMolFrags(mol))

    # classic, RDKit-standardized molecule
    mol_regular = preprocess_mol(mol)
    distance_matrix_regular = DistanceMatrix(mol_regular)
    adjacency_matrix_regular = AdjacencyMatrix(mol_regular)

    # hydrogen-explicit molecule
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    distance_matrix_hydrogens = DistanceMatrix(mol_hydrogens)

    # 2D descriptors
    descriptors_2d = [
        abc_index.calc(mol_regular, distance_matrix_regular),
        walk_count.calc(mol_regular, adjacency_matrix_regular),
        adjacency_matrix.calc(mol_regular, n_frags, adjacency_matrix_regular),
        wiener_index.calc(mol_regular, distance_matrix_regular),
        zagreb_index.calc(mol_regular, adjacency_matrix_regular),
        acid_base.calc(mol_regular),
        autocorrelation.calc(mol_hydrogens, distance_matrix_hydrogens),
        estate.calc(mol_regular),
        barysz_matrix.calc(mol_regular, n_frags),
    ]

    for values, feature_names in descriptors_2d:
        result[[idx_map[n] for n in feature_names]] = values

    # 3D descriptors
    if use_3D:
        descriptors_3d: list = []

        for values, feature_names in descriptors_3d:
            result[[idx_map[n] for n in feature_names]] = values

    return result
