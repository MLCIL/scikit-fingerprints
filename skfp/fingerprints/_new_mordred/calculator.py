# This code has been adapted from mordred-community:
# https://github.com/JacksonBurns/mordred-community
#
# Copyright (c) 2023, Jackson Burns and the mordredcommunity Team
# All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.

import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.descriptors import (
    abc_index,
    acid_base,
    atom_count,
    carbon_types,
    rdkit_descriptors,
    ring_count,
    rotatable_bond,
    wiener_index,
    zagreb_index,
)
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)

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

    # cache
    cache = MordredMolCache.from_mol(mol, use_3D=use_3D)

    # dependencies
    n_frags = cache.n_frags  # noqa: F841
    mol_regular = cache.mol_regular
    mol_kekulized = cache.mol_kekulized
    distance_matrix_regular = cache.distance_matrix_regular
    adjacency_matrix_regular = cache.adjacency_matrix_regular

    # 2D descriptors
    descriptors_2d = [
        abc_index.calc(mol_regular, distance_matrix_regular),
        acid_base.calc(mol_regular),
        wiener_index.calc(mol_regular, distance_matrix_regular),
        zagreb_index.calc(mol_regular, adjacency_matrix_regular),
        rdkit_descriptors.calc_2d(
            mol_regular,
            distance_matrix_regular,
        ),
        atom_count.calc(mol_regular),
        carbon_types.calc(mol_kekulized),
        rotatable_bond.calc(mol_regular),
        ring_count.calc(mol_regular),
    ]

    for values, feature_names in descriptors_2d:
        result[[idx_map[n] for n in feature_names]] = values

    # 3D descriptors
    if use_3D:
        mol_with_hydrogens = cache.mol_with_hydrogens
        if mol_with_hydrogens is None:
            raise RuntimeError("3D Mordred cache was not initialized.")

        descriptors_3d: list = [
            rdkit_descriptors.calc_3d(mol_with_hydrogens),
        ]

        for values, feature_names in descriptors_3d:
            result[[idx_map[n] for n in feature_names]] = values

    return result
