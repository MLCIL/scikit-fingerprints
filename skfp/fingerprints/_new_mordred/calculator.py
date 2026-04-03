# This code has been adapted from mordred-community:
# https://github.com/JacksonBurns/mordred-community
#
# Copyright (c) 2023, Jackson Burns and the mordredcommunity Team
# All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.

import numpy as np
from rdkit.Chem import GetMolFrags, Mol

from skfp.fingerprints._new_mordred.descriptors import abc_index
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol


def get_feature_names(use_3d: bool) -> np.ndarray:
    return (
        np.asarray(ALL_FEATURE_NAMES, dtype=object)
        if use_3d
        else np.asarray(FEATURE_NAMES_2D, dtype=object)
    )


def compute(mol: Mol, use_3D: bool) -> np.ndarray:  # noqa: ARG001
    # dependencies
    n_frags = len(GetMolFrags(mol))  # noqa: F841

    mol_regular, _ = preprocess_mol(mol)
    distance_matrix_regular = DistanceMatrix(mol_regular)

    # descriptors
    descriptors = [abc_index.calc(mol_regular, distance_matrix_regular)]

    return np.concatenate(descriptors, axis=1).astype(np.float32, copy=False)
