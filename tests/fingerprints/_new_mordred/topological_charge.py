import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.topological_charge import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were generated with mordred-community and are stored at
./references/topological_charge.json as expected[molecule][feature_name].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "topological_charge.json") as file:
    _REFERENCE = json.load(file)


@pytest.mark.parametrize("molecule", list(_REFERENCE["expected"]))
def test_topological_charge_reference_values(molecule, mordred_test_mols):
    mol = mordred_test_mols[molecule]

    values = calc(AdjacencyMatrix(mol), DistanceMatrix(mol))

    expected = _REFERENCE["expected"][molecule]
    vals_mordred = np.array([expected[f] for f in FEATURE_NAMES], dtype=float)
    assert_allclose(values, vals_mordred, atol=1e-4)


def test_feature_names():
    assert _REFERENCE["feature_names"] == FEATURE_NAMES
    assert (
        [f"GGI{order}" for order in range(1, 11)]
        + [f"JGI{order}" for order in range(1, 11)]
        + ["JGT10"]
    ) == FEATURE_NAMES


def test_global_is_sum_of_mean():
    # JGT10 is defined as the sum of the mean charges (JGI1..JGI10)
    mol = MolFromSmiles("CCN(CC)CCOC(=O)c1ccc(N)cc1")
    values = calc(AdjacencyMatrix(mol), DistanceMatrix(mol))

    jgi = values[10:20]
    jgt10 = values[20]
    assert_allclose(jgt10, jgi.sum(), atol=1e-5)


def test_symmetric_molecule_all_zero():
    # a molecule with symmetric charge distribution has vanishing charge terms
    mol = MolFromSmiles("c1ccccc1")
    values = calc(AdjacencyMatrix(mol), DistanceMatrix(mol))
    assert_allclose(values, 0.0, atol=1e-7)
