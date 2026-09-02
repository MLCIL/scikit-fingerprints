import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.molecular_distance_edge import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values generated from mordred-community and stored at
./references/molecular_distance_edge.json as expected[molecule][descriptor].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "molecular_distance_edge.json") as f:
    _REFERENCE = json.load(f)

_PARAMS = [
    (molecule, feature_name)
    for molecule, features in _REFERENCE.items()
    for feature_name in features
]


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols):
    computed = {}
    for name in _REFERENCE:
        mol_regular = preprocess_mol(mordred_test_mols[name])
        values = calc(
            AtomicProperties.from_mol(mol_regular),
            AdjacencyMatrix(mol_regular),
            DistanceMatrix.from_mol(mol_regular),
        )
        computed[name] = dict(zip(FEATURE_NAMES, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule, feature_name", _PARAMS)
def test_molecular_distance_edge_reference_values(
    molecule, feature_name, computed_values
):
    # JSON has no NaN literal, so empty buckets are stored as null (None)
    expected = _REFERENCE[molecule][feature_name]
    expected = np.nan if expected is None else np.float32(expected)
    actual = computed_values[molecule][feature_name]
    assert_allclose(actual, expected, rtol=1e-5, equal_nan=True)


def test_molecular_distance_edge_output_shape(mordred_test_mols):
    mol_regular = preprocess_mol(mordred_test_mols["Caffeine"])
    values = calc(
        AtomicProperties.from_mol(mol_regular),
        AdjacencyMatrix(mol_regular),
        DistanceMatrix.from_mol(mol_regular),
    )

    assert isinstance(values, np.ndarray)
    assert values.dtype == np.float32
    assert values.shape == (len(FEATURE_NAMES),)


def test_molecular_distance_edge_no_matching_atoms(mordred_test_mols):
    # Hexane has only carbons, so every nitrogen/oxygen bucket must be empty (NaN)
    # while carbon buckets that occur are finite
    mol_regular = preprocess_mol(mordred_test_mols["Hexane"])
    values = calc(
        AtomicProperties.from_mol(mol_regular),
        AdjacencyMatrix(mol_regular),
        DistanceMatrix.from_mol(mol_regular),
    )
    result = dict(zip(FEATURE_NAMES, values, strict=True))

    assert all(
        np.isnan(result[name])
        for name in FEATURE_NAMES
        if name.startswith(("MDEN", "MDEO"))
    )
    assert np.isfinite(result["MDEC-11"])
