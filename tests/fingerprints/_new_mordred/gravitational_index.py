import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.gravitational_index import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix3D,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values come from mordred-community's GravitationalIndex.yaml and are
stored at ./references/gravitational_index.json as expected[molecule][descriptor].
They are computed on the fixed 3D conformers in ./references/structures.sdf (the
same structures mordred uses), so the gravitational index (a 3D descriptor) is
reproduced exactly rather than depending on a freshly embedded conformer.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "gravitational_index.json") as file:
    _REFERENCE = json.load(file)


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols_hydrogens_3d):
    # cache computed values to avoid recomputing for all parametrizations
    computed = {}
    for name in _REFERENCE:
        # hydrogen-explicit molecule with the reference 3D conformer
        mol_hydrogens = mordred_test_mols_hydrogens_3d[name]
        conf_id = mol_hydrogens.GetIntProp("conf_id")

        # heavy-atom molecule; RemoveHs preserves the heavy-atom coordinates
        mol_regular = preprocess_mol(mol_hydrogens)

        values, feature_names = calc(
            AtomicProperties(mol_regular),
            AtomicProperties(mol_hydrogens),
            DistanceMatrix3D(mol_regular, conf_id),
            DistanceMatrix3D(mol_hydrogens, conf_id),
            AdjacencyMatrix(mol_regular),
            AdjacencyMatrix(mol_hydrogens),
        )
        computed[name] = dict(zip(feature_names, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", list(_REFERENCE))
@pytest.mark.parametrize("descriptor", FEATURE_NAMES)
def test_gravitational_index_reference_values(descriptor, molecule, computed_values):
    expected = _REFERENCE[molecule][descriptor]
    actual = computed_values[molecule][descriptor]
    assert_allclose(actual, expected, atol=1e-3)
