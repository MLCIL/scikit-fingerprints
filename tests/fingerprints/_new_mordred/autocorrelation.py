import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import AddHs, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.autocorrelation import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values from mordred-community tests are also used, stored at
./autocorrelation_reference.json as expected[descriptor][property][molecule],
each a list of values ordered by lag.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "autocorrelations.json") as file:
    _REFERENCE = json.load(file)


@pytest.fixture(scope="module")
def computed_values():
    # cache computed values to avoid recomputing for all parametrizations
    computed = {}
    for name, smiles in _REFERENCE["SMILES"].items():
        # Mordred uses explicit-hydrogen molecules
        mol = AddHs(MolFromSmiles(smiles))
        values = calc(AtomicProperties.from_mol(mol), DistanceMatrix.from_mol(mol))
        computed[name] = dict(zip(FEATURE_NAMES, values, strict=False))
    return computed


@pytest.mark.parametrize("molecule", list(_REFERENCE["SMILES"]))
@pytest.mark.parametrize(
    "prop",
    [
        "mass",
        "van_der_Waals_volume",
        "Sanderson_electronegativity",
        "polarizability",
        "ionization_potential",
        "intrinsic_state",
    ],
)
@pytest.mark.parametrize("descriptor", list(_REFERENCE["expected"]))
def test_autocorrelation_reference_values(descriptor, prop, molecule, computed_values):
    start_lag = 1 if descriptor in {"Moreau_autocorr", "Geary_autocorr"} else 0
    vals_mordred = np.array(
        _REFERENCE["expected"][descriptor][prop][molecule], dtype=float
    )
    vals_skfp = np.array(
        [
            computed_values[molecule][f"{descriptor}_{prop}_lag_{start_lag + i}"]
            for i in range(len(vals_mordred))
        ]
    )
    assert_allclose(vals_skfp, vals_mordred, atol=1e-3, equal_nan=True)
