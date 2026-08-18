import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.aromatic import FEATURE_NAMES, calc

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values come from ChemAxon cxcalc (as bundled with mordred-community)
and are stored at ./references/aromatic.json as expected[molecule][descriptor].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "aromatic.json") as f:
    _REFERENCE = json.load(f)


@pytest.mark.parametrize("molecule", list(_REFERENCE))
def test_aromatic_reference_values(molecule, mordred_test_mols):
    mol = mordred_test_mols[molecule]

    values = calc(mol)
    expected = [_REFERENCE[molecule][name] for name in FEATURE_NAMES]
    assert_allclose(values, expected)
