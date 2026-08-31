import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors import mol_filters
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.molecular_properties import (
    MolecularProperties,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were computed with mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Lipinski", "GhoseFilter"]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # too light and too few atoms for the Ghose filter
        ("CC", [1, 0]),
        ("CCO", [1, 0]),
        ("c1ccccc1", [1, 0]),
        # drug-like, both filters pass
        ("CC(=O)OC1=CC=CC=C1C(=O)O", [1, 1]),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", [1, 1]),
        # LogP below the Ghose lower bound of -0.4
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", [1, 0]),
        # LogP below the Ghose bound and molar refractivity under 40
        ("OCC1OC(O)C(O)C(O)C1O", [1, 0]),
        # LogP above 5, failing both filters
        ("CCCCCCCCCCCCCCCCCC(=O)O", [0, 0]),
        ("c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", [0, 0]),
        # 6 hydrogen bond donors, one over the Lipinski limit
        ("OC1C(O)C(O)C(O)C(O)C1O", [0, 0]),
        # over 500 Da
        ("CC1C(C(C(C(O1)OC2C(CC(C(C2O)OC3C(C(C(CO3)(C)O)NC)O)N)N)O)N)O", [0, 0]),
    ],
)
def test_mol_filters_values(smiles, expected):
    mol_regular = preprocess_mol(MolFromSmiles(smiles))

    values = mol_filters.calc(MolecularProperties.from_mol(mol_regular))
    assert_allclose(values, np.asarray(expected, dtype=np.float32))
