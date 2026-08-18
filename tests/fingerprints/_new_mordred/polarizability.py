import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.polarizability import (
    calc,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Hexane", [19.355102, 14.044898]),
        ("Benzene", [14.020758, 6.019242]),
        ("Caffeine", [26.03193, 18.03807]),
        ("Lycopene", [104.140408, 56.179592]),
        ("Epicatechin", [39.197102, 15.780898]),
        ("Limonene", [27.368688, 16.051312]),
        ("Allicin", [23.28993, 14.59007]),
        ("Glutathione", [39.047481, 23.556519]),
        ("Digoxin", [122.372752, 77.225248]),
        ("Capsaicin", [51.569411, 30.260589]),
        ("EllagicAcid", [33.796758, 11.227242]),
        ("Astaxanthin", [104.681236, 53.902764]),
    ],
)
def test_polarizability_values(name, expected, mordred_test_mols):
    mol_hydrogens = preprocess_mol(mordred_test_mols[name], explicit_hydrogens=True)

    values = calc(mol_hydrogens)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-5)
