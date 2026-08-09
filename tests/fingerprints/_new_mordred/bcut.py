import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.bcut import FEATURE_NAMES, calc

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected_smallest, expected_largest",
    [
        ("Hexane", 11.89, 12.11007471),
        ("Benzene", 11.85, 12.1500544),
        ("Caffeine", 11.99347887, 15.99592273),
        ("Cyanidin", 11.85, 15.99993732),
        ("Lycopene", 11.89, 12.11102666),
        ("Epicatechin", 11.85, 15.99993732),
        ("Limonene", 11.9, 12.10017173),
        ("Allicin", 11.79, 31.97307173),
        ("Glutathione", 11.99846101, 31.97267783),
        ("Digoxin", 11.89, 16.00805947),
        ("Capsaicin", 11.89, 15.99692975),
        ("EllagicAcid", 11.85, 16.00194285),
        ("Astaxanthin", 11.89, 15.99795509),
    ],
)
def test_bcut_mass_values(name, expected_smallest, expected_largest, mordred_test_mols):
    mol = mordred_test_mols[name]

    values = calc(mol)
    values = dict(zip(FEATURE_NAMES, values, strict=True))
    actual_smallest = values["BCUT_mass_smallest_eigval"]
    actual_largest = values["BCUT_mass_largest_eigval"]

    # Mordred tests also use such a large tolerance
    assert_allclose(actual_smallest, expected_smallest, atol=1)
    assert_allclose(actual_largest, expected_largest, atol=1)


def test_disconnected_mol_all_nan():
    mol = MolFromSmiles("[Na].[Cl]")
    values = calc(mol)
    assert_allclose(values, np.nan)
