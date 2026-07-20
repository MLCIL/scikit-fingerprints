import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.utils.sasa import solvent_accessible_surface_area

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    "name, expected_value",
    [
        ("Hexane", 296.910),
        ("Benzene", 243.552),
        ("Caffeine", 369.973),
        ("Cyanidin", 483.873),
        ("Lycopene", 1172.253),
        ("Epicatechin", 489.498),
        ("Limonene", 361.278),
        ("Allicin", 356.872),
        ("Glutathione", 530.679),
        ("Digoxin", 1074.428),
        ("Capsaicin", 641.527),
        ("EllagicAcid", 440.267),
        ("Astaxanthin", 1080.941),
        ("DMSO", 227.926),
        ("DiethylThioketone", 290.503),
        ("VinylsulfonicAcid", 246.033),
        ("Thiophene", 227.046),
        ("Triethoxyphosphine", 396.482),
        ("MethylphosphonicAcid", 235.685),
        ("MethylCyclopropane", 229.071),
        ("Acetonitrile", 182.197),
        ("Histidine", 335.672),
    ],
)
def test_sasa_reference_values(name, expected_value, mordred_test_mols_hydrogens_3d):
    mol = mordred_test_mols_hydrogens_3d[name]
    actual_value = solvent_accessible_surface_area(mol).sum()
    # 5% relative tolerance, matching mordred's own SASA test
    assert_allclose(actual_value, expected_value, rtol=0.05)
