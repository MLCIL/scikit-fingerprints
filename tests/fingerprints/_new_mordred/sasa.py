import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.utils.sasa import solvent_accessible_surface_area

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


@pytest.mark.parametrize(
    # total surface areas as mordred-community itself computes them, summing its
    # SurfaceArea over the atoms at mesh level 5, the level its CPSA descriptors
    # use; these molecules are the ones its own SASA test covers
    "name, expected_value",
    [
        ("Hexane", 285.439),
        ("Benzene", 235.120),
        ("Caffeine", 359.799),
        ("Cyanidin", 475.270),
        ("Lycopene", 1138.664),
        ("Epicatechin", 476.793),
        ("Limonene", 347.205),
        ("Allicin", 347.681),
        ("Glutathione", 520.563),
        ("Digoxin", 1054.415),
        ("Capsaicin", 622.420),
        ("EllagicAcid", 435.430),
        ("Astaxanthin", 1055.327),
        ("DMSO", 221.022),
        ("DiethylThioketone", 280.823),
        ("VinylsulfonicAcid", 240.107),
        ("Thiophene", 220.846),
        ("Triethoxyphosphine", 382.125),
        ("MethylphosphonicAcid", 230.275),
        ("MethylCyclopropane", 218.009),
        ("Acetonitrile", 177.003),
        ("Histidine", 327.239),
    ],
)
def test_sasa_reference_values(name, expected_value, mordred_test_mols_hydrogens_3d):
    mol = mordred_test_mols_hydrogens_3d[name]
    actual_value = solvent_accessible_surface_area(mol).sum()
    # 3% relative tolerance: we share mordred's radii, but integrate the exposed
    # fraction of each atom on FreeSASA's 100 test points against its 5112-point
    # icosphere, and RDKit exposes no way to raise that count
    assert_allclose(actual_value, expected_value, rtol=0.03)
