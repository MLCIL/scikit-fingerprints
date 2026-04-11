import os

import pytest
from rdkit import Chem

from skfp.fingerprints._new_mordred.utils.sasa import SurfaceArea

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# Reference values calculated by PyMOL
REFERENCE_DATA = [
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
]

SDF_FILE = os.path.join(os.path.dirname(__file__), "references", "structures.sdf")


@pytest.fixture(scope="module")
def sdf_mols() -> dict[str, Chem.Mol]:
    mols = {}
    for mol in Chem.SDMolSupplier(SDF_FILE, removeHs=False):
        if mol is not None:
            mols[mol.GetProp("_Name")] = mol
    return mols


@pytest.mark.parametrize(
    "name,expected_sasa",
    REFERENCE_DATA,
    ids=[row[0] for row in REFERENCE_DATA],
)
def test_sasa_reference_values(name, expected_sasa, sdf_mols):
    """
    Check per-molecule SASA against reference values calculated with PyMOL.
    """
    mol = sdf_mols[name]
    actual = sum(SurfaceArea.from_mol(mol).surface_area())

    rel_error = abs((actual - expected_sasa) / expected_sasa)
    assert rel_error < 0.05, f"large SASA error in {name}: {rel_error:.4f}"
