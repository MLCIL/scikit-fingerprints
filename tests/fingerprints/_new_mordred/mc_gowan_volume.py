import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.mc_gowan_volume import calc
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were computed with mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["VMcGowan"]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # hydrogens carry volume and their bonds are subtracted, so these values
        # only come out right on the hydrogen-explicit molecule
        ("CC", 39.04),
        ("CCO", 44.91),
        ("c1ccccc1", 71.64),
        ("C1CCCCC1", 84.54),
        ("CC(=O)OC1=CC=CC=C1C(=O)O", 128.79),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", 136.32),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 177.71),
        ("OCC1OC(O)C(O)C(O)C1O", 119.76),
        ("CS(=O)(=O)N", 63.02),
        ("O=P(O)(O)O", 54.80),
        ("C#N", 26.33),
        # no hydrogens at all
        ("FC(F)(Cl)Br", 58.21),
        ("Clc1ccccc1", 83.88),
        # two ions: no hydrogens and no bonds, so the correction term vanishes
        ("[Na+].[Cl-]", 53.66),
        # a metal, to check the table covers more than the organic elements
        ("[Pt](Cl)(Cl)([NH3])[NH3]", 105.79),
    ],
)
def test_mc_gowan_volume_values(smiles, expected):
    mol_hydrogens = preprocess_mol(MolFromSmiles(smiles), explicit_hydrogens=True)

    values = calc(AtomicProperties.from_mol(mol_hydrogens))
    assert_allclose(values, np.asarray([expected], dtype=np.float32), rtol=1e-5)
