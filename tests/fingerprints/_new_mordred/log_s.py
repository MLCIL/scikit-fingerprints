import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors import log_s
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were computed with mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["FilterItLogS"]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # hydroxyl, methyl and chain methylene
        ("CCO", 0.129203),
        # aromatic CH only
        ("c1ccccc1", -1.364304),
        # ether oxygen
        ("CCOCC", -1.070286),
        # tertiary and primary amine nitrogen
        ("CN(C)C", -0.200274),
        ("CN", 0.392049),
        # three-connected aromatic nitrogen
        ("Cn1cccc1", -0.446589),
        # ring methylene, and a ring carbon without hydrogens
        ("C1CCCCC1", -1.436441),
        ("CC1(C)CCCCC1", -2.249909),
        # drug-like molecules, mixing substituted aromatic carbons with the rest
        ("CC(=O)OC1=CC=CC=C1C(=O)O", -1.240852),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", -0.518727),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", -2.834961),
        ("CC(=O)Nc1ccc(O)cc1", -1.586460),
        # one halogen each, then several at once
        ("Fc1ccccc1", -1.833001),
        ("Clc1ccccc1", -2.196517),
        ("Brc1ccccc1", -2.478313),
        ("Ic1ccccc1", -2.595716),
        ("FC(F)(Cl)C(F)Br", -2.287405),
    ],
)
def test_log_s_values(smiles, expected):
    mol_regular = preprocess_mol(MolFromSmiles(smiles))

    values = log_s.calc(mol_regular)
    assert_allclose(values, np.asarray([expected], dtype=np.float32), rtol=1e-5)
