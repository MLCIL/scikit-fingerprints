import json
import math
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.chi import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values come from PaDEL-descriptor (as bundled with mordred-community)
and are stored at ./references/chi.json as expected[molecule][descriptor].

Null entries mean the expected value is NaN (averaged chi with no subgraphs of
that order). Missing entries were marked 'skip' in the original YAML and are
not tested.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_SMILES = {
    "Hexane": "CCCCCC",
    "Benzene": "c1ccccc1",
    "Caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Lycopene": "CC(C)=CCC/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(\\C)CCC=C(C)C",
    "Epicatechin": "Oc1cc(O)c2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@H](O)C2",
    "Limonene": "C=C(C)[C@@H]1CC=C(C)CC1",
    "Allicin": "C=CCS[S@](=O)CC=C",
    "Glutathione": "N[C@@H](CCC(=O)N[C@@H](CS)C(=O)NCC(=O)O)C(=O)O",
    "Digoxin": (
        "C[C@H]1O[C@@H](O[C@H]2[C@@H](O)C[C@H](O[C@H]3[C@@H](O)C[C@H](O[C@H]4CC"
        "[C@@]5(C)[C@H](CC[C@@H]6[C@@H]5C[C@@H](O)[C@]5(C)[C@@H](C7=CC(=O)OC7)CC"
        "[C@]65O)C4)O[C@@H]3C)O[C@@H]2C)C[C@H](O)[C@@H]1O"
    ),
    "Capsaicin": "COc1cc(CNC(=O)CCCC/C=C/C(C)C)ccc1O",
    "EllagicAcid": "O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23",
    "Astaxanthin": "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C(=O)[C@@H](O)CC2(C)C)C(C)(C)C[C@H](O)C1=O",
}

with open(Path(__file__).parent / "references" / "chi.json") as f:
    _REFERENCE = json.load(f)


@pytest.fixture(scope="module")
def computed_values():
    computed = {}
    for name, smiles in _SMILES.items():
        mol = preprocess_mol(MolFromSmiles(smiles))
        props = AtomicProperties(mol)
        values, feature_names = calc(props, Subgraphs(mol, props))
        computed[name] = dict(zip(feature_names, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", list(_SMILES))
@pytest.mark.parametrize("feature_name", FEATURE_NAMES)
def test_chi_reference_values(feature_name, molecule, computed_values):
    if feature_name not in _REFERENCE[molecule]:
        pytest.skip("no reference value (mordred skip)")
    expected = _REFERENCE[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    if expected is None:
        assert math.isnan(actual)
    else:
        assert_allclose(actual, expected, atol=1e-4)
