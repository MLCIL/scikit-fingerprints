import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.fragment_complexity import (
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

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

_REFERENCE = {
    "Hexane": 5,
    "Benzene": 6,
    "Caffeine": 43.06,
    "Lycopene": 39,
    "Epicatechin": 109.06,
    "Limonene": 10,
    "Allicin": 8.03,
    "Glutathione": 19.1,
    "Digoxin": 874.14,
    "Capsaicin": 22.04,
    "EllagicAcid": 163.08,
    "Astaxanthin": 133.04,
}


@pytest.fixture(scope="module")
def computed_values():
    computed = {}
    for name, smiles in _SMILES.items():
        mol = preprocess_mol(MolFromSmiles(smiles))
        values, feature_names = calc(AtomicProperties(mol))
        computed[name] = dict(zip(feature_names, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", list(_SMILES))
def test_fragment_complexity_reference_values(molecule, computed_values):
    expected = _REFERENCE[molecule]
    actual = computed_values[molecule]["fragCpx"]
    assert_allclose(actual, expected, atol=1e-3)
