import importlib

import numpy as np
import pytest
from mordred import BondCount, Calculator
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = [
    "nBonds",
    "nBondsO",
    "nBondsS",
    "nBondsD",
    "nBondsT",
    "nBondsA",
    "nBondsM",
    "nBondsKS",
    "nBondsKD",
]

SMILES = [
    "C",
    "CC",
    "C=C",
    "C#N",
    "c1ccccc1",
    "C1CCCCC1",
    "c1ccccc1.C=C",
    "[Na+].[Cl-]",
    "O=S(=O)(O)O",
]

MORDRED_MATCHING_FEATURE_NAMES = [
    "nBondsO",
    "nBondsD",
    "nBondsT",
    "nBondsA",
    "nBondsM",
    "nBondsKD",
]


@pytest.fixture(scope="module")
def mordred_bond_count_calc():
    return Calculator(BondCount, ignore_3D=True)


def _expected_no_explicit_h(mol):
    mol_regular = preprocess_mol(mol)
    mol_kekulized = preprocess_mol(mol, kekulize=True)
    bonds_regular = list(mol_regular.GetBonds())
    bonds_kekulized = list(mol_kekulized.GetBonds())

    def is_aromatic(bond):
        return bond.GetIsAromatic() or bond.GetBondType() == BondType.AROMATIC

    values = [
        len(bonds_regular),
        len(bonds_regular),
        sum(bond.GetBondType() == BondType.SINGLE for bond in bonds_regular),
        sum(bond.GetBondType() == BondType.DOUBLE for bond in bonds_regular),
        sum(bond.GetBondType() == BondType.TRIPLE for bond in bonds_regular),
        sum(is_aromatic(bond) for bond in bonds_regular),
        sum(
            is_aromatic(bond) or bond.GetBondType() != BondType.SINGLE
            for bond in bonds_regular
        ),
        sum(bond.GetBondType() == BondType.SINGLE for bond in bonds_kekulized),
        sum(bond.GetBondType() == BondType.DOUBLE for bond in bonds_kekulized),
    ]
    return np.asarray(values, dtype=np.float32)


def _mordred_values(mol, mordred_bond_count_calc):
    return dict(
        zip(
            (str(desc) for desc in mordred_bond_count_calc.descriptors),
            mordred_bond_count_calc(mol),
            strict=False,
        )
    )


def test_bond_count_feature_names_are_in_mordred_order():
    bond_count = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.bond_count"
    )

    assert bond_count.FEATURE_NAMES == FEATURE_NAMES


@pytest.mark.parametrize("smiles", SMILES)
def test_bond_count_uses_no_explicit_h_expected_values(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    bond_count = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.bond_count"
    )

    values, feature_names = bond_count.calc(cache)
    expected = _expected_no_explicit_h(mol)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=0, atol=0)


@pytest.mark.parametrize("smiles", SMILES)
def test_bond_count_matches_mordred_where_no_explicit_h_policy_does_not_change_values(
    smiles, mordred_bond_count_calc
):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    bond_count = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.bond_count"
    )

    values, feature_names = bond_count.calc(cache)
    observed = dict(zip(feature_names, values, strict=True))
    mordred_values = _mordred_values(mol, mordred_bond_count_calc)

    expected = np.asarray(
        [mordred_values[name] for name in MORDRED_MATCHING_FEATURE_NAMES],
        dtype=np.float32,
    )
    actual = np.asarray(
        [observed[name] for name in MORDRED_MATCHING_FEATURE_NAMES],
        dtype=np.float32,
    )
    assert_allclose(actual, expected, rtol=0, atol=0)


def test_bond_count_documents_no_explicit_h_difference_from_default_mordred(
    mordred_bond_count_calc,
):
    mol = Chem.MolFromSmiles("CC")
    cache = MordredMolCache.from_mol(mol, use_3D=False)
    bond_count = importlib.import_module(
        "skfp.fingerprints._new_mordred.descriptors.bond_count"
    )

    values, feature_names = bond_count.calc(cache)
    observed = dict(zip(feature_names, values, strict=True))
    mordred_values = _mordred_values(mol, mordred_bond_count_calc)

    # Mordred's default nBonds, nBondsS, and nBondsKS count explicit C-H bonds.
    # New Mordred descriptors intentionally keep 2D molecules hydrogen-suppressed.
    assert observed["nBonds"] == 1
    assert observed["nBondsS"] == 1
    assert observed["nBondsKS"] == 1
    assert mordred_values["nBonds"] == 7
    assert mordred_values["nBondsS"] == 7
    assert mordred_values["nBondsKS"] == 7


@pytest.mark.parametrize("smiles", ["CC", "c1ccccc1.C=C"])
def test_calculator_fills_bond_count_columns(smiles):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _expected_no_explicit_h(mol)

    assert observed.dtype == np.float32
    assert_allclose(observed[idxs], expected, rtol=0, atol=0)
