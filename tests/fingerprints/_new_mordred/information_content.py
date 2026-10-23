import json
from pathlib import Path

import numpy as np
import pytest
from mordred import Calculator
from mordred import InformationContent as MordredInformationContent
from numpy.testing import assert_allclose
from rdkit.Chem import Mol, MolFromSmiles, RenumberAtoms, SmilesParserParams
from rdkit.Chem.rdchem import Bond

from skfp.fingerprints._new_mordred.descriptors.information_content import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    bonds_apply_func,
    preprocess_mol,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values are stored at ./references/information_content.json as
expected[molecule][descriptor], and cover only the orders and molecules Mordred's own
test suite does not skip. They are rounded to three decimals.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "information_content.json") as f:
    _REFERENCE = json.load(f)

_MOLECULES = list(_REFERENCE["expected"])
_FEATURE_NAMES = _REFERENCE["feature_names"]


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols):
    computed = {}
    for name in _MOLECULES:
        mol_hydrogens = preprocess_mol(mordred_test_mols[name], explicit_hydrogens=True)
        mol_kekulized_hydrogens = preprocess_mol(
            mol_hydrogens, kekulize=True, sanitize=False
        )
        kekulized_bond_types_hydrogens = bonds_apply_func(
            Bond.GetBondType, mol_kekulized_hydrogens, np.intp
        )

        values = calc(
            mol_hydrogens,
            AtomicProperties(mol_hydrogens),
            kekulized_bond_types_hydrogens,
        )
        computed[name] = dict(zip(FEATURE_NAMES, values, strict=True))
    return computed


@pytest.mark.parametrize("molecule", _MOLECULES)
@pytest.mark.parametrize("feature_name", _FEATURE_NAMES)
def test_information_content_reference_values(feature_name, molecule, computed_values):
    if feature_name not in _REFERENCE["expected"][molecule]:
        pytest.skip("mordred skips this order for this molecule")

    expected = _REFERENCE["expected"][molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    atol = 0.05 if feature_name.startswith("modified information content") else 1e-3
    assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=True)


def test_feature_names():
    assert set(_FEATURE_NAMES).issubset(set(FEATURE_NAMES))


def _calc(mol: Mol) -> dict[str, float]:
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    kekulized_bond_types_hydrogens = bonds_apply_func(
        Bond.GetBondType,
        preprocess_mol(mol_hydrogens, kekulize=True, sanitize=False),
        np.intp,
    )
    values = calc(
        mol_hydrogens, AtomicProperties(mol_hydrogens), kekulized_bond_types_hydrogens
    )
    return dict(zip(FEATURE_NAMES, values, strict=True))


def _modified_information_content(mol: Mol) -> list[float]:
    values = _calc(mol)
    return [values[f"modified information content order {o}"] for o in range(6)]


def _parse_keeping_hydrogens(smiles: str) -> Mol:
    params = SmilesParserParams()
    params.removeHs = False
    return MolFromSmiles(smiles, params)


# isotope labels, including classes that mix isotopes of one element and classes of
# a single isotope
_ISOTOPE_SMILES = [
    "[2H]c1ccccc1C",
    "[2H]C([2H])([2H])O",
    "[2H]C([2H])CCO",
    "[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]",
    "[2H]Oc1ccccc1",
    "[13CH3]c1ccccc1",
]


@pytest.mark.parametrize("smiles", _ISOTOPE_SMILES)
def test_modified_information_content_of_isotopes_matches_mordred(smiles):
    # Mordred weighs a class by the mass of its last atom, which for these inputs
    # agrees with the atomic mass of the class's atoms, or, for a class mixing
    # isotopes, with the atomic weight of their element
    mol = MolFromSmiles(smiles)
    calculator = Calculator([MordredInformationContent.ModifiedIC(o) for o in range(6)])
    expected = [float(value) for value in calculator(mol)]

    assert_allclose(_modified_information_content(mol), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "mol",
    [
        *(MolFromSmiles(smiles) for smiles in _ISOTOPE_SMILES),
        # Mordred's values for this one change with the order of its atoms
        _parse_keeping_hydrogens("[H]C([H])([H])C([2H])=O"),
    ],
)
def test_modified_information_content_does_not_depend_on_atom_order(mol):
    reversed_mol = RenumberAtoms(mol, list(range(mol.GetNumAtoms()))[::-1])

    assert_allclose(
        _modified_information_content(reversed_mol),
        _modified_information_content(mol),
        rtol=1e-6,
    )
