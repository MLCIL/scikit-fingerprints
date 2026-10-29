import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Kekulize, Mol, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.aromatic import FEATURE_NAMES, calc
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values come from ChemAxon cxcalc (as bundled with mordred-community)
and are stored at ./references/aromatic.json as expected[molecule][descriptor].

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "aromatic.json") as f:
    _REFERENCE = json.load(f)


@pytest.mark.parametrize("molecule", list(_REFERENCE))
def test_aromatic_reference_values(molecule, mordred_test_mols):
    mol = mordred_test_mols[molecule]

    values = _calc(mol)
    expected = [_REFERENCE[molecule][name] for name in FEATURE_NAMES]
    assert_allclose(values, expected)


@pytest.mark.parametrize("smiles", ["c1ccc2ccccc2c1", "Cc1ccncc1", "c1ccsc1"])
def test_aromatic_counts_of_kekulized_input(smiles):
    # Mordred reads the input as given, which is not re-sanitized: an input
    # kekulized without aromatic flags has no aromatic atoms or bonds
    mol = MolFromSmiles(smiles)
    Kekulize(mol, clearAromaticFlags=True)

    assert_allclose(_calc(mol), [0, 0])


def _calc(mol: Mol):
    return calc(AtomicProperties(preprocess_mol(mol, explicit_hydrogens=True)))
