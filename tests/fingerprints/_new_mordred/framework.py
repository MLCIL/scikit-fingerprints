import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import AddHs, Mol, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.framework import calc
from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values come from mordred-community's own reference data
(mordred/tests/references/Framework.yaml) and are stored at
./references/framework.json as expected[molecule][descriptor]. Cyanidin is
commented out there and is left out here too. The additional cases below were
computed with mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "framework.json") as f:
    _REFERENCE = json.load(f)


def _calc(mol: Mol) -> float:
    props = AtomicProperties.from_mol(mol)
    values = calc(props, RingSets(mol, props), AddHs(mol).GetNumAtoms())
    return float(values[0])


def _calc_from_smiles(smiles: str) -> float:
    return _calc(preprocess_mol(MolFromSmiles(smiles)))


@pytest.mark.parametrize("molecule", list(_REFERENCE))
def test_framework_reference_values(molecule, mordred_test_mols):
    value = _calc(mordred_test_mols[molecule])

    assert_allclose(value, _REFERENCE[molecule]["fMF"], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "smiles, expected",
    [
        # no rings at all, so the framework is empty
        ("N", 0.0),
        ("CCCC", 0.0),
        ("[Na+].[Cl-]", 0.0),
        # a single ring has no other ring to link to; the denominator counts
        # hydrogens, which is why benzene is 6/12 rather than 1
        ("c1ccccc1", 0.5),
        ("C1CCCCC1", 1 / 3),
        ("C1CC1", 1 / 3),
        # a chain hanging off a lone ring is not a linker: 6/24
        ("c1ccccc1CCCC", 0.25),
        # fused and directly bonded rings have nothing between them
        ("c1ccc2ccccc2c1", 10 / 18),
        ("c1ccccc1c1ccccc1", 12 / 22),
        # atoms between two rings are linkers, and count towards the framework
        ("c1ccccc1Cc1ccccc1", 13 / 25),
        ("c1ccccc1CCCCc1ccccc1", 16 / 34),
        # only the bridge counts here, not the two carboxyl groups
        ("OC(=O)c1ccccc1CCc1ccccc1C(=O)O", 14 / 34),
        # two rings with no path between them contribute no linkers
        ("c1ccccc1.c1ccccc1", 0.5),
    ],
)
def test_framework_values(smiles, expected):
    assert_allclose(_calc_from_smiles(smiles), expected, rtol=1e-6, atol=1e-7)


def test_hydrogens_count_towards_the_denominator():
    # the ratio is over every atom including hydrogens, so the same ring system
    # gives a different value for the saturated and the aromatic molecule
    assert_allclose(_calc_from_smiles("c1ccccc1"), 6 / 12, rtol=1e-6)
    assert_allclose(_calc_from_smiles("C1CCCCC1"), 6 / 18, rtol=1e-6)


def test_empty_molecule_is_nan():
    values = calc(
        AtomicProperties.from_mol(MolFromSmiles("")),
        RingSets(MolFromSmiles(""), AtomicProperties.from_mol(MolFromSmiles(""))),
        0,
    )

    assert np.all(np.isnan(values))
