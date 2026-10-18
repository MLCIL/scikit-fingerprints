import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.AllChem import EmbedMolecule

from skfp.fingerprints._new_mordred.descriptors.cpsa import (
    FEATURE_NAMES_2D,
    FEATURE_NAMES_3D,
    calc_2d,
    calc_3d,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import gasteiger_charges
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values generated from mordred-community and stored at
./references/cpsa.json as expected[molecule][descriptor].

Surface-area-dependent descriptors are compared with rtol=0.05, matching the
5% relative tolerance mordred itself uses for SASA (see mordred/tests/test_SASA.py).
mordred-community reproduces these reference values exactly. RNCS/RPCS, however,
hinge on a single atom's SASA rather than the total: RDKit's per-atom SASA
diverges too much from mordred's own SurfaceArea there (even though total SASA
agrees within 5%), so they are marked xfail.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "cpsa.json") as f:
    _REFERENCE = json.load(f)

_MOLECULES = list(_REFERENCE)

_PER_ATOM_SASA = {"RNCS", "RPCS"}

_FEATURE_NAMES = [
    pytest.param(
        name,
        marks=pytest.mark.xfail(
            reason="depends on per-atom SASA; RDKit and mordred diverge",
            strict=False,
        ),
    )
    if name in _PER_ATOM_SASA
    else name
    for name in [*FEATURE_NAMES_2D, *FEATURE_NAMES_3D]
]


@pytest.fixture(scope="module")
def computed_values(mordred_test_mols_hydrogens_3d):
    computed = {}
    for name in _MOLECULES:
        mol = mordred_test_mols_hydrogens_3d[name]
        charges = gasteiger_charges(mol)
        cpsa_2d = calc_2d(charges)
        values_3d = calc_3d(mol, cpsa_2d, charges)
        computed[name] = dict(zip(FEATURE_NAMES_2D, cpsa_2d, strict=True)) | dict(
            zip(FEATURE_NAMES_3D, values_3d, strict=True)
        )
    return computed


@pytest.mark.parametrize("molecule", _MOLECULES)
@pytest.mark.parametrize("feature_name", _FEATURE_NAMES)
def test_cpsa_reference_values(feature_name, molecule, computed_values):
    expected = _REFERENCE[molecule][feature_name]
    actual = computed_values[molecule][feature_name]
    assert_allclose(actual, expected, rtol=0.05, equal_nan=True)


def _embedded(smiles: str) -> Mol:
    mol = preprocess_mol(MolFromSmiles(smiles), explicit_hydrogens=True)
    EmbedMolecule(mol, randomSeed=0)
    return mol


def _compute(mol: Mol) -> dict[str, float]:
    charges = gasteiger_charges(mol)
    cpsa_2d = calc_2d(charges)
    values_3d = calc_3d(mol, cpsa_2d, charges)
    return dict(zip(FEATURE_NAMES_2D, cpsa_2d, strict=True)) | dict(
        zip(FEATURE_NAMES_3D, values_3d, strict=True)
    )


def test_relative_charges_without_atoms_of_a_sign_are_zero():
    # as in Mordred, a sign without atoms gives 0.0 rather than an undefined value
    assert_allclose(calc_2d(np.array([0.3, 0.1, 0.0])), [0.0, 0.75])
    assert_allclose(calc_2d(np.array([-0.2, -0.2])), [0.5, 0.0])


def test_descriptors_without_charged_atoms_follow_mordred():
    # the Gasteiger charges of dioxygen are all zero, so it has no charged atoms
    values = _compute(_embedded("O=O"))

    for name in ["RNCG", "RPCG", "RNCS", "RPCS", "TPSA", "RPSA"]:
        assert values[name] == 0.0, name
    for prefix in ["PNSA", "PPSA", "DPSA", "FNSA", "FPSA", "WNSA", "WPSA"]:
        # sums over no atoms, but version 5 divides by their number
        assert_allclose([values[f"{prefix}{v}"] for v in range(1, 5)], 0.0)
        assert np.isnan(values[f"{prefix}5"]), prefix
    assert values["TASA"] > 0.0
    assert_allclose(values["RASA"], 1.0, rtol=1e-6)


def test_missing_gasteiger_charge_makes_all_values_nan():
    # Gasteiger has no parameters for sodium, so like Mordred nothing is defined
    mol = _embedded("CC(=O)O.[NaH]")
    assert np.isnan(gasteiger_charges(mol)).any()

    values = _compute(mol)
    assert_allclose(list(values.values()), np.nan)
