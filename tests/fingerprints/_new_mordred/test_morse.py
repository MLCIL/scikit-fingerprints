import importlib

import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)

MORSE_FEATURE_NAMES = [
    *[f"Mor{i:02d}" for i in range(1, 33)],
    *[f"Mor{i:02d}m" for i in range(1, 33)],
    *[f"Mor{i:02d}v" for i in range(1, 33)],
    *[f"Mor{i:02d}se" for i in range(1, 33)],
    *[f"Mor{i:02d}p" for i in range(1, 33)],
]


@pytest.fixture(scope="module")
def mordred_morse_calc():
    return Calculator(descriptors.MoRSE, ignore_3D=False)


def _embedded_mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _mordred_expected(mol, mordred_calc):
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_calc.descriptors),
            mordred_calc(mol),
            strict=False,
        )
    )
    return np.asarray(
        [
            np.nan
            if isinstance(mordred_values[name], Missing)
            else mordred_values[name]
            for name in MORSE_FEATURE_NAMES
        ],
        dtype=np.float32,
    )


def test_morse_feature_names_are_in_mordred_order():
    morse = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.morse")

    assert morse.FEATURE_NAMES == MORSE_FEATURE_NAMES
    assert len(morse.FEATURE_NAMES) == 160


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "c1ccccc1"])
def test_morse_descriptors_match_mordred(smiles, mordred_morse_calc):
    mol = _embedded_mol(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    morse = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.morse")

    values, feature_names = morse.calc(cache)
    expected = _mordred_expected(mol, mordred_morse_calc)

    assert feature_names == MORSE_FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", ["C", "CC", "CCO", "c1ccccc1"])
def test_calculator_fills_morse_columns(smiles, mordred_morse_calc):
    mol = _embedded_mol(smiles)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in MORSE_FEATURE_NAMES]
    expected = _mordred_expected(mol, mordred_morse_calc)

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)


def test_morse_returns_nan_without_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    morse = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.morse")

    values, feature_names = morse.calc(cache)

    assert feature_names == MORSE_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_morse_returns_nan_without_conformer():
    mol = Chem.MolFromSmiles("CCO")

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in MORSE_FEATURE_NAMES]

    assert np.isnan(observed[idxs]).all()


def test_morse_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    morse = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.morse")

    values, feature_names = morse.calc(cache)

    assert feature_names == MORSE_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_calculator_morse_returns_nan_for_2d_conformer():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.Compute2DCoords(mol)

    observed = compute(mol, use_3D=True)
    idxs = [ALL_FEATURE_NAMES.index(name) for name in MORSE_FEATURE_NAMES]

    assert np.isnan(observed[idxs]).all()


def test_morse_returns_nan_for_single_atom_3d_molecule():
    mol = Chem.MolFromSmiles("[He]")
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    mol.AddConformer(conf)
    cache = MordredMolCache.from_mol(mol, use_3D=True)
    morse = importlib.import_module("skfp.fingerprints._new_mordred.descriptors.morse")

    values, feature_names = morse.calc(cache)

    assert feature_names == MORSE_FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values).all()


def test_morse_descriptors_are_3d_only():
    assert all(name not in FEATURE_NAMES_2D for name in MORSE_FEATURE_NAMES)
