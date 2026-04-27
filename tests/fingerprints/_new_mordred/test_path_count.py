import math

import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import path_count
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "MPC2",
    "MPC3",
    "MPC4",
    "MPC5",
    "MPC6",
    "MPC7",
    "MPC8",
    "MPC9",
    "MPC10",
    "TMPC10",
    "piPC1",
    "piPC2",
    "piPC3",
    "piPC4",
    "piPC5",
    "piPC6",
    "piPC7",
    "piPC8",
    "piPC9",
    "piPC10",
    "TpiPC10",
]

SMILES = ["C", "CC", "CCC", "C=C", "C#C", "c1ccccc1", "CC(C)C", "C.C"]


@pytest.fixture(scope="module")
def mordred_path_count_calc():
    return Calculator(descriptors.PathCount, ignore_3D=True)


def _mordred_values(mol, calc):
    values = dict(
        zip((str(desc) for desc in calc.descriptors), calc(mol), strict=False)
    )
    return np.asarray(
        [
            np.nan if isinstance(values[name], Missing) else values[name]
            for name in FEATURE_NAMES
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_path_count_matches_mordred(smiles, mordred_path_count_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = path_count.calc(cache)
    expected = _mordred_values(mol, mordred_path_count_calc)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_path_count_columns(smiles, mordred_path_count_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_values(mol, mordred_path_count_calc)

    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)


def test_pi_path_counts_are_log_scaled_for_multiple_bond_orders():
    cache = MordredMolCache.from_mol(Chem.MolFromSmiles("C=C"), use_3D=False)
    values, feature_names = path_count.calc(cache)
    observed = dict(zip(feature_names, values, strict=False))

    assert observed["piPC1"] == pytest.approx(math.log(2 + 1))
    assert observed["TpiPC10"] == pytest.approx(math.log(2 + 2 + 1))


def test_pi_total_includes_atom_count_for_aromatic_molecule(mordred_path_count_calc):
    mol = Chem.MolFromSmiles("c1ccccc1")
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = path_count.calc(cache)
    observed = dict(zip(feature_names, values, strict=False))
    expected = dict(
        zip(
            (str(desc) for desc in mordred_path_count_calc.descriptors),
            mordred_path_count_calc(mol),
            strict=False,
        )
    )

    assert observed["piPC1"] == pytest.approx(math.log(6 * 1.5 + 1))
    assert observed["TpiPC10"] == pytest.approx(expected["TpiPC10"])
    assert math.exp(observed["TpiPC10"]) - 1 > sum(
        math.exp(observed[f"piPC{order}"]) - 1 for order in range(1, 11)
    )
