import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import vdw_volume_abc
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["Vabc"]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


def _mordred_values(mol, mordred_2d_calc):
    return dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("C", 20.579526276115534),
        ("CCO", 44.029279503721554),
        ("c1ccccc1", 73.25715765669321),
        ("C1CCCCC1", 84.15715765669322),
        ("c1ccc2ccccc2c1", 111.27526276115532),
    ],
)
def test_vdw_volume_abc_reference_values(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = vdw_volume_abc.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, [expected], rtol=1e-6, atol=1e-6)


def test_vdw_volume_abc_returns_nan_for_unsupported_element():
    mol = Chem.MolFromSmiles("[Og]")
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = vdw_volume_abc.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values[0])


@pytest.mark.parametrize("smiles", ["C", "CCO"])
def test_vdw_volume_abc_differs_from_mordred_default(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, _ = vdw_volume_abc.calc(cache)
    mordred_values = _mordred_values(mol, mordred_2d_calc)

    assert values[0] != mordred_values["Vabc"]


def test_calculator_fills_vdw_volume_abc_column():
    mol = Chem.MolFromSmiles("CCO")
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]

    assert_equal(observed[idxs].dtype, np.float32)
    assert_allclose(
        observed[idxs],
        vdw_volume_abc.calc(cache)[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_vdw_volume_abc_feature_order():
    names = FEATURE_NAMES_2D

    assert names.index("PetitjeanIndex") < names.index("Vabc") < names.index("VAdjMat")
