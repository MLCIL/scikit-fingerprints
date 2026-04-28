import numpy as np
import pytest
from mordred import Calculator
from mordred import VertexAdjacencyInformation as MordredVAdjMat
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import vertex_adjacency_information
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = ["VAdjMat"]


@pytest.fixture(scope="module")
def mordred_vadjmat_calc():
    return Calculator(MordredVAdjMat.VertexAdjacencyInformation())


def _mordred_value(mol, mordred_vadjmat_calc):
    return mordred_vadjmat_calc(mol)["VAdjMat"]


@pytest.mark.parametrize("smiles", ["C", "[Na+]"])
def test_vertex_adjacency_information_returns_nan_for_zero_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = vertex_adjacency_information.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert np.isnan(values[0])


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC", 1.0),
        ("CCC", 2.0),
        ("c1ccccc1", 1 + np.log2(6)),
        ("CC.CC", 2.0),
    ],
)
def test_vertex_adjacency_information_reference_values(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = vertex_adjacency_information.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert values.dtype == np.float32
    assert_allclose(values, [expected], rtol=1e-6, atol=1e-6)


def test_vertex_adjacency_information_preserves_heavy_heavy_semantics():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = vertex_adjacency_information.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, [1.0], rtol=1e-6, atol=1e-6)


def test_calculator_fills_vertex_adjacency_information_column():
    mol = Chem.MolFromSmiles("CCC")
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]

    assert_equal(observed[idxs].dtype, np.float32)
    assert_allclose(
        observed[idxs],
        vertex_adjacency_information.calc(cache)[0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_vertex_adjacency_information_feature_order():
    names = FEATURE_NAMES_2D

    assert names.index("Vabc") < names.index("VAdjMat") < names.index("MWC01")


@pytest.mark.parametrize("smiles", ["CC", "CCC", "c1ccccc1", "CC.CC"])
def test_vertex_adjacency_information_matches_mordred(smiles, mordred_vadjmat_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = vertex_adjacency_information.calc(cache)
    expected = _mordred_value(mol, mordred_vadjmat_calc)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, [expected], rtol=1e-6, atol=1e-6)
