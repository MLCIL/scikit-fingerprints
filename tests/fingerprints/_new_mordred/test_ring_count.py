import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import ring_count

FEATURE_NAMES = ring_count.FEATURE_NAMES
GENERAL_RING_FEATURE_NAMES = [
    "nRing",
    "nHRing",
    "naRing",
    "naHRing",
    "nARing",
    "nAHRing",
]

SMILES = [
    "C1CC1",
    "C1CCC1",
    "C1CCCC1",
    "C1CCCCC1",
    "C1CCCCCCCCCCCC1",
    "c1ccccc1",
    "c1ccncc1",
    "C1CCNCC1",
    "C1CCC2CCCCC2C1",
    "c1ccc2ccccc2c1",
    "C1CCC2(CC1)CCCC2",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize("smiles", SMILES)
def test_ring_count_matches_mordred(smiles, mordred_2d_calc):
    mol = Chem.MolFromSmiles(smiles)

    values, feature_names = ring_count.calc(mol)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    expected = np.asarray(
        [mordred_values[name] for name in feature_names], dtype=np.float32
    )

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_ring_count_includes_general_ring_descriptors():
    assert ring_count.FEATURE_NAMES[: len(GENERAL_RING_FEATURE_NAMES)] == (
        GENERAL_RING_FEATURE_NAMES
    )
