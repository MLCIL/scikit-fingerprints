import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose, assert_equal
from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import (
    atom_count,
    carbon_types,
    rdkit_descriptors,
    ring_count,
    rotatable_bond,
)
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)

CALCULATOR_2D_FEATURE_NAMES = [
    *rdkit_descriptors.FEATURE_NAMES_2D,
    *atom_count.FEATURE_NAMES,
    *carbon_types.FEATURE_NAMES,
    *rotatable_bond.FEATURE_NAMES,
    *ring_count.FEATURE_NAMES,
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.fixture(scope="module")
def mordred_all_calc():
    return Calculator(descriptors, ignore_3D=False)


def test_calculator_fills_2d_descriptor_columns(mordred_2d_calc):
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")

    observed = compute(mol, use_3D=False)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_2d_calc.descriptors),
            mordred_2d_calc(mol),
            strict=False,
        )
    )
    idxs = [FEATURE_NAMES_2D.index(name) for name in CALCULATOR_2D_FEATURE_NAMES]
    expected_values = np.asarray(
        [mordred_values[name] for name in CALCULATOR_2D_FEATURE_NAMES],
        dtype=np.float32,
    )

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], expected_values, rtol=1e-6, atol=1e-6)


def test_calculator_fills_3d_descriptor_columns(mordred_all_calc):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    observed = compute(mol, use_3D=True)
    mordred_values = dict(
        zip(
            (str(desc) for desc in mordred_all_calc.descriptors),
            mordred_all_calc(mol),
            strict=False,
        )
    )
    idxs = [
        ALL_FEATURE_NAMES.index(name) for name in rdkit_descriptors.FEATURE_NAMES_3D
    ]
    expected_values = np.asarray(
        [mordred_values[name] for name in rdkit_descriptors.FEATURE_NAMES_3D],
        dtype=np.float32,
    )

    assert_equal(np.isnan(observed[idxs]), False)
    assert_allclose(observed[idxs], expected_values, rtol=1e-6, atol=1e-6)
