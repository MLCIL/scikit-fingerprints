import numpy as np
import pytest
from mordred import Calculator, descriptors
from numpy.testing import assert_allclose
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors import atom_count

FEATURE_NAMES = [
    "nAtom",
    "nHeavyAtom",
    "nSpiro",
    "nBridgehead",
    "nHetero",
    "nH",
    "nB",
    "nC",
    "nN",
    "nO",
    "nS",
    "nP",
    "nF",
    "nCl",
    "nBr",
    "nI",
    "nX",
]


@pytest.fixture(scope="module")
def mordred_2d_calc():
    return Calculator(descriptors, ignore_3D=True)


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "c1ccccc1",
        "C(F)(Cl)(Br)I",
        "O=P(O)(O)O",
        "CS(=O)(=O)N",
        "C[N+](C)(C)C",
    ],
)
def test_atom_count_matches_mordred(smiles, mordred_2d_calc):
    mol = MolFromSmiles(smiles)

    values, feature_names = atom_count.calc(mol)
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
