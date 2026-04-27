import numpy as np
from numpy.testing import assert_allclose
from rdkit import Chem
from rdkit.Chem import AllChem

from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    DistanceMatrix,
    DistanceMatrix3D,
)


def test_distance_matrix_radius_uses_cached_eccentricities():
    mol = Chem.MolFromSmiles("CCO")
    distance_matrix = DistanceMatrix(mol)

    assert distance_matrix.radius == 1


def test_3d_distance_matrix_radius_uses_cached_eccentricities():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    distance_matrix = DistanceMatrix3D(mol)

    expected = np.min(np.max(distance_matrix.matrix, axis=0))
    assert_allclose(distance_matrix.radius, expected)
