import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors import topological_index
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

FEATURE_NAMES = ["Diameter", "Radius", "TopoShapeIndex", "PetitjeanIndex"]


@pytest.mark.parametrize(
    "graph_radius, graph_diameter, expected",
    [
        # Diameter, Radius, TopoShapeIndex = (D - R) / R, PetitjeanIndex = (D - R) / D
        (2, 3, [3, 2, 0.5, 1 / 3]),  # e.g. butane
        (3, 3, [3, 3, 0.0, 0.0]),  # e.g. benzene
        (1, 2, [2, 1, 1.0, 0.5]),  # e.g. ethanol
        (1, 1, [1, 1, 0.0, 0.0]),  # e.g. ethane
        (0, 0, [0, 0, np.nan, np.nan]),  # single atom
    ],
)
def test_topological_index_values(graph_radius, graph_diameter, expected):
    values = topological_index.calc(graph_radius, graph_diameter)
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)


@pytest.mark.parametrize(
    # expected in FEATURE_NAMES order: Diameter, Radius, TopoShapeIndex, PetitjeanIndex
    "name, expected",
    [
        ("Hexane", [5, 3, 0.666666667, 0.4]),
        ("Benzene", [3, 3, 0.0, 0.0]),
        ("Caffeine", [6, 4, 0.5, 0.333333333]),
        ("Cyanidin", [10, 5, 1.0, 0.5]),
        ("Lycopene", [31, 16, 0.9375, 0.483870968]),
        ("Epicatechin", [10, 5, 1.0, 0.5]),
        ("Limonene", [6, 3, 1.0, 0.5]),
        ("Allicin", [7, 4, 0.75, 0.428571429]),
        ("Glutathione", [12, 6, 1.0, 0.5]),
        ("Digoxin", [28, 14, 1.0, 0.5]),
        ("Capsaicin", [15, 8, 0.875, 0.466666667]),
        ("EllagicAcid", [9, 5, 0.8, 0.444444444]),
        ("Astaxanthin", [27, 14, 0.928571429, 0.481481481]),
    ],
)
def test_topological_index_reference_values(name, expected, mordred_test_mols):
    mol_regular = preprocess_mol(mordred_test_mols[name])
    distance_matrix_regular = DistanceMatrix(mol_regular)

    values = topological_index.calc(
        distance_matrix_regular.radius, distance_matrix_regular.diameter
    )
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
