import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors import topological_index

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
    values, feature_names = topological_index.calc(graph_radius, graph_diameter)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, np.asarray(expected, dtype=np.float32), rtol=1e-6)
