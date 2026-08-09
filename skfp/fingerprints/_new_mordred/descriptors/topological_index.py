import numpy as np

FEATURE_NAMES = ["Diameter", "Radius", "TopoShapeIndex", "PetitjeanIndex"]


"""
Topological index descriptors (radius, diameter, shape and Petitjean indices).

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def calc(graph_radius: float, graph_diameter: float) -> np.ndarray:
    r"""
    Compute the Mordred topological index descriptors.

    Given the graph radius :math:`R` and diameter :math:`D`:

    - ``Diameter`` :math:`= D`
    - ``Radius`` :math:`= R`
    - ``TopoShapeIndex`` :math:`= (D - R) / R`, NaN when :math:`R = 0`
    - ``PetitjeanIndex`` :math:`= (D - R) / D`, NaN when :math:`D = 0`
    """
    topo_shape_index = (
        np.nan if graph_radius == 0 else (graph_diameter - graph_radius) / graph_radius
    )
    petitjean_index = (
        np.nan
        if graph_diameter == 0
        else (graph_diameter - graph_radius) / graph_diameter
    )

    values = np.asarray(
        [graph_diameter, graph_radius, topo_shape_index, petitjean_index],
        dtype=np.float32,
    )
    return values
