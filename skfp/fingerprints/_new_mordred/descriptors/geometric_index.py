import numpy as np
from rdkit.Chem import Atom, Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_mass,
    get_polarizability,
    get_sanderson_electronegativity,
    get_van_der_waals_volume,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix3D

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "GeomDiameter",
    "GeomRadius",
    "GeomShapeIndex",
    "GeomPetitjeanIndex",
]

@np.errstate(divide="ignore", invalid="ignore")
def calc(distance_matrix_3d: DistanceMatrix3D) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred geometrical index descriptors from 3D coordinates.
    """
    radius = distance_matrix_3d.radius
    diameter = distance_matrix_3d.diameter

    shape_index = np.nan if radius == 0.0 else (diameter - radius) / radius
    petitjean_index = np.nan if diameter == 0.0 else (diameter - radius) / diameter
    values = np.asarray(
        [diameter, radius, shape_index, petitjean_index], dtype=np.float32
    )
    return values, FEATURE_NAMES
