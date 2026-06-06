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

# atomic properties used to weight distance matrices
_PROPS = [
    ("unweighted", None),  # pure interatomic distance
    ("mass", get_mass),
    ("van_der_Waals_volume", get_van_der_waals_volume),
    ("Sanderson_electronegativity", get_sanderson_electronegativity),
    ("polarizability", get_polarizability),
]
_DISTANCES = range(1, 33)

FEATURE_NAMES = [
    f"MoRSE_{prop_name}_dist_{dist}"
    for prop_name, prop_func in _PROPS
    for dist in _DISTANCES
]


@np.errstate(divide="ignore", invalid="ignore")
def calc(
    mol_3d: Mol, distance_matrix_3d: DistanceMatrix3D
) -> tuple[np.ndarray, list[str]]:
    """
    MoRSE descriptors.

    Quantifies correlation of 3D interatomic distances, weighted by various
    properties. Property values are normalized by the value for carbon prior
    to weighting.
    """
    atoms = list(mol_3d.GetAtoms())
    num_atoms = len(atoms)

    if num_atoms < 2:
        return np.full(160, np.nan, dtype=np.float32), FEATURE_NAMES

    kernels = []
    for distance in _DISTANCES:
        if distance == 1:
            n = np.ones((num_atoms, num_atoms), dtype=np.float32)
        else:
            sr = (distance - 1) * distance_matrix_3d.matrix
            np.fill_diagonal(sr, 1.0)
            n = np.sin(sr) / sr
        np.fill_diagonal(n, 0.0)
        kernels.append(n)

    prop_vectors = []
    for name, func in _PROPS:
        if name == "unweighted":
            props = np.ones(num_atoms)
        else:
            props = np.fromiter(
                (func(a) for a in atoms),  # type: ignore
                dtype=np.float32,
                count=num_atoms,
            )
            props = props / func(Atom(6))  # type: ignore
        prop_vectors.append(props)

    values = [0.5 * (props @ n @ props) for props in prop_vectors for n in kernels]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
