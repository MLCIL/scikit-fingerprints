import numpy as np
from rdkit.Chem import Atom, Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_mass,
    get_polarizability,
    get_sanderson_electronegativity,
    get_van_der_waals_volume,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# atomic properties used to weight distance matrices
_PROPS_NAMES = [
    "none",  # pure interatomic distance
    "mass",
    "van_der_Waals_volume",
    "Sanderson_electronegativity",
    "polarizability",
]
_PROPS_FUNCS = [
    None,  # pure interatomic distance
    get_mass,
    get_van_der_waals_volume,
    get_sanderson_electronegativity,
    get_polarizability,
]


FEATURE_NAMES = [
    f"MoRSE_{prop}_dist_{dist}" for prop in _PROPS_NAMES for dist in range(33)
]


@np.errstate(divide="ignore", invalid="ignore")
def calc(
    mol_3d: Mol, distance_matrix_3d: DistanceMatrix
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

    values = []
    for name, func in zip(_PROPS_NAMES, _PROPS_FUNCS, strict=True):
        for distance in range(1, 33):
            if name == "none":
                props = np.ones(num_atoms)
            else:
                props = np.asarray([func(atom) for atom in atoms])
                carbon_prop = func(Atom(6))
                props = props / carbon_prop

            props = props.reshape(-1, 1)

            if distance == 1:
                n = np.ones((num_atoms, num_atoms), dtype=np.float32)
            else:
                sr = (distance - 1) * distance_matrix_3d.matrix
                np.fill_diagonal(sr, 1)
                n = np.sin(sr) / sr

            np.fill_diagonal(n, 0)

            value = 0.5 * np.ravel(props @ n @ props.T)[0]
            values.append(value)

    return np.asarray(values, np.float32), FEATURE_NAMES
