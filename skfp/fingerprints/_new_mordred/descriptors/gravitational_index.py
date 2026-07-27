import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix3D,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["GRAV", "GRAVH", "GRAVp", "GRAVHp"]


def calc(
    mol_regular: Mol,
    mol_hydrogens_conformer: Mol,
    distance_matrix_3d_regular: DistanceMatrix3D,
    distance_matrix_3d_hydrogens: DistanceMatrix3D,
    adjacency_matrix_regular: AdjacencyMatrix,
    adjacency_matrix_hydrogens: AdjacencyMatrix,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred gravitational index descriptors.

    The gravitational index sums ``m_i * m_j / D_ij ** 2`` (Newton's gravitation
    analogy) over atom pairs, where ``m`` are atomic masses and ``D`` are 3D
    interatomic distances.

    The four variants are the 2x2 product of two flags:

    - ``H`` suffix: use the hydrogen-explicit molecule
      (``mol_hydrogens_conformer``) instead of the heavy-atom-only molecule
      (``mol_regular``, with hydrogens suppressed).
    - ``p`` suffix: sum only over bonded pairs (adjacency matrix) instead of
      over all atom pairs.

    All distance and adjacency matrices are injected by the calculator.
    ``mol_regular`` still carries the 3D conformer (``RemoveHs`` preserves the
    heavy-atom coordinates), so its 3D distance matrix is well defined.
    """
    grav, grav_pair = _variant_values(
        mol_regular, distance_matrix_3d_regular, adjacency_matrix_regular
    )
    grav_h, grav_h_pair = _variant_values(
        mol_hydrogens_conformer,
        distance_matrix_3d_hydrogens,
        adjacency_matrix_hydrogens,
    )

    values = np.asarray([grav, grav_h, grav_pair, grav_h_pair], dtype=np.float32)
    return values, FEATURE_NAMES


def _variant_values(
    mol: Mol,
    distance_matrix_3d: DistanceMatrix3D,
    adjacency_matrix: AdjacencyMatrix,
) -> tuple[np.float32, np.float32]:
    """
    Return the (all-pairs, bonded-pairs) gravitational indices for one molecule.
    """
    masses = np.asarray([atom.GetMass() for atom in mol.GetAtoms()], dtype=np.float32)
    mass_products = masses[:, np.newaxis] * masses
    np.fill_diagonal(mass_products, 0.0)

    # float64 for the copy (astype also avoids mutating the cached matrix): the
    # inverse-square and the summation over all pairs are precision-sensitive, so
    # keep them in double precision. Diagonal distances are 0, so set them to 1
    # to avoid division by zero (the mass diagonal is 0, so these terms vanish).
    distances = distance_matrix_3d.matrix.astype(np.float64)
    np.fill_diagonal(distances, 1.0)
    inverse_squared_distances = distances**-2

    weighted = mass_products * inverse_squared_distances
    all_pairs = 0.5 * np.sum(weighted)
    bonded_pairs = 0.5 * np.sum(weighted * adjacency_matrix.matrix)
    # cast the final scalars to float32 (the fingerprint's dtype); the heavy math
    # above stays in float64
    return np.float32(all_pairs), np.float32(bonded_pairs)
