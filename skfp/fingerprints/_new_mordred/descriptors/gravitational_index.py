import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix3D

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "GRAV",
    "GRAV with hydrogens",
    "bond-restricted GRAV",
    "bond-restricted GRAV with hydrogens",
]


def calc(
    props_regular: AtomicProperties,
    props_hydrogens: AtomicProperties,
    distance_matrix_3d_regular: DistanceMatrix3D,
    distance_matrix_3d_hydrogens: DistanceMatrix3D,
) -> np.ndarray:
    """
    Compute Mordred gravitational index descriptors.

    The gravitational index sums ``m_i * m_j / D_ij ** 2`` (Newton's gravitation
    analogy) over atom pairs, where ``m`` are atomic masses and ``D`` are 3D
    interatomic distances.

    The four variants are the 2x2 product of two flags:

    - ``H`` suffix: use the hydrogen-explicit molecule (``props_hydrogens``)
      instead of the heavy-atom-only molecule (``props_regular``, with hydrogens
      suppressed).
    - ``p`` suffix: sum only over bonded pairs instead of over all atom pairs.

    Each molecule comes with its own 3D distance matrix. The hydrogen-suppressed
    molecule still carries the 3D conformer (``RemoveHs`` preserves the heavy-atom
    coordinates), so its 3D distance matrix is well defined.
    """
    grav, grav_pair = _variant_values(props_regular, distance_matrix_3d_regular)
    grav_h, grav_h_pair = _variant_values(props_hydrogens, distance_matrix_3d_hydrogens)

    values = np.asarray([grav, grav_h, grav_pair, grav_h_pair], dtype=np.float32)
    return values


def _variant_values(
    props: AtomicProperties, distance_matrix_3d: DistanceMatrix3D
) -> tuple[np.float32, np.float32]:
    """
    Return the (all-pairs, bonded-pairs) gravitational indices for one molecule.

    Mordred sums the bonded pairs as ``0.5 * sum(W * A)``. ``A`` is 1 at ``[b, e]``
    and ``[e, b]`` for each bond, and ``W`` is symmetric, so this equals the sum of
    ``W[b, e]`` over the bonds, read directly from the bond index arrays.
    """
    # RDKit masses, not the Mordred mass table used by the other descriptors
    masses = props.masses.astype(np.float32)
    mass_products = masses[:, np.newaxis] * masses
    np.fill_diagonal(mass_products, 0.0)

    # float64 for the copy (astype also avoids mutating the cached matrix): the
    # inverse-square and the summation over all pairs are precision-sensitive, so
    # keep them in double precision. Diagonal distances are 0, so set them to 1
    # to avoid division by zero (the mass diagonal is 0, so these terms vanish).
    distances = distance_matrix_3d.matrix.astype(np.float64)
    # Mordred divides by every squared distance, so two atoms in the same place,
    # e.g. ions that conformer generation stacked, leave both indices undefined
    np.fill_diagonal(distances, 1.0)
    if (distances == 0.0).any():
        return np.float32(np.nan), np.float32(np.nan)
    inverse_squared_distances = distances**-2

    weighted = mass_products * inverse_squared_distances
    all_pairs = 0.5 * np.sum(weighted)
    bonded_pairs = weighted[props.bond_begin_idxs, props.bond_end_idxs].sum()
    # cast the final scalars to float32 (the fingerprint's dtype); the heavy math
    # above stays in float64
    return np.float32(all_pairs), np.float32(bonded_pairs)
