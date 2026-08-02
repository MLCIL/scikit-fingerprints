import itertools

import numpy as np

from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.matrix_attributes import MatrixAttributes

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "SpAbs_Dt",
    "SpMax_Dt",
    "SpDiam_Dt",
    "SpAD_Dt",
    "SpMAD_Dt",
    "LogEE_Dt",
    "SM1_Dt",
    "VE1_Dt",
    "VE2_Dt",
    "VE3_Dt",
    "VR1_Dt",
    "VR2_Dt",
    "VR3_Dt",
    "DetourIndex",
]


def calc(
    atomic_props_regular: AtomicProperties,
    distance_matrix_regular: DistanceMatrix,
    rings_regular: RingSets,
    n_frags: int,
) -> np.ndarray:
    """
    Detour matrix descriptor.

    Computes matrix-aggregating features (spectral and Randic-like) of the
    detour matrix, together with the detour index. The detour matrix is
    undefined for disconnected molecules, so NaN is returned for those.
    """
    # avoids unnecessary eigendecomposition for disconnected molecules
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    detour_matrix = _get_detour_matrix(
        atomic_props_regular,
        distance_matrix_regular.matrix,
        rings_regular.simple_ring_atom_sets,
    )
    attrs = MatrixAttributes(
        detour_matrix,
        atomic_props_regular,
        hermitian=True,
        n_frags=n_frags,
    )

    values = np.asarray(
        [
            attrs.graph_energy,  # SpAbs_Dt
            attrs.leading_eigenvalue,  # SpMax_Dt
            attrs.spectral_diameter,  # SpDiam_Dt
            attrs.sp_ad,  # SpAD_Dt
            attrs.sp_mad,  # SpMAD_Dt
            attrs.log_ee,  # LogEE_Dt
            attrs.sm1,  # SM1_Dt
            attrs.ve1,  # VE1_Dt
            attrs.ve2,  # VE2_Dt
            attrs.ve3,  # VE3_Dt
            attrs.vr1,  # VR1_Dt
            attrs.vr2,  # VR2_Dt
            attrs.vr3,  # VR3_Dt
            # detour index = sum over unordered atom pairs - the matrix is
            # symmetric, so halving the full sum avoids double-counting each pair
            int(0.5 * detour_matrix.sum()),  # DetourIndex
        ],
        dtype=np.float32,
    )

    return values


def _get_detour_matrix(
    props: AtomicProperties,
    shortest_distances: np.ndarray,
    ring_atom_sets: list[set[int]],
) -> np.ndarray:
    """
    Build the detour (maximum topological distance) matrix of a molecule.

    Entry ``(i, j)`` is the length of the longest simple path between atoms ``i``
    and ``j``.

    Outside the rings there is only one simple path between two atoms, so the
    detour distance is the ordinary distance; a path can differ from the shortest
    one only where it crosses a ring system. Every ring system is added on top of
    the distance matrix as a correction, how much longer the longest way through it
    is than the shortest one, see _ring_system_correction().

    A simple path can never leave a ring system and come back, since that would
    mean visiting the atom it left through twice, which is what makes the
    corrections of the ring systems independent of one another.
    """
    detour = shortest_distances.copy()

    for ring_system in _get_ring_systems(ring_atom_sets):
        atoms = np.fromiter(sorted(ring_system), dtype=np.intp, count=len(ring_system))
        detour += _get_ring_system_correction(props, shortest_distances, atoms)

    return detour


def _get_ring_systems(ring_atom_sets: list[set[int]]) -> list[set[int]]:
    """
    Merge the rings that share at least two atoms into ring systems.

    Two rings sharing that many atoms can be traversed as one, while rings meeting
    at a single (spiro) atom cannot: a path through one of them and into the other
    would have to pass that atom twice.
    """
    systems: list[set[int]] = []
    for ring in ring_atom_sets:
        merged = set(ring)
        rest = []
        for system in systems:
            if len(system & merged) >= 2:
                merged |= system
            else:
                rest.append(system)
        systems = [*rest, merged]

    # merging two systems can bring a third within reach, so repeat until stable
    while True:
        for first, second in itertools.combinations(systems, 2):
            if len(first & second) >= 2:
                systems.remove(second)
                first |= second
                break
        else:
            return systems


def _get_ring_system_correction(
    props: AtomicProperties, shortest_distances: np.ndarray, atoms: np.ndarray
) -> np.ndarray:
    """
    How much longer the longest way through one ring system is than the shortest,
    for every pair of atoms of the molecule.

    A path between two atoms enters the ring system at the atom of it nearest to
    where the path starts and leaves at the one nearest to where it ends, which is
    where the correction for that pair is read off. Pairs entering and leaving at
    the same atom never traverse the system, and their correction is zero.
    """
    within = shortest_distances[atoms[:, np.newaxis], atoms]
    longest = _longest_paths_in_ring_system(props, atoms, within)

    # every atom reaches the ring system through the one of its atoms closest to
    # it, since all of its paths into the system pass through that atom
    gates = np.argmin(shortest_distances[:, atoms], axis=1)
    return (longest - within)[gates[:, np.newaxis], gates]


def _longest_paths_in_ring_system(
    props: AtomicProperties, atoms: np.ndarray, within: np.ndarray
) -> np.ndarray:
    """
    Longest simple path between every two atoms of one ring system.
    """
    neighbors = _get_neighbors_adjacency_lists(props, atoms)

    if all(len(adjacent) == 2 for adjacent in neighbors.values()):
        # a plain ring: the two ways round it between two atoms make up the whole
        # ring, so the longer one is however much of it the shorter one leaves
        longest = len(atoms) - within
        np.fill_diagonal(longest, 0.0)
        return longest

    # fused or bridged rings have to be searched through
    longest = np.zeros_like(within)
    position = {atom: idx for idx, atom in enumerate(atoms.tolist())}
    for (start, end), distance in _get_longest_simple_paths(neighbors).items():
        longest[position[start], position[end]] = distance
        longest[position[end], position[start]] = distance
    return longest


def _get_neighbors_adjacency_lists(
    props: AtomicProperties, atoms: np.ndarray
) -> dict[int, list[int]]:
    """
    Adjacency lists of a set of atoms, counting only the bonds between them.
    """
    inside = np.zeros(props.num_atoms, dtype=bool)
    inside[atoms] = True
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs
    within = inside[begins] & inside[ends]

    neighbors: dict[int, list[int]] = {atom: [] for atom in atoms.tolist()}
    for begin, end in zip(begins[within].tolist(), ends[within].tolist(), strict=True):
        neighbors[begin].append(end)
        neighbors[end].append(begin)

    return neighbors


def _get_longest_simple_paths(G: dict[int, list[int]]) -> dict[tuple[int, int], int]:
    """
    Longest simple path length between every pair of nodes of a ring system.

    Returns a mapping ``(i, j) -> distance`` with ``i < j``. Solved by brute-force
    DFS, which is affordable because the fused ring systems of molecules are small.
    This is pessimistically exponential, as the longest simple path is NP-hard for
    graphs in general.
    """
    longest_distances: dict[tuple[int, int], int] = {}

    # s: source node, e: end node
    for s in G:
        for e, dist in _get_longest_paths_from_source(G, s).items():
            if s < e:
                longest_distances[(s, e)] = dist

    return longest_distances


def _get_longest_paths_from_source(
    G: dict[int, list[int]],
    s: int,
) -> dict[int, int]:
    """
    Longest simple path length from a single source ``s`` to every node.

    Returns a mapping ``node -> distance``, with 0 for the source itself.
    """
    result = dict.fromkeys(G, 0)
    visited = {s}
    _dfs(s, 0, G, visited, result)
    return result


def _dfs(
    u: int,
    dist: int,
    G: dict[int, list[int]],
    visited: set[int],
    result: dict[int, int],
) -> None:
    """
    Recursive DFS with backtracking for :func:`_longest_paths_from`.

    Explores every simple path leaving ``u`` and records in ``result`` the
    maximum distance reached at each visited node.
    """
    dist += 1

    # u: current node, v: neighbor being explored
    for v in G[u]:
        if v in visited:
            continue

        result[v] = max(result[v], dist)

        visited.add(v)
        _dfs(v, dist, G, visited, result)
        visited.remove(v)
