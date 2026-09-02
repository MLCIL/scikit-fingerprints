import itertools

import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

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
    ring_systems = _get_ring_systems(ring_atom_sets)

    # the rings are the ones RDKit perceives, which for cage-like molecules do not
    # always describe the whole cycle structure, see _are_ring_systems_complete()
    if not _are_ring_systems_complete(props, ring_systems):
        ring_systems = _get_bridgeless_components(props)

    # a molecule without rings is its own detour matrix, a single atom included,
    # so it is handled automatically
    detour = shortest_distances.copy()

    for ring_system in ring_systems:
        atoms = np.array(sorted(ring_system), dtype=np.intp)
        detour += _get_ring_system_correction(props, shortest_distances, atoms)

    return detour


def _get_ring_systems(ring_atom_sets: list[set[int]]) -> list[set[int]]:
    """
    Merge the rings that share at least two atoms into ring systems.

    Two rings sharing that many atoms can be traversed as one, while rings meeting
    at a single (spiro) atom cannot: a path through one of them and into the other
    would have to pass that atom twice.
    """
    systems = [set(ring) for ring in ring_atom_sets]

    # merging two systems can bring a third within reach, one sharing a single
    # atom with each of them and two with the two of them together, so a single
    # pass is not enough. Every merge breaks out to start the scan over, and the
    # systems are returned once a whole pass finds no pair left to merge.
    while True:
        for first, second in itertools.combinations(systems, 2):
            if len(first & second) >= 2:
                systems.remove(second)
                first |= second
                break
        else:
            return systems


def _are_ring_systems_complete(
    props: AtomicProperties, ring_systems: list[set[int]]
) -> bool:
    """
    Whether every cycle of the molecule lies inside one of the ring systems.

    The corrections of the ring systems are only independent of one another when
    no cycle spans more than one of them, and that is not guaranteed by merging
    the perceived rings. Cage-like molecules break it in two ways: RDKit's rings
    do not always cover every cycle of such a molecule, and two rings sharing a
    single atom can still be part of one cage, where a path leaving through one
    of the shared atoms comes back through another, unlike a true spiro atom.

    Both show up as cycles unaccounted for, which is what comparing two cycle
    ranks catches: the one of the whole molecule against the ones of the ring
    systems summed. The rank counts how many cycles are independent, the number
    of bonds it carries over a spanning tree, each of them closing one cycle.

    The rank of the molecule (``bonds - atoms + 1``, for a connected one) splits
    over the parts it cannot be traversed between, so the two come out equal when
    the systems hold every cycle. Sharing no bonds, the systems are always
    independent of one another, so their sum can never come out higher. It comes
    out lower when a cycle has bonds no single system holds, whether it was left
    out of the perceived rings or split between two systems: it counts towards the
    rank of the molecule, but towards none of theirs.
    """
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs

    # the ring systems share no bonds - two rings sharing one would share both of
    # its atoms and be merged - so their ranks are summed without double-counting
    rank_sum = 0
    is_inside = np.zeros(props.num_atoms, dtype=bool)
    for ring_system in ring_systems:
        ring_system_atoms = list(ring_system)
        is_inside[ring_system_atoms] = True
        num_bonds = np.count_nonzero(is_inside[begins] & is_inside[ends])
        rank_sum += num_bonds - len(ring_system_atoms) + 1
        is_inside[ring_system_atoms] = False

    return rank_sum == props.num_bonds - props.num_atoms + 1


def _get_bridgeless_components(props: AtomicProperties) -> list[set[int]]:
    """
    Ring systems of a molecule whose perceived rings do not describe it fully.

    Takes away the bonds that no cycle passes through, and keeps what the atoms
    fall apart into. Such bonds are exactly the bridges of the molecular graph,
    so every cycle survives whole inside one of the components, which is what
    the corrections need, see _are_ring_systems_complete(). The components are
    coarser than the ring systems - a spiro atom holds one together instead of
    separating two - which only leaves more for the search through them to do.
    """
    is_bridge = _get_bridges(props)
    begins = props.bond_begin_idxs[~is_bridge]
    ends = props.bond_end_idxs[~is_bridge]

    graph = coo_array(
        (np.ones(len(begins), dtype=bool), (begins, ends)),
        shape=(props.num_atoms, props.num_atoms),
    )
    num_components, labels = connected_components(graph, directed=False)

    components: list[set[int]] = [set() for _ in range(num_components)]
    for atom, label in enumerate(labels.tolist()):
        components[label].add(atom)

    # single atoms and bonds hold no cycle and need no correction
    return [component for component in components if len(component) > 2]


def _get_bridges(props: AtomicProperties) -> np.ndarray:
    """
    Mask of the bonds that no cycle of the molecule passes through.

    Hopcroft-Tarjan bridge finding, iterative to keep molecules with long chains
    from exhausting the recursion limit. Each atom is stamped with the step of
    the search that reached it, and carries the earliest stamp anything below it
    can climb back to; a bond is a bridge when nothing below it reaches past it.
    """
    neighbors = _get_neighbors_with_bonds(props)
    num_atoms = props.num_atoms
    step = 0
    stamps = np.full(num_atoms, -1, dtype=np.intp)  # when each atom was reached
    climbs = np.zeros(num_atoms, dtype=np.intp)  # how far back each one climbs
    is_bridge = np.zeros(props.num_bonds, dtype=bool)

    for root in range(num_atoms):
        if stamps[root] != -1:
            continue

        stamps[root] = climbs[root] = step
        step += 1
        # the atom being explored, the bond it was reached through, and the
        # neighbors of it left to visit
        stack = [(root, -1, iter(neighbors[root]))]

        while stack:
            atom, incoming_bond, remaining = stack[-1]

            for neighbor, bond in remaining:
                if bond == incoming_bond:
                    continue

                if stamps[neighbor] == -1:
                    stamps[neighbor] = climbs[neighbor] = step
                    step += 1
                    stack.append((neighbor, bond, iter(neighbors[neighbor])))
                    break

                # a bond back to an atom already reached, a way up out of here
                climbs[atom] = min(climbs[atom], stamps[neighbor])
            else:
                # every way out of this atom explored, hand up how far it climbs
                stack.pop()
                if stack:
                    previous = stack[-1][0]
                    climbs[previous] = min(climbs[previous], climbs[atom])
                    # nothing below the bond gets past this atom, so no cycle
                    # passes through it
                    if climbs[atom] > stamps[previous]:
                        is_bridge[incoming_bond] = True

    return is_bridge


def _get_neighbors_with_bonds(props: AtomicProperties) -> list[list[tuple[int, int]]]:
    """
    Adjacency lists of all atoms, each neighbor with the bond leading to it.
    """
    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(props.num_atoms)]
    begins = props.bond_begin_idxs.tolist()
    ends = props.bond_end_idxs.tolist()
    for bond, (begin, end) in enumerate(zip(begins, ends, strict=True)):
        neighbors[begin].append((end, bond))
        neighbors[end].append((begin, bond))

    return neighbors


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
    neighbors = _get_ring_system_neighbors(props, atoms)

    if all(len(adjacent) == 2 for adjacent in neighbors):
        # a plain ring: the two ways round it between two atoms make up the whole
        # ring, so the longer one is however much of it the shorter one leaves
        longest = len(atoms) - within
        np.fill_diagonal(longest, 0.0)
        return longest

    # fused or bridged rings have to be searched through, once from every atom.
    # This is pessimistically exponential, as the longest simple path is NP-hard
    # for graphs in general, but affordable because ring systems are small.
    longest = np.zeros_like(within)
    for source in range(len(atoms)):
        distances = [0] * len(atoms)
        _walk_every_path(source, 0, neighbors, {source}, distances)
        longest[source] = distances

    return longest


def _get_ring_system_neighbors(
    props: AtomicProperties, atoms: np.ndarray
) -> list[list[int]]:
    """
    Adjacency lists of one ring system, holding only the bonds between its atoms
    and numbering them by their position in ``atoms``.
    """
    positions = np.full(props.num_atoms, -1, dtype=np.intp)
    positions[atoms] = np.arange(len(atoms))

    begins = positions[props.bond_begin_idxs]
    ends = positions[props.bond_end_idxs]
    is_inside = (begins >= 0) & (ends >= 0)

    neighbors: list[list[int]] = [[] for _ in atoms]
    for begin, end in zip(
        begins[is_inside].tolist(), ends[is_inside].tolist(), strict=True
    ):
        neighbors[begin].append(end)
        neighbors[end].append(begin)

    return neighbors


def _walk_every_path(
    atom: int,
    distance: int,
    neighbors: list[list[int]],
    visited: set[int],
    longest: list[int],
) -> None:
    """
    Longest simple path from one atom of a ring system to each of the others.

    Explores every simple path leaving ``atom`` by DFS with backtracking, keeping
    in ``longest`` the greatest distance any of them reaches each atom at.
    """
    distance += 1

    for neighbor in neighbors[atom]:
        if neighbor in visited:
            continue

        longest[neighbor] = max(longest[neighbor], distance)

        visited.add(neighbor)
        _walk_every_path(neighbor, distance, neighbors, visited, longest)
        visited.remove(neighbor)
