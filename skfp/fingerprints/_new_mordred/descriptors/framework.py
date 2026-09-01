from collections import deque

import numpy as np

from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
Molecular framework ratio descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["fMF"]


def calc(
    props_regular: AtomicProperties,
    rings_regular: RingSets,
    num_atoms_hydrogens: int,
) -> np.ndarray:
    """
    Compute the molecular framework ratio, the share of the molecule taken up by
    its framework: the ring atoms together with the linkers joining them.

    The denominator counts hydrogens, so benzene scores 6 / 12 rather than 1.

    Based on Bemis, G. W., & Murcko, M. A. (1996). The properties of known
    drugs. 1. Molecular frameworks. Journal of Medicinal Chemistry, 39(15),
    2887-2893. https://doi.org/10.1021/jm9602928
    """
    if num_atoms_hydrogens == 0:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    ring_atom_sets = rings_regular.simple_ring_atom_sets
    framework_atoms = {atom for ring in ring_atom_sets for atom in ring}
    framework_atoms |= _linker_atoms(props_regular, ring_atom_sets)

    return np.asarray([len(framework_atoms) / num_atoms_hydrogens], dtype=np.float32)


def _linker_atoms(props: AtomicProperties, ring_atom_sets: list[set[int]]) -> set[int]:
    """
    Atoms outside every ring that lie on a shortest path between two rings.

    Each ring is contracted to a single node, so a route between two rings is a
    path in that quotient graph. A breadth-first search from one ring reaches all
    the others at once, so one search per ring suffices, taking one shortest path
    per pair.

    Hydrogens are left out of the graph: with a single bond, a hydrogen can only
    be a dead end, never an interior atom of a path between two rings.
    """
    if len(ring_atom_sets) < 2:
        return set()

    # Mordred labels an atom by the last ring holding it, so an atom shared by
    # fused rings belongs to the highest-numbered one, and a ring whose atoms are
    # all claimed by later rings drops out of the search
    num_atoms = props.num_atoms
    node_of_atom = list(range(num_atoms))
    for ring_idx, ring in enumerate(ring_atom_sets):
        for atom in ring:
            node_of_atom[atom] = num_atoms + ring_idx

    adjacency: list[list[int]] = [[] for _ in range(num_atoms + len(ring_atom_sets))]
    for begin, end in zip(
        props.bond_begin_idxs.tolist(), props.bond_end_idxs.tolist(), strict=True
    ):
        begin_node, end_node = node_of_atom[begin], node_of_atom[end]
        # a bond inside one ring becomes a self-loop once contracted
        if begin_node != end_node:
            adjacency[begin_node].append(end_node)
            adjacency[end_node].append(begin_node)

    ring_nodes = sorted(
        {node_of_atom[atom] for ring in ring_atom_sets for atom in ring}
    )
    linkers: set[int] = set()
    for source in ring_nodes:
        linkers |= _linkers_from_ring(adjacency, source, ring_nodes, num_atoms)

    return linkers


def _linkers_from_ring(
    adjacency: list[list[int]],
    source: int,
    ring_nodes: list[int],
    num_atoms: int,
) -> set[int]:
    """
    Walk out from one ring and collect the non-ring atoms on the shortest path
    to every other ring reachable from it.
    """
    predecessor = {source: source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in predecessor:
                predecessor[neighbor] = node
                queue.append(neighbor)

    linkers: set[int] = set()
    for target in ring_nodes:
        if target == source or target not in predecessor:
            continue
        node = predecessor[target]
        while node != source:
            # ring nodes sit past the atom indices, so anything below is an atom
            if node < num_atoms:
                linkers.add(node)
            node = predecessor[node]

    return linkers
