from functools import cached_property

import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.ragged import ragged_indices, run_starts

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# connected subgraphs are enumerated up to this many bonds, the largest order the
# chi descriptors use
MAX_ORDER = 7

# subgraph classes used by the chi descriptors: a chain contains a cycle, a path
# is an unbranched acyclic subgraph, and clusters are branched acyclic subgraphs,
# further split by whether any atom has exactly two subgraph bonds
SUBGRAPH_TYPES = ("chain", "path", "path_cluster", "cluster")


class Subgraphs:
    """
    Connected subgraphs of a molecule, as arrays of bond indices.

    Subgraphs of each order are grown from those of the previous order by adding
    one adjacent bond, entirely with array operations. RDKit's own enumeration
    (``FindAllSubgraphsOfLengthN``) is not used, because reading its per-subgraph
    C++ vectors into NumPy costs far more than the enumeration itself.
    """

    def __init__(self, props: AtomicProperties):
        # atom indices of the two ends of every bond, shape (n_bonds, 2)
        self._bond_atoms = np.stack(
            [props.bond_begin_idxs, props.bond_end_idxs], axis=1
        )
        self._num_atoms = props.num_atoms
        self._num_bonds = props.num_bonds

    def node_sets(self, order: int, subgraph_type: str) -> list[np.ndarray]:
        """
        Atom indices of the subgraphs of one order and type.

        Returned as arrays of shape ``(n_subgraphs, n_nodes)``, one per
        distinct node count, for bulk computations.

        Acyclic subgraphs of a given order always span ``order + 1`` atoms and so
        need only one array, while chains can span fewer.
        """
        if order == 0:
            # the order 0 subgraphs are the individual atoms, of every type
            return [np.arange(self._num_atoms)[:, None]]
        return self._classified(order)[subgraph_type]

    def path_bond_idxs(self, order: int) -> np.ndarray:
        """
        Bond indices of the subgraphs of one order that are paths, i.e. the
        self-avoiding paths spanning that many bonds, shape ``(n_paths, order)``.

        Unlike the other subgraph types, paths are available for any order, since
        they are grown on their own and their number stays modest.
        """
        return self._paths(order)[0]

    @cached_property
    def _subgraph_cache(self) -> dict[int, np.ndarray]:
        return {}

    @cached_property
    def _classified_cache(self) -> dict[int, dict[str, list[np.ndarray]]]:
        return {}

    @cached_property
    def _path_cache(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return {}

    @cached_property
    def _bond_adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bonds sharing an atom with each bond, as (starts, counts, flat) arrays,
        i.e. the neighbors of bond ``b`` are
        ``flat[starts[b] : starts[b] + counts[b]]``.
        """
        starts, counts, flat = self._atom_adjacency
        # every pair of distinct bonds incident to a common atom is a neighbor pair
        owner, within = ragged_indices(counts * counts)
        num_bonds_of_atom = counts[owner]
        first = flat[starts[owner] + within // num_bonds_of_atom]
        second = flat[starts[owner] + within % num_bonds_of_atom]

        pairs = np.stack([first, second], axis=1)[first != second]
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
        neighbor_counts = np.bincount(pairs[:, 0], minlength=self._num_bonds)
        return run_starts(neighbor_counts), neighbor_counts, pairs[:, 1]

    @cached_property
    def _atom_adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bonds incident to each atom, as (starts, counts, flat) arrays.
        """
        # the (bond, bond end) slots, ordered by the atom they belong to
        slot_atoms = self._bond_atoms.ravel()
        slots = np.argsort(slot_atoms, kind="stable")
        counts = np.bincount(slot_atoms, minlength=self._num_atoms)
        return run_starts(counts), counts, slots // 2

    def _subgraph_bond_idxs(self, order: int) -> np.ndarray:
        """
        Bond indices of every connected subgraph with ``order`` bonds, ascending
        within each row, shape ``(n_subgraphs, order)``.
        """
        rows = self._subgraph_cache.get(order)
        if rows is not None:
            return rows

        if order == 1:
            rows = np.arange(self._num_bonds, dtype=np.intp)[:, None]
        else:
            # every connected subgraph with `order` bonds is a connected subgraph
            # with one bond fewer plus a bond adjacent to it, so growing all of the
            # smaller ones by every adjacent bond yields all of them (with repeats)
            smaller = self._subgraph_bond_idxs(order - 1)
            candidates, owner = self._neighbors_of(smaller, self._bond_adjacency)
            smaller_rows = smaller[owner // (order - 1)]

            # a bond adjacent to the subgraph may already belong to it; dropping
            # those candidates first keeps the arrays below as small as possible
            keep = ~(smaller_rows == candidates[:, None]).any(axis=1)
            rows = np.concatenate(
                [smaller_rows[keep], candidates[keep][:, None]], axis=1
            )
            rows.sort(axis=1)
            rows = rows[_unique_row_idxs(rows, self._num_bonds)]

        self._subgraph_cache[order] = rows
        return rows

    @staticmethod
    def _neighbors_of(
        rows: np.ndarray, adjacency: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        For every element of every row, all of its neighbors in ``adjacency``.

        Returns the neighbors and, for each of them, the index of the element it
        came from in the flattened ``rows``.
        """
        starts, counts, flat = adjacency
        elements = rows.ravel()
        neighbor_counts = counts[elements]
        owner, within = ragged_indices(neighbor_counts)
        neighbors = flat[np.repeat(starts[elements], neighbor_counts) + within]
        return neighbors, owner

    def _classified(self, order: int) -> dict[str, list[np.ndarray]]:
        node_sets = self._classified_cache.get(order)
        if node_sets is None:
            node_sets = _classify_subgraphs(
                self._subgraph_bond_idxs(order), self._bond_atoms, order
            )
            self._classified_cache[order] = node_sets
        return node_sets

    def _paths(self, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Self-avoiding paths with ``order`` bonds: their bond indices (ascending
        within each row), the atoms they span (in no particular order) and their
        two endpoint atoms.
        """
        paths = self._path_cache.get(order)
        if paths is not None:
            return paths

        if order == 1:
            bond_idxs = np.arange(self._num_bonds, dtype=np.intp)[:, None]
            # a single bond spans its two atoms, which are also its endpoints
            paths = bond_idxs, self._bond_atoms, self._bond_atoms
        else:
            paths = self._grow_paths(*self._paths(order - 1))

        self._path_cache[order] = paths
        return paths

    def _grow_paths(
        self, bond_idxs: np.ndarray, atom_idxs: np.ndarray, end_atoms: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extend every path by one bond at either of its two ends.
        """
        # bonds incident to an endpoint atom, and the atom they would add
        candidates, owner = self._neighbors_of(end_atoms, self._atom_adjacency)
        path_of, extended_end = owner // 2, owner % 2
        # the atom on the far side of the candidate bond
        new_atoms = self._bond_atoms[candidates].sum(axis=1) - end_atoms.ravel()[owner]

        # a self-avoiding path may not revisit an atom, which also rules out the
        # bonds it already contains
        keep = ~np.any(atom_idxs[path_of] == new_atoms[:, None], axis=1)
        candidates, path_of = candidates[keep], path_of[keep]
        extended_end, new_atoms = extended_end[keep], new_atoms[keep]

        new_bond_idxs = np.concatenate(
            [bond_idxs[path_of], candidates[:, None]], axis=1
        )
        new_bond_idxs.sort(axis=1)  # ascending rows, as the deduplication needs
        new_atom_idxs = np.concatenate([atom_idxs[path_of], new_atoms[:, None]], axis=1)
        # the new path ends at the atom just added and at the end left untouched
        new_end_atoms = np.stack(
            [new_atoms, end_atoms[path_of, 1 - extended_end]], axis=1
        )

        # every path was grown twice, once from each of its two ends
        unique = _unique_row_idxs(new_bond_idxs, self._num_bonds)
        return new_bond_idxs[unique], new_atom_idxs[unique], new_end_atoms[unique]


def _unique_row_idxs(rows: np.ndarray, num_values: int) -> np.ndarray:
    """
    Find the indices of the distinct rows of an integer array with ascending rows.
    """
    num_rows, num_cols = rows.shape
    if num_rows < 2:
        return np.arange(num_rows)

    # rows of small integers pack into a single integer key, which deduplicates
    # much faster than comparing the rows themselves
    if num_values**num_cols <= np.iinfo(np.int64).max:
        powers = np.asarray(
            [num_values**exponent for exponent in range(num_cols)], dtype=np.int64
        )
        return np.unique(rows @ powers, return_index=True)[1]

    return np.unique(rows, axis=0, return_index=True)[1]


def _classify_subgraphs(
    bond_idxs: np.ndarray, bond_atoms: np.ndarray, order: int
) -> dict[str, list[np.ndarray]]:
    """
    Group every connected subgraph with ``order`` bonds by its subgraph type,
    returning the atom index sets of each type.
    """
    num_subgraphs = len(bond_idxs)
    if num_subgraphs == 0:
        return {subgraph_type: [] for subgraph_type in SUBGRAPH_TYPES}

    # the endpoints of the subgraph bonds, sorted within each subgraph so that
    # repeated atoms form runs, shape (n_subgraphs, 2 * order)
    row_width = 2 * order
    endpoints = np.sort(
        bond_atoms[bond_idxs].reshape(num_subgraphs, row_width), axis=1
    ).ravel()

    # one run of equal atom indices is one atom of the subgraph, and the length of
    # the run is that atom's degree within the subgraph
    is_run_start = np.ones(endpoints.shape, dtype=bool)
    is_run_start[1:] = endpoints[1:] != endpoints[:-1]
    is_run_start[::row_width] = True  # a run never spans two subgraphs
    run_start_idxs = np.flatnonzero(is_run_start)
    degrees = np.diff(run_start_idxs, append=endpoints.size)
    subgraph_of_run = run_start_idxs // row_width

    num_nodes = np.bincount(subgraph_of_run, minlength=num_subgraphs)
    has_degree_2 = (
        np.bincount(subgraph_of_run, weights=degrees == 2, minlength=num_subgraphs) > 0
    )
    first_run = run_starts(num_nodes)
    max_degree = np.maximum.reduceat(degrees, first_run)
    nodes = endpoints[run_start_idxs]

    # a connected subgraph contains a cycle iff it has at least as many edges as
    # nodes; the acyclic ones are told apart by their degree sequence
    is_cyclic = order >= num_nodes
    is_branched = ~is_cyclic & (max_degree > 2)
    masks = {
        "chain": is_cyclic,
        "path": ~is_cyclic & (max_degree <= 2),
        "path_cluster": is_branched & has_degree_2,
        "cluster": is_branched & ~has_degree_2,
    }

    return {
        subgraph_type: _gather_node_sets(
            nodes, first_run, num_nodes, np.flatnonzero(mask)
        )
        for subgraph_type, mask in masks.items()
    }


def _gather_node_sets(
    nodes: np.ndarray,
    first_run: np.ndarray,
    num_nodes: np.ndarray,
    selected: np.ndarray,
) -> list[np.ndarray]:
    """
    Atom indices of the selected subgraphs, as rectangular arrays grouped by their
    node count.
    """
    node_sets = []
    for size in np.unique(num_nodes[selected]):
        rows = selected[num_nodes[selected] == size]
        node_sets.append(nodes[first_run[rows][:, None] + np.arange(size)])
    return node_sets
