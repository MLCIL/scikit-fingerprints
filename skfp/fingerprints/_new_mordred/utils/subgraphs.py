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

_MAX_INT64 = int(np.iinfo(np.int64).max)

# subgraph classes used by the chi descriptors: a chain contains a cycle, a path
# is an unbranched acyclic subgraph, and clusters are branched acyclic subgraphs,
# further split by whether any atom has exactly two subgraph bonds
SUBGRAPH_TYPES = ("chain", "path", "path_cluster", "cluster")

# a set of paths: their bond indices, the atoms they span and their two end atoms
Paths = tuple[np.ndarray, np.ndarray, np.ndarray]


class ClassifiedSubgraphs:
    """
    The subgraphs of one order, sorted into their types.

    The atom index sets of a type are gathered when they are first asked for, since
    the chi descriptors want different types at different orders: only paths at the
    first two orders, and everything but the branched types at order seven, which is
    where most of the subgraphs are.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        first_run: np.ndarray,
        num_nodes: np.ndarray,
        masks: dict[str, np.ndarray],
        paths: Paths,
    ):
        self._nodes = nodes
        self._first_run = first_run
        self._num_nodes = num_nodes
        self._masks = masks
        self.paths = paths
        self._node_sets: dict[str, list[np.ndarray]] = {}

    def node_sets(self, subgraph_type: str) -> list[np.ndarray]:
        """
        Atom indices of the subgraphs of one type, grouped by their atom count.
        """
        gathered = self._node_sets.get(subgraph_type)
        if gathered is None:
            gathered = _gather_node_sets(
                self._nodes,
                self._first_run,
                self._num_nodes,
                np.flatnonzero(self._masks[subgraph_type]),
            )
            self._node_sets[subgraph_type] = gathered
        return gathered


class Subgraphs:
    """
    Connected subgraphs of a molecule, as arrays of bond indices.

    Subgraphs of each order are grown from those of the previous order by adding
    one adjacent bond, entirely with array operations. Manual enumeration is faster
    than calling RDKit FindAllSubgraphsOfLengthN() when NumPy conversion costs
    are considered.
    """

    def __init__(self, props: AtomicProperties):
        # atom indices of the two ends of every bond, shape (n_bonds, 2)
        self._bond_atoms = np.stack(
            [props.bond_begin_idxs, props.bond_end_idxs], axis=1
        )
        self._num_atoms = props.num_atoms
        self._num_bonds = props.num_bonds

        # which bonds an atom takes part in, and which bonds share an atom
        self._atom_adjacency = self._build_atom_adjacency()
        self._line_graph_adjacency = _line_graph_adjacency(props)

        # subgraphs, their classification and their paths, filled in per order as
        # the descriptors ask for them
        self._subgraph_cache: dict[int, np.ndarray] = {}
        self._classified_cache: dict[int, ClassifiedSubgraphs] = {}
        self._path_cache: dict[int, Paths] = {}

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
        return self._classified(order).node_sets(subgraph_type)

    def path_bond_idxs(self, order: int) -> np.ndarray:
        """
        Bond indices of the subgraphs of one order that are paths, i.e. the
        self-avoiding paths spanning that many bonds, shape ``(n_paths, order)``.

        Unlike the other subgraph types, paths are available for any order, since
        they are grown on their own and their number stays modest.
        """
        return self._paths(order)[0]

    def _build_atom_adjacency(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if not self._subgraph_cache:
            self._subgraph_cache = _enumerate_subgraphs(
                self._line_graph_adjacency, self._num_bonds, MAX_ORDER
            )
        return self._subgraph_cache[order]

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

    def _classified(self, order: int) -> ClassifiedSubgraphs:
        classified = self._classified_cache.get(order)
        if classified is None:
            classified = _classify_subgraphs(
                self._subgraph_bond_idxs(order), self._bond_atoms, order
            )
            self._classified_cache[order] = classified
        return classified

    def _paths(self, order: int) -> Paths:
        """
        Self-avoiding paths with ``order`` bonds: their bond indices (ascending
        within each row), the atoms they span (in no particular order) and their
        two endpoint atoms.

        Up to :data:`MAX_ORDER` the paths are the ones the classification has
        already picked out of the subgraphs; the longer ones are grown from those,
        since enumerating every subgraph of those orders would cost far more.
        """
        paths = self._path_cache.get(order)
        if paths is not None:
            return paths

        if order <= MAX_ORDER:
            paths = self._classified(order).paths
        else:
            paths = self._grow_paths(*self._paths(order - 1))

        self._path_cache[order] = paths
        return paths

    def _grow_paths(
        self, bond_idxs: np.ndarray, atom_idxs: np.ndarray, end_atoms: np.ndarray
    ) -> Paths:
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
    if num_values**num_cols > _MAX_INT64:
        return np.unique(rows, axis=0, return_index=True)[1]

    powers = np.asarray(
        [num_values**exponent for exponent in range(num_cols)], dtype=np.int64
    )
    return _first_of_each_value(rows @ powers)


def _first_of_each_value(keys: np.ndarray) -> np.ndarray:
    """
    Find the index of the first occurrence of every distinct value.
    """
    order = np.argsort(keys, kind="stable")
    is_first = np.ones(len(keys), dtype=bool)
    is_first[1:] = np.diff(keys[order]) != 0
    return order[is_first]


def _classify_subgraphs(
    bond_idxs: np.ndarray, bond_atoms: np.ndarray, order: int
) -> ClassifiedSubgraphs:
    """
    Group every connected subgraph with ``order`` bonds by its subgraph type.

    The paths among them are picked out here as well, since the path counts need
    them as bonds rather than as atoms.
    """
    num_subgraphs = len(bond_idxs)
    if num_subgraphs == 0:
        empty = np.empty(0, dtype=np.intp)
        no_subgraphs = dict.fromkeys(SUBGRAPH_TYPES, empty)
        return ClassifiedSubgraphs(empty, empty, empty, no_subgraphs, _no_paths(order))

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

    # a path spans one atom more than it has bonds, and ends at the two atoms that
    # only one of its bonds touches
    path_rows = np.flatnonzero(masks["path"])
    if len(path_rows):
        path_atoms = nodes[first_run[path_rows][:, np.newaxis] + np.arange(order + 1)]
        path_degrees = degrees[
            first_run[path_rows][:, np.newaxis] + np.arange(order + 1)
        ]
        paths = (
            bond_idxs[path_rows],
            path_atoms,
            path_atoms[path_degrees == 1].reshape(len(path_rows), 2),
        )
    else:
        paths = _no_paths(order)

    classified = ClassifiedSubgraphs(nodes, first_run, num_nodes, masks, paths)
    # the atoms of the paths are exactly their node sets, already gathered above
    classified._node_sets["path"] = [paths[1]] if len(path_rows) else []
    return classified


def _no_paths(order: int) -> Paths:
    """Empty path arrays of the shapes an order-``order`` path set would have."""
    return (
        np.empty((0, order), dtype=np.intp),
        np.empty((0, order + 1), dtype=np.intp),
        np.empty((0, 2), dtype=np.intp),
    )


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
    for size in np.flatnonzero(np.bincount(num_nodes[selected])):
        rows = selected[num_nodes[selected] == size]
        node_sets.append(nodes[first_run[rows][:, None] + np.arange(size)])
    return node_sets


def _enumerate_subgraphs(
    adjacency: np.ndarray, num_bonds: int, max_order: int
) -> dict[int, np.ndarray]:
    """
    Bond indices of every connected subgraph with up to ``max_order`` bonds, as one
    array of shape ``(n_subgraphs, order)`` per order.

    Every subgraph is produced exactly once, following ESU (Wernicke, "Efficient
    detection of network motifs", 2006). The subgraphs wanted here are sets of
    bonds, so the enumeration runs on the line graph, whose nodes are the bonds of
    the molecule and whose edges join the bonds that share an atom: a set of bonds
    is a connected subgraph of the molecule exactly when it is connected there.

    ESU makes each subgraph appear once by fixing, for every subgraph, one order in
    which it may be built:

    - the lowest-numbered bond of a subgraph is its root, and only bonds above the
      root are ever added, so a subgraph can only be built from its own lowest bond;
    - when a bond is added, the bonds already on offer stay on offer, minus the ones
      already used, and the newly reachable bonds are only those the added bond
      brings in on its own -- bonds already touching the subgraph were on offer
      before and were passed over deliberately.

    The second rule is what stops the same set from being reached by adding its
    bonds in different orders: passing over a bond drops it from that branch for
    good.
    """
    # a subgraph of one bond, rooted at that bond, extensible by its later neighbors
    bonds = np.arange(num_bonds, dtype=np.intp)[:, np.newaxis]
    roots = bonds[:, 0]
    extensions, extension_counts = _later_neighbors(adjacency, roots, roots)

    subgraphs = {1: bonds}
    for order in range(2, max_order + 1):
        bonds, roots, extensions, extension_counts = _extend(
            adjacency, bonds, roots, extensions, extension_counts
        )
        subgraphs[order] = np.sort(bonds, axis=1)

    return subgraphs


def _line_graph_adjacency(props: AtomicProperties) -> np.ndarray:
    """
    Which bonds share an atom, as a dense boolean matrix.

    Dense is the right shape here: molecules have few enough bonds that the matrix
    stays small, and the enumeration below asks about arbitrary bond pairs, which a
    matrix answers in one indexing operation.
    """
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs
    shares_atom = (
        (begins[:, np.newaxis] == begins)
        | (begins[:, np.newaxis] == ends)
        | (ends[:, np.newaxis] == begins)
        | (ends[:, np.newaxis] == ends)
    )
    np.fill_diagonal(shares_atom, False)
    return shares_atom


def _later_neighbors(
    adjacency: np.ndarray, bonds: np.ndarray, roots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every bond, its neighbors numbered above the root of its subgraph, as a flat
    array and the count belonging to each subgraph.
    """
    is_later_neighbor = adjacency[bonds] & (
        np.arange(adjacency.shape[1]) > roots[:, np.newaxis]
    )
    owners, neighbors = np.nonzero(is_later_neighbor)
    counts = np.bincount(owners, minlength=len(bonds))
    return neighbors, counts


def _extend(
    adjacency: np.ndarray,
    bonds: np.ndarray,
    roots: np.ndarray,
    extensions: np.ndarray,
    extension_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Grow every subgraph by each of the bonds it may still be extended by.

    A subgraph with ``m`` bonds available spawns ``m`` children, the one built with
    the ``i``-th available bond inheriting only the ``m - i - 1`` bonds after it.
    """
    parents, position = ragged_indices(extension_counts)
    first_extension = run_starts(extension_counts)
    added = extensions[first_extension[parents] + position]

    child_bonds = np.concatenate([bonds[parents], added[:, np.newaxis]], axis=1)
    child_roots = roots[parents]

    # what the child inherits: the bonds its parent had not offered yet
    inherited_counts = extension_counts[parents] - position - 1
    inherited_owners, inherited_position = ragged_indices(inherited_counts)
    inherited = extensions[
        first_extension[parents[inherited_owners]]
        + position[inherited_owners]
        + 1
        + inherited_position
    ]

    # what the added bond brings in: its later neighbors that the subgraph could not
    # already reach, since those were on offer before and were passed over
    candidates, candidate_counts = _later_neighbors(adjacency, added, child_roots)
    candidate_owners, _ = ragged_indices(candidate_counts)
    is_new = ~adjacency[candidates, bonds[parents[candidate_owners]].T].any(axis=0) & ~(
        candidates[:, np.newaxis] == bonds[parents[candidate_owners]]
    ).any(axis=1)

    child_extensions, child_counts = _merge_extensions(
        len(child_bonds),
        inherited_owners,
        inherited,
        candidate_owners[is_new],
        candidates[is_new],
    )
    return child_bonds, child_roots, child_extensions, child_counts


def _merge_extensions(
    num_children: int,
    inherited_owners: np.ndarray,
    inherited: np.ndarray,
    new_owners: np.ndarray,
    new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate the inherited and the newly reachable bonds of every child into one
    flat array, ordered by child.
    """
    owners = np.concatenate([inherited_owners, new_owners])
    values = np.concatenate([inherited, new])
    by_owner = np.argsort(owners, kind="stable")
    counts = np.bincount(owners, minlength=num_children)
    return values[by_owner], counts
