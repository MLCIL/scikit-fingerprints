from functools import cached_property
from itertools import chain

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

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
    Connected subgraphs of a molecule with 1 to :data:`MAX_ORDER` bonds.
    """

    def __init__(self, mol: Mol, props: AtomicProperties):
        self._mol = mol
        # atom indices of the two ends of every bond, shape (n_bonds, 2)
        self._bond_atoms = np.stack(
            [props.bond_begin_idxs, props.bond_end_idxs], axis=1
        )
        self._num_atoms = props.num_atoms

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
        return self._of_order(order)[0][subgraph_type]

    def path_bond_idxs(self, order: int) -> np.ndarray:
        """
        Bond indices of the subgraphs of one order that are paths, i.e. the
        self-avoiding paths spanning that many bonds, shape ``(n_paths, order)``.
        """
        return self._of_order(order)[1]

    @cached_property
    def _cache(self) -> dict[int, tuple[dict[str, list[np.ndarray]], np.ndarray]]:
        return {}

    def _of_order(self, order: int) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
        if order not in self._cache:
            self._cache[order] = _classify_subgraphs(self._mol, order, self._bond_atoms)
        return self._cache[order]


def _classify_subgraphs(
    mol: Mol, order: int, bond_atoms: np.ndarray
) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    """
    Group every connected subgraph with ``order`` bonds by its subgraph type.

    Returns the per-type atom index sets and, separately, the bond indices of the
    subgraphs that are paths.
    """
    subgraphs = Chem.FindAllSubgraphsOfLengthN(mol, order)
    num_subgraphs = len(subgraphs)
    if num_subgraphs == 0:
        empty = {subgraph_type: [] for subgraph_type in SUBGRAPH_TYPES}  # type: ignore
        return empty, np.empty((0, order), dtype=np.intp)

    # bond indices of every subgraph, shape (n_subgraphs, order)
    # inner vectors are copied with [:], it's faster than iteration
    bond_idxs = np.fromiter(
        chain.from_iterable(subgraph[:] for subgraph in subgraphs),
        dtype=np.intp,
        count=num_subgraphs * order,
    ).reshape(num_subgraphs, order)

    # the endpoints of those bonds, sorted within each subgraph so that repeated
    # atoms form runs, shape (n_subgraphs, 2 * order)
    row_width = 2 * order
    endpoints = np.sort(
        bond_atoms[bond_idxs].reshape(num_subgraphs, row_width), axis=1
    ).ravel()

    # one run of equal atom indices is one atom of the subgraph, and the length of
    # the run is that atom's degree within the subgraph
    is_run_start = np.ones(endpoints.shape, dtype=bool)
    is_run_start[1:] = endpoints[1:] != endpoints[:-1]
    is_run_start[::row_width] = True  # a run never spans two subgraphs
    run_starts = np.flatnonzero(is_run_start)
    degrees = np.diff(run_starts, append=endpoints.size)
    subgraph_of_run = run_starts // row_width

    num_nodes = np.bincount(subgraph_of_run, minlength=num_subgraphs)
    has_degree_2 = (
        np.bincount(subgraph_of_run, weights=degrees == 2, minlength=num_subgraphs) > 0
    )
    first_run = np.concatenate([[0], np.cumsum(num_nodes)[:-1]])
    max_degree = np.maximum.reduceat(degrees, first_run)
    nodes = endpoints[run_starts]

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

    node_sets = {
        subgraph_type: _gather_node_sets(
            nodes, first_run, num_nodes, np.flatnonzero(mask)
        )
        for subgraph_type, mask in masks.items()
    }
    return node_sets, bond_idxs[masks["path"]]


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
