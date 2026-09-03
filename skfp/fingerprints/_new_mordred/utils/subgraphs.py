from dataclasses import dataclass

import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# NOTE: subgraph order = its number of bonds

# connected subgraphs up to this many bonds, the largest order the
# chi descriptors use
SUBGRAPH_MAX_NUM_BONDS = 7


@dataclass
class Paths:
    """
    A set of self-avoiding paths spanning the same number of bonds.

    Bonds ascend within each row. Atoms are in no particular order.
    A path spans one atom more than it has bonds, so its atoms form
    one rectangular block.
    """

    bond_idxs: np.ndarray  # (n_paths, order)
    atom_idxs: np.ndarray  # (n_paths, order + 1)
    end_atoms: np.ndarray  # (n_paths, 2)


class SubgraphsTopology:
    """
    Topological properties for all connected subgraphs of one order (size).

    Holds, for each subgraph:
    - the atoms it spans
    - whether it closes a cycle
    - how branched it is

    Those are mask arrays, e.g. ``is_cyclic[i]``.
    """

    def __init__(
        self,
        atoms: np.ndarray,
        atom_offsets: np.ndarray,
        atom_counts: np.ndarray,
        is_cyclic: np.ndarray,
        max_degree: np.ndarray,
        has_degree_2: np.ndarray,
    ):
        # the atoms of every subgraph, concatenated; a subgraph's own atoms are
        # the run of atom_counts entries starting at its atom_offsets entry
        self._atoms = atoms
        self._atom_offsets = atom_offsets
        self._atom_counts = atom_counts

        # the three per-subgraph arrays below all have shape (n_subgraphs,)
        self.is_cyclic = is_cyclic
        self.max_degree = max_degree
        self.has_degree_2 = has_degree_2

    @property
    def is_path(self) -> np.ndarray:
        """
        Path = acyclic and unbranched.
        """
        return ~self.is_cyclic & (self.max_degree <= 2)

    def atom_idxs(self, subgraph_idx: int) -> np.ndarray:
        """
        Get the atoms one subgraph spans, the run of ``atom_counts`` entries that
        it owns.
        """
        start = self._atom_offsets[subgraph_idx]
        return self._atoms[start : start + self._atom_counts[subgraph_idx]]

    def atom_products(self, values: np.ndarray) -> np.ndarray:
        """
        Product of each row of ``values``, an array of shape (n_rows, n_atoms),
        over the atoms of every subgraph, shape (n_rows, n_subgraphs).

        Subgraphs of one order need not span equally many atoms, so their atoms
        are kept flat and multiplied one run at a time.
        """
        return np.multiply.reduceat(values[:, self._atoms], self._atom_offsets, axis=1)


class Subgraphs:
    """
    Connected subgraphs of a molecule, as arrays of bond indices.

    The subgraphs come from the ESU recursion in ``_enumerate_subgraphs()``.
    This algorithm is quite a bit faster than using RDKit.
    """

    def __init__(self, props: AtomicProperties):
        # atom indices of the two ends of every bond, shape (n_bonds, 2)
        self.bond_atoms = np.stack([props.bond_begin_idxs, props.bond_end_idxs], axis=1)
        self._num_atoms = props.num_atoms

        # which bonds each atom takes part in
        self.bonds_of_atom = self._build_atom_adjacency()

        # which bonds share an atom
        bonds_share_atom = self._line_graph_neighbors(
            self.bond_atoms, self.bonds_of_atom
        )

        # bond indexes of every subgraph, one entry per order
        # every order is enumerated in one pass, since each is grown from previous one
        subgraphs = self._enumerate_subgraphs(
            bonds_share_atom, props.num_bonds, SUBGRAPH_MAX_NUM_BONDS
        )

        # topology and paths of every subgraph, one tuple for each
        # both hold one entry per order, indexed by order - 1, since order 0 has
        # no bonds and is never enumerated
        analyzed = [
            self._topology_and_paths(bond_idxs, self.bond_atoms, order)
            for order, bond_idxs in enumerate(subgraphs, start=1)
        ]
        self._topologies = tuple(topology for topology, _ in analyzed)
        self._paths = tuple(paths for _, paths in analyzed)

    def topology(self, order: int) -> SubgraphsTopology:
        """
        Graph topology info for all subgraphs with a given order (number of bonds).
        """
        return self._topologies[order - 1]

    def paths(self, order: int) -> Paths:
        """
        Select all paths of a given order (length, number of bonds), up to
        ``SUBGRAPH_MAX_NUM_BONDS``.
        """
        return self._paths[order - 1]

    def _build_atom_adjacency(self) -> np.ndarray:
        """
        Bonds incident to each atom, array of shape (n_atoms, max_atom_degree).

        Unused cells for lower-degree atoms get value -1.
        """
        degrees = np.bincount(self.bond_atoms.ravel(), minlength=self._num_atoms)
        max_degree = int(degrees.max()) if degrees.size else 0
        bonds_of_atom = np.full((self._num_atoms, max_degree), -1, dtype=np.intp)

        # filling row by row keeps each atom's bonds in ascending order
        next_slot = np.zeros(self._num_atoms, dtype=np.intp)
        for bond, (begin, end) in enumerate(self.bond_atoms):
            for atom in (begin, end):
                bonds_of_atom[atom, next_slot[atom]] = bond
                next_slot[atom] += 1

        return bonds_of_atom

    @staticmethod
    def _line_graph_neighbors(
        bond_atoms: np.ndarray, bonds_of_atom: np.ndarray
    ) -> list[set[int]]:
        """
        For every bond, the bonds that share an atom with it.

        Calculates line graph (https://en.wikipedia.org/wiki/Line_graph),
        whose nodes are molecule bonds. Connected subgraph in a molecule
        is a set of bonds connected in a line graph. Thus, this representation
        allows quick enumeration.
        """
        neighbors = []
        for bond, (begin, end) in enumerate(bond_atoms):
            # a bond touches every bond sitting at either of its two atoms
            shared = set(bonds_of_atom[begin].tolist())
            shared.update(bonds_of_atom[end].tolist())
            shared.discard(bond)
            shared.discard(-1)  # padding of the lower-degree atoms
            neighbors.append(shared)
        return neighbors

    @staticmethod
    def _enumerate_subgraphs(
        neighbors: list[set[int]], num_bonds: int, max_order: int
    ) -> tuple[np.ndarray, ...]:
        """
        Enumerate connected subgraphs up to ``max_order`` bonds.

        Returns one entry per order, indexed by order - 1: the bond indices of each
        subgraph of that order, shape (n_subgraphs, order)

        Subgraph enumeration uses ESU algorithm Wernicke, "Efficient detection of
        network motifs"), which fixes for each subgraph a single order in which
        it may be built:

        - the lowest-numbered bond of a subgraph is its root
        - only bonds above the root are added, so a subgraph can only grow from
          its own lowest bond
        - when a bond is added, the bonds already considered stay, and the only
          bonds joining them are those that the added bond reaches on its own - a bond
          already touching the subgraph was considered before and was passed over

        The last rule is what stops one set of bonds from being reached by adding its
        bonds in different orders: passing over a bond drops it from that branch for
        good.
        """
        by_order: list[list[tuple[int, ...]]] = [[] for _ in range(max_order)]

        # ``touching`` is the bonds the subgraph can already reach; whatever the next
        # bond adds has to lie beyond these, since these were on offer before and were
        # declined. It is threaded through rather than recomputed, since a subgraph
        # reaches exactly what its parent reached plus what the added bond reaches
        def grow(
            subgraph: list[int], offered: set[int], touching: set[int], root: int
        ) -> None:
            by_order[len(subgraph) - 1].append(tuple(sorted(subgraph)))
            if len(subgraph) == max_order:
                return

            # popping from a copy makes a declined bond stay declined for this branch,
            # while each child gets its own offer set to work through
            offered = offered.copy()
            while offered:
                added = offered.pop()
                newly_reachable = {
                    bond
                    for bond in neighbors[added]
                    if bond > root and bond not in subgraph and bond not in touching
                }
                # adding a bond extends the reach of the subgraph by its own
                grow(
                    [*subgraph, added],
                    offered | newly_reachable,
                    touching | neighbors[added],
                    root,
                )

        for root_bond in range(num_bonds):
            above_root = {bond for bond in neighbors[root_bond] if bond > root_bond}
            grow([root_bond], above_root, neighbors[root_bond], root_bond)

        result = tuple(
            np.array(rows, dtype=np.intp)
            if rows
            else np.empty((0, order), dtype=np.intp)
            for order, rows in enumerate(by_order, start=1)
        )

        return result

    @classmethod
    def _topology_and_paths(
        cls, bond_idxs: np.ndarray, bond_atoms: np.ndarray, order: int
    ) -> tuple[SubgraphsTopology, Paths]:
        """
        Describe the shape of every connected subgraph with ``order`` bonds and pick
        the paths out of them. Both can be computed in a single pass.
        """
        num_subgraphs = len(bond_idxs)
        if num_subgraphs == 0:
            empty = np.empty(0, dtype=np.intp)
            no_flags = np.empty(0, dtype=bool)
            topology = SubgraphsTopology(empty, empty, empty, no_flags, empty, no_flags)
            return topology, cls._no_paths(order)

        # the two atoms of each subgraph bond, one row per subgraph, sorted within the
        # row so that repeats of an atom land next to each other, then flattened
        row_width = 2 * order
        endpoints = np.sort(
            bond_atoms[bond_idxs].reshape(num_subgraphs, row_width), axis=1
        ).ravel()

        # one run of equal atoms is one atom of the subgraph, its length that atom's
        # degree within the subgraph
        is_run_start = np.ones(endpoints.shape, dtype=bool)
        is_run_start[1:] = endpoints[1:] != endpoints[:-1]
        is_run_start[::row_width] = True  # rows are back to back, a run spans only one
        run_start_idxs = np.flatnonzero(is_run_start)

        # one entry per run, i.e. per (subgraph, atom), the runs of a subgraph adjacent
        degrees = np.diff(run_start_idxs, append=endpoints.size)
        subgraph_of_run = run_start_idxs // row_width
        atoms = endpoints[run_start_idxs]

        # ragged view: subgraph i owns atom_counts[i] entries from atom_offsets[i]
        atom_counts = np.bincount(subgraph_of_run, minlength=num_subgraphs)
        atom_offsets = cls._run_starts(atom_counts)

        # a tree has one bond fewer than atoms, so no fewer means a cycle
        is_cyclic = order >= atom_counts
        has_degree_2 = (
            np.bincount(subgraph_of_run, weights=degrees == 2, minlength=num_subgraphs)
            > 0
        )

        topology = SubgraphsTopology(
            atoms,
            atom_offsets,
            atom_counts,
            is_cyclic=is_cyclic,
            max_degree=np.maximum.reduceat(degrees, atom_offsets),
            has_degree_2=has_degree_2,
        )
        paths = cls._extract_paths(
            bond_idxs, topology, atoms, atom_offsets, degrees, order
        )
        return topology, paths

    @classmethod
    def _extract_paths(
        cls,
        bond_idxs: np.ndarray,
        topology: SubgraphsTopology,
        atoms: np.ndarray,
        atom_offsets: np.ndarray,
        degrees: np.ndarray,
        order: int,
    ) -> Paths:
        """
        Read the paths from the subgraphs.

        A path spans exactly ``order + 1`` atoms, so its atoms form one rectangular
        block of shape ``(n_paths, order + 1)``. It ends at the two atoms that only
        one of its bonds touches.
        """
        path_rows = np.flatnonzero(topology.is_path)
        if not len(path_rows):
            return cls._no_paths(order)

        within_path = np.arange(order + 1)
        path_starts = atom_offsets[path_rows][:, np.newaxis]
        path_atoms = atoms[path_starts + within_path]
        path_degrees = degrees[path_starts + within_path]
        end_atoms = path_atoms[path_degrees == 1].reshape(len(path_rows), 2)
        return Paths(
            bond_idxs=bond_idxs[path_rows],
            atom_idxs=path_atoms,
            end_atoms=end_atoms,
        )

    @staticmethod
    def _no_paths(order: int) -> Paths:
        """
        Empty paths, to handle case with no paths found.
        """
        return Paths(
            bond_idxs=np.empty((0, order), dtype=np.intp),
            atom_idxs=np.empty((0, order + 1), dtype=np.intp),
            end_atoms=np.empty((0, 2), dtype=np.intp),
        )

    @staticmethod
    def _run_starts(counts: np.ndarray) -> np.ndarray:
        """
        Since we use flattened 2D arrays for speed, we need to get where each row
        actually starts. Calculcated as a prefix sum of lengths.

        E.g. for counts = [3, 0, 2] this is [0, 3, 3] - rows 1 and 2 share a start
        offset because row 1 is empty.
        """
        # cumsum gives each row's end, i.e. the inclusive prefix sum
        # subtracting the row's own length turns that into its start
        row_ends = np.cumsum(counts)
        return row_ends - counts
