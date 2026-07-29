from functools import cached_property

import numpy as np
from rdkit.Chem import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix, Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _cache_prefix(use_bond_orders: bool, use_atom_weights: bool = False) -> str:
    """
    Name under which RDKit memoizes a graph matrix on the molecule.

    Every weighted flavor needs a name of its own, because asking for one under the
    default (empty) name hands back an unweighted matrix computed earlier. The
    unweighted matrices keep that default name, which is where RDKit itself, and
    therefore most of the memoized matrices, put them.
    """
    if not use_bond_orders and not use_atom_weights:
        return ""
    return f"skfp_bond_orders{use_bond_orders:d}_atom_weights{use_atom_weights:d}"


class DistanceMatrix:
    hermitian = True

    def __init__(
        self, mol: Mol, use_bond_orders: bool = False, use_atom_weights: bool = False
    ):
        self.matrix: np.ndarray
        self.matrix = GetDistanceMatrix(
            mol,
            useBO=use_bond_orders,
            useAtomWts=use_atom_weights,
            prefix=_cache_prefix(use_bond_orders, use_atom_weights),
        )

    @classmethod
    def with_hydrogens_added(
        cls, distances: "DistanceMatrix", props_hydrogens: AtomicProperties
    ) -> "DistanceMatrix":
        """
        Distance matrix of ``AddHs(mol)``, built from the distance matrix of ``mol``.

        The added hydrogens are terminal atoms, so the shortest path to a hydrogen
        is the shortest path to the atom it hangs off plus that one bond, and the
        one between two hydrogens has such a bond at both ends. Letting RDKit rerun
        all-pairs shortest paths on the twice-as-large molecule costs several times
        more than filling the matrix in this way.
        """
        heavy_distances = distances.matrix
        num_heavy = len(heavy_distances)
        num_atoms = props_hydrogens.num_atoms

        # the added hydrogens come after every other atom and have one bond each,
        # so the lower end of such a bond is the atom the hydrogen hangs off
        begins = props_hydrogens.bond_begin_idxs
        ends = props_hydrogens.bond_end_idxs
        hydrogen_bonds = np.flatnonzero((begins >= num_heavy) | (ends >= num_heavy))
        hydrogens = np.maximum(begins[hydrogen_bonds], ends[hydrogen_bonds])
        parents = np.empty(num_atoms - num_heavy, dtype=np.intp)
        parents[hydrogens - num_heavy] = np.minimum(
            begins[hydrogen_bonds], ends[hydrogen_bonds]
        )

        matrix = np.empty((num_atoms, num_atoms))
        matrix[:num_heavy, :num_heavy] = heavy_distances
        to_hydrogens = heavy_distances[:, parents] + 1.0
        matrix[:num_heavy, num_heavy:] = to_hydrogens
        matrix[num_heavy:, :num_heavy] = to_hydrogens.T
        matrix[num_heavy:, num_heavy:] = heavy_distances[np.ix_(parents, parents)] + 2.0
        np.fill_diagonal(matrix, 0.0)

        # the matrix is already known, so the RDKit call in __init__ is skipped
        derived = cls.__new__(cls)
        derived.matrix = matrix
        return derived

    @cached_property
    def eccentricities(self) -> np.ndarray:
        return self.matrix.max(axis=0)

    @cached_property
    def radius(self) -> np.floating:
        return self.eccentricities.min()

    @cached_property
    def diameter(self) -> np.floating:
        return self.matrix.max()


class AdjacencyMatrix:
    hermitian = True

    def __init__(self, mol: Mol, use_bond_orders: bool = False):
        self._base: np.ndarray
        self._base = GetAdjacencyMatrix(
            mol, useBO=use_bond_orders, prefix=_cache_prefix(use_bond_orders)
        )
        self._orders = [self._base]

    @property
    def matrix(self) -> np.ndarray:
        return self._base

    def order(self, n: int = 1) -> np.ndarray:
        while len(self._orders) < n:
            self._orders.append(self._orders[-1].dot(self._base))
        return self._orders[n - 1]

    @cached_property
    def degree(self) -> np.ndarray:
        """Number of edges incident to each vertex (atom).

        By default ``use_bond_orders=False``, so bond orders are ignored and each
        bond counts as one edge. In that case, atom degree equals the atom
        valence, i.e. the number of bonds each atom forms.
        """
        return self._base.sum(axis=0, dtype=float)


class DistanceMatrix3D:
    def __init__(self, mol: Mol, conf_id: int = 0, use_atom_weights: bool = False):
        self.matrix: np.ndarray
        self.matrix = Get3DDistanceMatrix(
            mol, confId=conf_id, useAtomWts=use_atom_weights
        )

    @cached_property
    def eccentricities(self) -> np.ndarray:
        return self.matrix.max(axis=0)

    @cached_property
    def radius(self) -> np.floating:
        return self.eccentricities.min()

    @cached_property
    def diameter(self) -> np.floating:
        return self.matrix.max()
