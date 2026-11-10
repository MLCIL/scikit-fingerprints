from functools import cached_property

import numpy as np
from rdkit.Chem import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix, Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


class DistanceMatrix:
    """
    Topological distance between every pair of atoms, with the derived quantities
    the descriptors ask for.

    Wraps a matrix that is either read from a molecule with :meth:`from_mol` or
    built from the matrix of a smaller molecule with :meth:`with_hydrogens_added`.
    """

    hermitian = True

    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    # how far the farthest atom is from each atom, and the extremes of that, computed
    # on first use, since most matrices are only ever read as a whole

    @cached_property
    def eccentricities(self) -> np.ndarray:
        return self.matrix.max(axis=0)

    @cached_property
    def radius(self) -> np.float64:
        return self.eccentricities.min() if len(self.matrix) else np.float64(0.0)

    @cached_property
    def diameter(self) -> np.float64:
        # the largest distance of all is the largest of the per-atom largest ones
        return self.eccentricities.max() if len(self.matrix) else np.float64(0.0)

    @classmethod
    def from_mol(cls, mol: Mol, use_bond_orders: bool = False) -> "DistanceMatrix":
        """
        Distances of a molecule, as RDKit computes them, optionally with every bond
        weighted by its order.
        """
        return cls(
            GetDistanceMatrix(
                mol, useBO=use_bond_orders, prefix=_cache_prefix(use_bond_orders)
            )
        )

    @classmethod
    def with_hydrogens_added(
        cls, distances: "DistanceMatrix", props_hydrogens: AtomicProperties
    ) -> "DistanceMatrix":
        """
        Distances of ``AddHs(mol)``, built from the distances of ``mol``.

        The added hydrogens are terminal atoms, so the shortest path to a hydrogen
        is the shortest path to the atom it hangs off plus that one bond, and the
        one between two hydrogens has such a bond at both ends. All-pairs shortest
        paths with RDKit is more expensive than such filling.

        Atoms of different fragments are not connected by any path, and RDKit reports
        a fixed placeholder distance for such a pair, which those added bonds must
        not grow.

        The hydrogens may also have been added to another molecule ``mol`` was
        derived from, which numbers its atoms differently when it holds explicit
        hydrogens. The distances are then computed from ``props_hydrogens.mol``.
        """
        heavy_distances = distances.matrix
        num_heavy = len(heavy_distances)
        if not _has_terminal_atoms_appended(heavy_distances, props_hydrogens):
            return cls.from_mol(props_hydrogens.mol)

        num_atoms = props_hydrogens.num_atoms
        parents = _hydrogen_parents(props_hydrogens, num_heavy)

        matrix = np.empty((num_atoms, num_atoms))
        matrix[:num_heavy, :num_heavy] = heavy_distances
        to_hydrogens = heavy_distances[:, parents] + 1.0
        matrix[:num_heavy, num_heavy:] = to_hydrogens
        matrix[num_heavy:, :num_heavy] = to_hydrogens.T
        matrix[num_heavy:, num_heavy:] = heavy_distances[parents][:, parents] + 2.0

        # RDKit uses a placeholder distance between atoms of from different fragments
        _rdkit_disconnected_distance = 1e8
        np.minimum(matrix, _rdkit_disconnected_distance, out=matrix)
        np.fill_diagonal(matrix, 0.0)
        return cls(matrix)


def _has_terminal_atoms_appended(
    heavy_distances: np.ndarray, props_hydrogens: AtomicProperties
) -> bool:
    """
    Whether the molecule of ``props_hydrogens`` is the one ``heavy_distances`` were
    computed for, with only terminal atoms appended: its first atoms are bonded
    exactly as the pairs one bond apart in ``heavy_distances``, and every other atom
    has a single bond, to one of those first atoms.
    """
    num_heavy = len(heavy_distances)
    num_added = props_hydrogens.num_atoms - num_heavy
    begins, ends = props_hydrogens.bond_begin_idxs, props_hydrogens.bond_end_idxs
    lower, upper = np.minimum(begins, ends), np.maximum(begins, ends)

    # every appended atom ends exactly one bond, which starts among the first atoms
    to_added = upper >= num_heavy
    if (
        num_added < 0
        or (lower[to_added] >= num_heavy).any()
        or not np.array_equal(
            np.bincount(upper[to_added] - num_heavy, minlength=num_added),
            np.ones(num_added, dtype=np.intp),
        )
    ):
        return False

    # the bonds among the first atoms are the pairs exactly one bond apart
    # the matrix is symmetric, so each such pair appears in it twice
    lower, upper = lower[~to_added], upper[~to_added]
    num_adjacent_pairs = np.count_nonzero(heavy_distances == 1.0) // 2
    return len(lower) == num_adjacent_pairs and bool(
        (heavy_distances[lower, upper] == 1.0).all()
    )


def _hydrogen_parents(props_hydrogens: AtomicProperties, num_heavy: int) -> np.ndarray:
    """
    For every hydrogen that ``AddHs`` appended, the atom it hangs off.

    Those hydrogens come after every other atom and have one bond each, so the lower
    end of such a bond is the atom it belongs to.
    """
    begins = props_hydrogens.bond_begin_idxs
    ends = props_hydrogens.bond_end_idxs
    hydrogen_bonds = np.flatnonzero((begins >= num_heavy) | (ends >= num_heavy))
    hydrogens = np.maximum(begins[hydrogen_bonds], ends[hydrogen_bonds])

    parents = np.empty(props_hydrogens.num_atoms - num_heavy, dtype=np.intp)
    parents[hydrogens - num_heavy] = np.minimum(
        begins[hydrogen_bonds], ends[hydrogen_bonds]
    )
    return parents


class AdjacencyMatrix:
    """
    Adjacency matrix of molecular graph, every bond counting as one edge.
    """

    hermitian = True

    def __init__(self, mol: Mol):
        self.matrix = GetAdjacencyMatrix(mol)


class DistanceMatrix3D:
    """
    Euclidean distance between every pair of atoms in a conformer.
    """

    def __init__(self, mol: Mol, conf_id: int = 0):
        self.matrix = Get3DDistanceMatrix(mol, confId=conf_id)

    # the same derived quantities as for DistanceMatrix, computed on first use

    @cached_property
    def eccentricities(self) -> np.ndarray:
        return self.matrix.max(axis=0)

    @cached_property
    def radius(self) -> np.float64:
        return self.eccentricities.min() if len(self.matrix) else np.float64(0.0)

    @cached_property
    def diameter(self) -> np.float64:
        return self.eccentricities.max() if len(self.matrix) else np.float64(0.0)


def _cache_prefix(use_bond_orders: bool) -> str:
    """
    Name under which RDKit memoizes a distance matrix on the molecule.

    The bond-order weighted matrix needs a name of its own, because asking for it
    under the default (empty) name hands back an unweighted matrix computed earlier.
    The unweighted matrix keeps that default name, which is where RDKit itself, and
    therefore most of the memoized matrices, put it.
    """
    return "skfp_bond_orders" if use_bond_orders else ""
