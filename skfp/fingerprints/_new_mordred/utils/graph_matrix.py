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
        # how far the farthest atom is from each atom, and the extremes of that
        self.eccentricities = matrix.max(axis=0)
        self.radius = self.eccentricities.min() if len(matrix) else np.float64(0.0)
        self.diameter = matrix.max() if len(matrix) else np.float64(0.0)

    @classmethod
    def from_mol(
        cls, mol: Mol, use_bond_orders: bool = False, use_atom_weights: bool = False
    ) -> "DistanceMatrix":
        """
        Distances of a molecule, as RDKit computes them.
        """
        return cls(
            GetDistanceMatrix(
                mol,
                useBO=use_bond_orders,
                useAtomWts=use_atom_weights,
                prefix=_cache_prefix(use_bond_orders, use_atom_weights),
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
        """
        heavy_distances = distances.matrix
        num_heavy = len(heavy_distances)
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
    Adjacency matrix of molecular graph.

    Also calculates higher order variants, i.e. powers of that matrix. For
    n-th order adjacency matrix, the entries count the walks of a given
    length between two atoms.
    """

    hermitian = True

    def __init__(self, mol: Mol, use_bond_orders: bool = False):
        self.matrix = GetAdjacencyMatrix(
            mol, useBO=use_bond_orders, prefix=_cache_prefix(use_bond_orders)
        )
        # By default ``use_bond_orders=False``, so bond orders are ignored, and each
        # bond counts as one edge. In that case, atom degree equals the atom
        # valence, i.e. the number of bonds each atom forms.
        self.degree = self.matrix.sum(axis=0, dtype=float)
        self._powers = [self.matrix]

    def order(self, n: int = 1) -> np.ndarray:
        """
        Return the ``n``-th power of the matrix, computed on first use and kept.
        """
        while len(self._powers) < n:
            self._powers.append(self._powers[-1].dot(self.matrix))
        return self._powers[n - 1]


class DistanceMatrix3D:
    """
    Euclidean distance between every pair of atoms in a conformer.
    """

    def __init__(self, mol: Mol, conf_id: int = 0, use_atom_weights: bool = False):
        self.matrix = Get3DDistanceMatrix(
            mol, confId=conf_id, useAtomWts=use_atom_weights
        )
        self.eccentricities = self.matrix.max(axis=0)
        self.radius = self.eccentricities.min() if len(self.matrix) else np.float64(0.0)
        self.diameter = self.matrix.max() if len(self.matrix) else np.float64(0.0)


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
