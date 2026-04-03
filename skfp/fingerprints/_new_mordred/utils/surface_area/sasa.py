from collections import defaultdict

import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_to_numpy
from skfp.fingerprints._new_mordred.utils.periodic_table import VDW_RADII

from ._mesh import SphereMesh


class SurfaceArea:
    r"""Calculate solvent accessible surface area.

    Parameters
    ----------
    radii : np.ndarray of shape (N,)
        Atomic radius + solvent radius vector.

    xyzs : np.ndarray of shape (N, 3)
        Atomic position matrix.

    level : int, default=4
        Mesh level. Subdivide icosahedron n-1 times.

        .. math::

            N_{\rm points} = 5 \times 4^{level} - 8
    """

    def __init__(self, radii: np.ndarray, xyzs: np.ndarray, level: int = 4):
        self.radii = radii
        self.radii_sq = radii**2
        self.xyzs = xyzs
        self._gen_neighbor_list()
        self.sphere = SphereMesh(level).vertices.T

    def _gen_neighbor_list(self):
        r = self.radii[:, np.newaxis] + self.radii

        d = np.sqrt(np.sum((self.xyzs[:, np.newaxis] - self.xyzs) ** 2, axis=2))

        ns = defaultdict(list)
        for i, j in np.transpose(np.nonzero(d <= r)):
            if i == j:
                continue

            ns[i].append((j, d[i, j]))

        for neighbors_list in ns.values():
            neighbors_list.sort(key=lambda i: i[1])

        self.neighbors = ns

    def atomic_sa(self, i: int) -> float:
        r"""Calculate atomic surface area.

        Parameters
        ----------
        i : int
            Atom index.

        Returns
        -------
        float
            Surface area of the atom.
        """
        sa = 4.0 * np.pi * self.radii_sq[i]

        neighbors = self.neighbors.get(i)

        if neighbors is None:
            return sa

        XYZi = self.xyzs[i, np.newaxis].T

        sphere = self.sphere * self.radii[i] + XYZi
        N = sphere.shape[1]

        for j, _ in neighbors:
            XYZj = self.xyzs[j, np.newaxis].T

            d2 = (sphere - XYZj) ** 2
            mask = (d2[0] + d2[1] + d2[2]) > self.radii_sq[j]
            sphere = np.compress(mask, sphere, axis=1)

        return sa * sphere.shape[1] / N

    def surface_area(self) -> list[float]:
        r"""Calculate all atomic surface areas.

        Returns
        -------
        list[float]
            Surface area for each atom.
        """
        return [self.atomic_sa(i) for i in range(len(self.radii))]

    @classmethod
    def from_mol(
        cls, mol: Mol, conformer: int = -1, solvent_radius: float = 1.4, level: int = 4
    ) -> "SurfaceArea":
        r"""Construct SurfaceArea from RDKit Mol.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Input molecule.

        conformer : int, default=-1
            Conformer ID.

        solvent_radius : float, default=1.4
            Solvent radius.

        level : int, default=4
            Mesh level.

        Returns
        -------
        SurfaceArea
            Constructed SurfaceArea instance.
        """
        rs = atoms_to_numpy(lambda a: VDW_RADII[a.GetAtomicNum()] + solvent_radius, mol)

        conf = mol.GetConformer(conformer)

        ps = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

        return cls(rs, ps, level)
