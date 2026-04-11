from rdkit.Chem import Mol
from rdkit.Chem.rdFreeSASA import CalcSASA, SASAAlgorithm, SASAOpts

from skfp.fingerprints._new_mordred.utils.periodic_table import vdw_radii


class SurfaceArea:
    r"""Calculate solvent accessible surface area via RDKit's Shrake-Rupley.

    Thin wrapper around :func:`rdkit.Chem.rdFreeSASA.CalcSASA` that uses
    van der Waals radii from :mod:`periodic_table` and exposes the same
    ``from_mol`` / ``surface_area`` interface as the previous manual port.
    """

    def __init__(self, per_atom_sa: list[float]):
        self._per_atom_sa = per_atom_sa

    def surface_area(self) -> list[float]:
        r"""Return per-atom surface areas."""
        return self._per_atom_sa

    @classmethod
    def from_mol(
        cls, mol: Mol, conformer: int = -1, solvent_radius: float = 1.4
    ) -> "SurfaceArea":
        r"""Construct SurfaceArea from an RDKit Mol.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Input molecule with at least one 3D conformer.

        conformer : int, default=-1
            Conformer ID passed to RDKit.

        solvent_radius : float, default=1.4
            Solvent probe radius added to each atomic vdW radius.
        """
        radii = [vdw_radii(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

        opts = SASAOpts()
        opts.algorithm = SASAAlgorithm.ShrakeRupley
        opts.probeRadius = solvent_radius
        CalcSASA(mol, radii, confIdx=conformer, opts=opts)

        per_atom = [float(atom.GetProp("SASA")) for atom in mol.GetAtoms()]
        return cls(per_atom)
