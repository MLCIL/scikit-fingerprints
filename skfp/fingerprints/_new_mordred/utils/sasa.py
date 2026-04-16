from rdkit.Chem import Mol
from rdkit.Chem.rdFreeSASA import CalcSASA, SASAAlgorithm, SASAOpts

from skfp.fingerprints._new_mordred.utils.periodic_table import vdw_radii


class SurfaceArea:
    """
    Solvent accessible surface area (SASA).

    Computes per-atom solvent accessible surface area using the Shrake-Rupley
    algorithm, as implemented in :func:`rdkit.Chem.rdFreeSASA.CalcSASA`. Atomic
    van der Waals radii are taken from RDKit's periodic table.
    """

    def __init__(self, per_atom_sa: list[float]):
        self._per_atom_sa = per_atom_sa

    def surface_area(self) -> list[float]:
        """
        Return per-atom solvent accessible surface areas, in the same order
        as atoms in the underlying molecule.
        """
        return self._per_atom_sa

    @classmethod
    def from_mol(
        cls, mol: Mol, conformer: int = -1, solvent_radius: float = 1.4
    ) -> "SurfaceArea":
        """
        Construct a ``SurfaceArea`` instance from an RDKit molecule.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Input molecule with at least one 3D conformer.

        conformer : int, default=-1
            Conformer ID passed to RDKit. The default value of ``-1`` selects
            the most recently added conformer.

        solvent_radius : float, default=1.4
            Solvent probe radius, in angstroms, added to each atomic van der
            Waals radius. The default corresponds to a water probe.
        """
        radii = [vdw_radii(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

        opts = SASAOpts()
        opts.algorithm = SASAAlgorithm.ShrakeRupley
        opts.probeRadius = solvent_radius
        CalcSASA(mol, radii, confIdx=conformer, opts=opts)

        per_atom = [atom.GetDoubleProp("SASA") for atom in mol.GetAtoms()]
        return cls(per_atom)
