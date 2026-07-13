import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdFreeSASA import CalcSASA, SASAAlgorithm, SASAOpts

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    get_van_der_waals_radius_rdkit,
)


def solvent_accessible_surface_area(
    mol: Mol, conformer: int = -1, solvent_radius: float = 1.4
) -> np.ndarray:
    """
    Solvent accessible surface area (SASA).

    Computes per-atom solvent accessible surface area using the Shrake-Rupley
    algorithm, as implemented in :func:`rdkit.Chem.rdFreeSASA.CalcSASA`. Atomic
    van der Waals radii are taken from RDKit's periodic table.

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

    Returns
    -------
    per_atom_sasa : np.ndarray of shape (n_atoms,)
        Per-atom solvent accessible surface areas, in the same order as atoms
        in ``mol``.
    """
    radii = [get_van_der_waals_radius_rdkit(atom) for atom in mol.GetAtoms()]

    opts = SASAOpts()
    opts.algorithm = SASAAlgorithm.ShrakeRupley
    opts.probeRadius = solvent_radius
    CalcSASA(mol, radii, confIdx=conformer, opts=opts)

    return np.fromiter(
        (atom.GetDoubleProp("SASA") for atom in mol.GetAtoms()), dtype=np.float32
    )
