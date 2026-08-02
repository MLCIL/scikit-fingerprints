import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.rdFreeSASA import CalcSASA, SASAAlgorithm, SASAOpts

from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func
from skfp.fingerprints._new_mordred.utils.periodic_table import VAN_DER_WAALS_RADII


def solvent_accessible_surface_area(
    mol: Mol, conformer: int = -1, solvent_radius: float = 1.4
) -> np.ndarray:
    """
    Solvent accessible surface area (SASA).

    Computes per-atom solvent accessible surface area using the Shrake-Rupley
    algorithm, as implemented in :func:`rdkit.Chem.rdFreeSASA.CalcSASA`.

    Atomic van der Waals radii come from the same table mordred-community uses
    (Handbook of Chemistry and Physics, 94th edition), not from RDKit's periodic
    table, whose radii are up to 0.15 A larger and pull the CPSA descriptors well
    away from their mordred reference values.

    The areas still differ from mordred's by a few percent, because mordred
    integrates over a 5112-point icosphere while FreeSASA's Shrake-Rupley uses 100
    test points and RDKit exposes no way to raise that.

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
        in ``mol``. All NaN if any atom has no tabulated radius.
    """
    atomic_nums = atoms_apply_func(Atom.GetAtomicNum, mol, np.intp)
    radii = VAN_DER_WAALS_RADII.lookup(atomic_nums)

    # the table stops at rutherfordium, and FreeSASA segfaults on a NaN radius
    if not np.isfinite(radii).all():
        return np.full(len(radii), np.nan, dtype=np.float32)

    opts = SASAOpts()
    opts.algorithm = SASAAlgorithm.ShrakeRupley
    opts.probeRadius = solvent_radius
    CalcSASA(mol, radii.tolist(), confIdx=conformer, opts=opts)

    return atoms_apply_func(_read_sasa, mol, np.float32)


def _read_sasa(atom: Atom) -> float:
    """Per-atom surface area, which CalcSASA leaves behind on the atoms."""
    return atom.GetDoubleProp("SASA")
