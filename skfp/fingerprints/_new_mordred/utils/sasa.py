import numpy as np
from rdkit.Chem import Atom, Mol
from rdkit.Chem.rdFreeSASA import CalcSASA, SASAAlgorithm, SASAOpts

from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func
from skfp.fingerprints._new_mordred.utils.periodic_table import VAN_DER_WAALS_RADII


def solvent_accessible_surface_area(
    mol: Mol, conformer: int = -1, solvent_radius: float = 1.4
) -> np.ndarray:
    """
    Solvent accessible surface area (SASA).

    Computes per-atom solvent accessible surface area using the Lee-Richards
    algorithm, as implemented in :func:`rdkit.Chem.rdFreeSASA.CalcSASA`. Atomic
    van der Waals radii are taken from Mordred's table, which has none for a dummy
    atom or for elements from rutherfordium (104) on, so all areas are ``nan`` for
    a molecule with such an atom.

    Mordred computes SASA with its own dense Shrake-Rupley mesh. FreeSASA's
    Lee-Richards, at its default of 20 slices per atom, comes closer to that than
    its Shrake-Rupley, whose 100 test points RDKit offers no way to raise.

    All areas are also ``nan`` when two atoms of the same radius sit at exactly the
    same position, for which Lee-Richards returns ``nan`` or a wrong area. Such a
    conformer is broken, typically with fragments embedded on top of each other,
    and Mordred's areas for it come down to floating-point rounding.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule with at least one 3D conformer.

    conformer : int, default=-1
        Conformer ID passed to RDKit. The default value of ``-1`` selects
        the first conformer.

    solvent_radius : float, default=1.4
        Solvent probe radius, in angstroms, added to each atomic van der
        Waals radius. The default corresponds to a water probe.

    Returns
    -------
    per_atom_sasa : np.ndarray of shape (n_atoms,)
        Per-atom solvent accessible surface areas, in the same order as atoms
        in ``mol``.
    """
    atomic_nums = atoms_apply_func(Atom.GetAtomicNum, mol, np.intp)
    radii = VAN_DER_WAALS_RADII.lookup(atomic_nums)

    # CalcSASA raises on a molecule without atoms
    if radii.size == 0:
        return np.empty(0, dtype=np.float32)
    if np.isnan(radii).any():
        return np.full(len(radii), np.nan, dtype=np.float32)

    positions = mol.GetConformer(conformer).GetPositions()
    if _has_stacked_atoms(positions, radii):
        return np.full(len(radii), np.nan, dtype=np.float32)

    opts = SASAOpts()
    opts.algorithm = SASAAlgorithm.LeeRichards
    opts.probeRadius = solvent_radius
    CalcSASA(mol, radii.tolist(), confIdx=conformer, opts=opts)

    # read back by atom index, which is faster than iterating mol.GetAtoms()
    return atoms_apply_func(_read_sasa, mol, np.float32)


def _has_stacked_atoms(positions: np.ndarray, radii: np.ndarray) -> bool:
    """
    Whether two atoms of the same radius share exactly the same position.

    Lee-Richards divides by their zero distance apart. A smaller atom inside a
    larger one at the same position is fine, being buried, so radius is compared too.
    """
    keys = np.column_stack([positions, radii])
    return len(np.unique(keys, axis=0)) < len(keys)


def _read_sasa(atom: Atom) -> float:
    """Per-atom surface area, which CalcSASA leaves behind on the atoms."""
    return atom.GetDoubleProp("SASA")
