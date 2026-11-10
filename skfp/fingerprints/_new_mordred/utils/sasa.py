import numpy as np
from rdkit.Chem import Atom, Mol, RWMol
from rdkit.Chem.rdFreeSASA import CalcSASA, SASAAlgorithm, SASAOpts

from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func
from skfp.fingerprints._new_mordred.utils.periodic_table import VAN_DER_WAALS_RADII


def solvent_accessible_surface_area(
    mol: Mol,
    atomic_nums: np.ndarray,
    distances: np.ndarray,
    conformer: int = -1,
    solvent_radius: float = 1.4,
) -> np.ndarray:
    """
    Solvent accessible surface area (SASA).

    Computes per-atom solvent accessible surface area using the Shrake-Rupley
    algorithm, as implemented in :func:`rdkit.Chem.rdFreeSASA.CalcSASA`. Mordred
    uses the same algorithm, though with a denser mesh of test points. Atomic van
    der Waals radii are taken from Mordred's table, which has none for a dummy atom
    or for elements from rutherfordium (104) on. As in Mordred, such an atom gets a
    ``nan`` area and covers no other atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule with at least one 3D conformer.

    atomic_nums : np.ndarray of shape (n_atoms,)
        Atomic number of every atom of ``mol``.

    distances : np.ndarray of shape (n_atoms, n_atoms)
        Euclidean distances between the atoms of ``mol`` in ``conformer``, as
        :func:`rdkit.Chem.Get3DDistanceMatrix` computes them.

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
    radii = VAN_DER_WAALS_RADII.lookup(atomic_nums)
    # the sphere Shrake-Rupley measures, reaching out to the probe centre
    probe_reach = radii + solvent_radius

    # CalcSASA reads an atom's first neighbor without checking that it has one, so
    # an atom whose probe sphere no other one's overlaps either faults the process
    # or comes out silently wrong (31.09 instead of 135.19 for a lone argon, RDKit
    # 2026.03.5). Such an atom is a whole unoccluded sphere, which is filled in
    # here, and it occludes nothing, so the rest are unaffected by its leaving.
    # An atom without radius has a nan reach, so it overlaps nothing either and is
    # left out with a nan area, as Mordred leaves it out of every atom's neighbors.
    overlaps = distances < probe_reach + probe_reach[:, np.newaxis]
    np.fill_diagonal(overlaps, False)  # an atom always overlaps itself
    # not only lone atoms: also those without radius, which overlap nothing
    is_lone = ~overlaps.any(axis=1)
    areas = (4.0 * np.pi * probe_reach**2).astype(np.float32)

    if is_lone.any():
        editable = RWMol(mol)
        for atom_idx in reversed(np.flatnonzero(is_lone).tolist()):
            editable.RemoveAtom(atom_idx)  # keeps the conformer of the rest
        mol, radii = editable.GetMol(), radii[~is_lone]

    if radii.size:
        opts = SASAOpts()
        opts.algorithm = SASAAlgorithm.ShrakeRupley
        opts.probeRadius = solvent_radius
        CalcSASA(mol, radii.tolist(), confIdx=conformer, opts=opts)
        # read back by atom index, which is faster than iterating mol.GetAtoms()
        areas[~is_lone] = atoms_apply_func(_read_sasa, mol, np.float32)

    return areas


def _read_sasa(atom: Atom) -> float:
    """Per-atom surface area, which CalcSASA leaves behind on the atoms."""
    return atom.GetDoubleProp("SASA")
