from collections.abc import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom, Bond

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def atoms_apply_func(
    f: Callable[[Atom], float], mol: Mol, dtype: str | np.dtype | type = "float"
) -> np.ndarray:
    """
    Apply a function to each atom of a molecule and return the results as a
    NumPy array of shape ``(N,)``, where ``N`` is the number of atoms.

    Atoms are fetched by index rather than by iterating ``mol.GetAtoms()``, whose
    Python-level sequence wrapper costs more than the atom lookup itself.
    """
    num_atoms = mol.GetNumAtoms()
    return np.fromiter(
        (f(mol.GetAtomWithIdx(idx)) for idx in range(num_atoms)), dtype, num_atoms
    )


def bonds_apply_func(
    f: Callable[[Bond], float], mol: Mol, dtype: str | np.dtype | type = "float"
) -> np.ndarray:
    """
    Apply a function to each bond of a molecule and return the results as a
    NumPy array of shape ``(B,)``, where ``B`` is the number of bonds.

    Bonds are fetched by index, for the same reason as in :func:`atoms_apply_func`.
    """
    num_bonds = mol.GetNumBonds()
    return np.fromiter(
        (f(mol.GetBondWithIdx(idx)) for idx in range(num_bonds)), dtype, num_bonds
    )


def preprocess_mol(
    mol: Mol,
    explicit_hydrogens: bool = False,
    kekulize: bool = False,
    sanitize: bool = True,
) -> Mol:
    """
    Preprocess an RDKit molecule before descriptor calculation, building one of the
    molecule variants that Mordred computes its descriptors from.

    Mordred builds every variant from the input molecule: the hydrogen-explicit one
    with ``AddHs``, and the hydrogen-suppressed one with ``RemoveHs``, which also
    re-sanitizes it, resetting aromaticity perception to a canonical state. Either
    one is kekulized last. The hydrogen-explicit molecule is never re-sanitized, so
    it keeps the bond types, aromaticity, atom order and conformers of the input.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule.

    explicit_hydrogens : bool, default=False
        If ``True``, add explicit hydrogens via ``AddHs``, before any kekulization.
        Otherwise, remove them via ``RemoveHs``, unless ``sanitize=False``.

    kekulize : bool, default=False
        If ``True``, kekulize the molecule as the last step, converting aromatic
        bonds to alternating single and double bonds.

    sanitize : bool, default=True
        Only used with ``explicit_hydrogens=False``. If ``True`` (default), remove
        hydrogens via ``RemoveHs``, which re-sanitizes the molecule. Set to ``False``
        for a molecule that already is one of the variants, e.g. to kekulize it,
        which then only copies it first.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The preprocessed molecule, always a new one.
    """
    # hydrogens first, as Mordred does: a ring with more than one valid Kekule
    # structure resolves differently once they are explicit
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    elif sanitize:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)
    else:
        # Kekulize works in place, so the caller's molecule keeps its aromatic bonds
        mol = Chem.Mol(mol)

    if kekulize:
        Chem.Kekulize(mol)

    return mol
