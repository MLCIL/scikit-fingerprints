from collections.abc import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def atoms_apply_func(
    f: Callable[[Atom], float], mol: Mol, dtype: str = "float"
) -> np.ndarray:
    """
    Apply a function to each atom of a molecule and return the results as a
    NumPy array of shape ``(N,)``, where ``N`` is the number of atoms.
    """
    return np.fromiter((f(a) for a in mol.GetAtoms()), dtype, mol.GetNumAtoms())


def preprocess_mol(
    mol: Mol,
    explicit_hydrogens: bool = False,
    kekulize: bool = False,
) -> Mol:
    """
    Preprocess an RDKit molecule before descriptor calculation.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule.

    explicit_hydrogens : bool, default=False
        If ``True``, add explicit hydrogens via ``AddHs``. If ``False``,
        remove hydrogens via ``RemoveHs``.

    kekulize : bool, default=False
        If ``True``, kekulize the molecule, converting aromatic bonds to
        alternating single and double bonds.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The preprocessed molecule, with all conformers removed.
    """
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)

    if kekulize:
        Chem.Kekulize(mol)

    return mol
