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
    f: Callable[[Atom], float], mol: Mol, dtype: str | np.dtype | type = "float"
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
    sanitize: bool = True,
) -> Mol:
    """
    Preprocess an RDKit molecule before descriptor calculation.

    By default (``sanitize=True``), starts with ``RemoveHs``, which re-runs
    ``SanitizeMol`` and resets aromaticity perception to a canonical state.
    This ensures consistent bond types regardless of prior operations such as
    conformer embedding, which can leave the molecule in a kekulized state.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule.

    explicit_hydrogens : bool, default=False
        If ``True``, add explicit hydrogens via ``AddHs`` as the final step.

    kekulize : bool, default=False
        If ``True``, kekulize the molecule after sanitization, converting
        aromatic bonds to alternating single and double bonds.

    sanitize : bool, default=True
        If ``True`` (default), start by removing hydrogens via ``RemoveHs``,
        which re-sanitizes the molecule and resets aromaticity perception.
        Set to ``False`` to skip this step and preserve the current molecule
        state, e.g. when 3D conformer coordinates must not be discarded.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The preprocessed molecule.
    """
    if sanitize:
        mol = Chem.RemoveHs(mol, updateExplicitCount=True)

    if kekulize:
        Chem.Kekulize(mol)

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    return mol
