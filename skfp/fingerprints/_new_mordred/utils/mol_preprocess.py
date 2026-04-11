from collections.abc import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom, Conformer

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def atoms_to_numpy(
    f: Callable[[Atom], float], mol: Mol, dtype: str = "float"
) -> np.ndarray:
    """
    Apply a function to each atom of a molecule and return the results as a
    NumPy array of shape ``(N,)``, where ``N`` is the number of atoms.
    """
    return np.fromiter((f(a) for a in mol.GetAtoms()), dtype, mol.GetNumAtoms())


def conformer_to_numpy(conf: Conformer) -> np.ndarray:
    """
    Convert an RDKit ``Conformer`` to a NumPy array of shape ``(N, 3)``,
    containing the 3D coordinates of each atom.
    """
    return np.array([list(conf.GetAtomPosition(i)) for i in range(conf.GetNumAtoms())])


def preprocess_mol(
    mol: Mol,
    explicit_hydrogens: bool = False,
    kekulize: bool = False,
    require_3D: bool = False,
    conformer_id: int = -1,
) -> tuple[Mol, np.ndarray | None]:
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

    require_3D : bool, default=False
        If ``True``, extract 3D coordinates from the selected conformer
        before stripping all conformers from the molecule.

    conformer_id : int, default=-1
        Conformer ID to use when extracting 3D coordinates. The default
        value of ``-1`` selects the most recently added conformer.

    Returns
    -------
    mol : rdkit.Chem.Mol
        The preprocessed molecule, with all conformers removed.

    coords : numpy.ndarray or None
        NumPy array of shape ``(N, 3)`` with 3D coordinates if
        ``require_3D`` is ``True`` and a 3D conformer was available,
        otherwise ``None``.
    """
    if explicit_hydrogens:
        m = Chem.AddHs(mol)
    else:
        m = Chem.RemoveHs(mol, updateExplicitCount=True)

    if kekulize:
        Chem.Kekulize(m)

    coords = None
    if require_3D:
        try:
            conf = m.GetConformer(conformer_id)
            if conf.Is3D():
                coords = conformer_to_numpy(conf)
        except ValueError:
            pass

    m.RemoveAllConformers()

    return m, coords
