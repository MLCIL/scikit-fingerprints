from collections.abc import Callable

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdchem import Atom, Conformer


def atoms_to_numpy(
    f: Callable[[Atom], float], mol: Mol, dtype: str = "float"
) -> np.ndarray:
    """Apply a function to each atom and return results as a numpy array of shape (N,)."""
    return np.fromiter((f(a) for a in mol.GetAtoms()), dtype, mol.GetNumAtoms())


def conformer_to_numpy(conf: Conformer) -> np.ndarray:
    """Convert an RDKit Conformer to a numpy array of shape (N, 3)."""
    return np.array([list(conf.GetAtomPosition(i)) for i in range(conf.GetNumAtoms())])


def preprocess_mol(
    mol: Mol,
    explicit_hydrogens: bool = False,
    kekulize: bool = False,
    require_3D: bool = False,
    conformer_id: int = -1,
) -> tuple[Mol, np.ndarray | None]:
    """Preprocess an RDKit Mol before descriptor calculation.

    Args:
        mol: rdkit.Chem.Mol instance.
        explicit_hydrogens: If True, add explicit hydrogens (AddHs).
                            If False, remove hydrogens (RemoveHs).
        kekulize: If True, kekulize the molecule (convert aromatic bonds
                  to alternating single/double bonds).
        require_3D: If True, extract 3D coordinates from the conformer
                    before stripping conformers from the mol.
        conformer_id: Which conformer to use when extracting 3D coords.
                      Defaults to -1 (the default/first conformer).

    Returns
    -------
        tuple of (mol, coords):
            mol - the preprocessed RDKit Mol (conformers removed).
            coords - numpy array of shape (N, 3) if require_3D and a 3D
                conformer is available, otherwise None.
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
