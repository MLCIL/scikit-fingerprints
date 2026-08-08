from rdkit.Chem import Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds

from skfp.utils.validators import require_atoms


@require_atoms()
def average_molecular_weight(mol: Mol) -> float:
    """
    Average molecular weight.

    Calculates the average molecular weight of the molecule, defined as the molecular
    weight divided by the number of atoms.

    This is different from "average molecular weight" in the context of isotopes [1]_.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the average molecular weight is to be calculated.

    References
    ----------
    .. [1] `<https://chemistry.stackexchange.com/questions/150993/discrepancy-when-calulating-mol-weights-with-chemsketch-and-python-rdkit/150999#150999>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import average_molecular_weight
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> average_molecular_weight(mol)
    13.018999999999998
    """
    return MolWt(mol) / mol.GetNumAtoms()


def bond_count(mol: Mol, bond_type: str | None = None) -> int:
    """
    Bond count.

    Counts the total number of bonds. If a specific bond type is specified,
    returns count of only that type.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the bond count is to be calculated.

    bond_type : str, default=None
        If ``None``, returns the total number of bonds. Otherwise, the valid
        options are RDKit bond types [1]_, e.g. "SINGLE", "DOUBLE", "TRIPLE",
        "AROMATIC".

    References
    ----------
    .. [1] `<https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import bond_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> bond_count(mol)
    6
    >>> bond_count(mol, "AROMATIC")
    6
    >>> bond_count(mol, "DOUBLE")
    0
    """
    if bond_type:
        return sum(
            1 for bond in mol.GetBonds() if bond.GetBondType().name == bond_type.upper()
        )
    return mol.GetNumBonds()


def element_atom_count(mol: Mol, atom_id: int | str) -> int:
    """
    Element atom count.

    Calculates the count of atoms of a specific type. In case of hydrogens,
    the total number is returned, counting both explicit and implicit ones.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the atom count is to be calculated.

    atom_id : int or str
        The atomic number of the atom type, e.g. 6 for carbon, 1 for hydrogen
        or symbol of the atom type, e.g. "C" for carbon,  "H" for hydrogen.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import element_atom_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> element_atom_count(mol, "C")
    6
    >>> element_atom_count(mol, 6)
    6

    >>> mol = MolFromSmiles("CCO")  # Ethanol
    >>> element_atom_count(mol, "H")
    6
    >>> element_atom_count(mol, 1)
    6
    """
    if atom_id in (1, "H"):
        # includeNeighbors also counts hydrogens present as explicit atoms in the
        # graph, e.g. after AddHs(), which GetTotalNumHs() alone would miss
        return sum(
            atom.GetTotalNumHs(includeNeighbors=True)
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() != 1
        )
    else:
        return sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == atom_id or atom.GetSymbol() == atom_id
        )


def heavy_atom_count(mol: Mol) -> int:
    """
    Heavy atom count.

    Calculates the number of heavy atoms (non-hydrogens) in the molecule.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the total atom count is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import heavy_atom_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> heavy_atom_count(mol)
    6
    """
    return mol.GetNumHeavyAtoms()


def molecular_weight(mol: Mol) -> float:
    """
    Molecular weight.

    Calculates the molecular weight of the molecule. This is average
    molecular weight in terms of isotopes [1]_.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the molecular weight is to be calculated.

    References
    ----------
    .. [1] `<https://chemistry.stackexchange.com/questions/150993/discrepancy-when-calulating-mol-weights-with-chemsketch-and-python-rdkit/150999#150999>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import molecular_weight
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> molecular_weight(mol)
    78.11399999999999
    """
    return MolWt(mol)


def number_of_rings(mol: Mol) -> int:
    """
    Number of rings.

    Calculates the total number of rings in the molecule.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the number of rings is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import number_of_rings
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> number_of_rings(mol)
    1
    """
    return CalcNumRings(mol)


def number_of_rotatable_bonds(mol: Mol) -> int:
    """
    Number of rotatable bonds.

    Calculates the total number of rotatable bonds in the molecule.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the number of rotatable bonds is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import number_of_rotatable_bonds
    >>> mol = MolFromSmiles("C=CC=C")  # Butadiene
    >>> number_of_rotatable_bonds(mol)
    1
    """
    return CalcNumRotatableBonds(mol)


def total_atom_count(mol: Mol) -> int:
    """
    Total atom count.

    Calculates the total number of atoms in the molecule. Includes hydrogens,
    both explicit and implicit.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which the total atom count is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import total_atom_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> total_atom_count(mol)
    12
    """
    return mol.GetNumAtoms(onlyExplicit=False)
