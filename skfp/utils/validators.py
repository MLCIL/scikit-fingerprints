import functools
from collections.abc import Callable, Sequence
from typing import Any

from rdkit.Chem import Mol, MolFromInchi, MolFromSmiles, MolToSmiles
from rdkit.Chem.PropertyMol import PropertyMol


def ensure_mols(X: Sequence[Any]) -> list[Mol]:
    """
    Ensure that all input sequence elements are RDKit ``Mol`` objects. Requires
    all input elements to be of the same type: string (SMILES or InChI  strings) or ``Mol``.
    In the case of SMILES or InChI strings, they are converted to RDKit ``Mol`` objects with
    default settings.
    """
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        types = {type(x) for x in X}
        raise TypeError(
            f"Passed values must be RDKit Mol objects, SMILES or InChI strings, got types: {types}"
        )

    mols = []
    for x in X:
        if isinstance(x, str):
            mols.append(MolFromInchi(x) if x.startswith("InChI=") else MolFromSmiles(x))
        else:
            mols.append(x)

    if any(x is None for x in mols):
        idx = mols.index(None)
        raise TypeError(f"Could not parse '{X[idx]}' at index {idx} as molecule")

    return mols


def ensure_smiles(X: Sequence[Any]) -> list[str]:
    """
    Ensure that all input sequence elements are SMILES strings. Requires all input
    elements to be of the same type: string (SMILES strings) or ``Mol``. In the case of
    RDKit ``Mol`` objects, they are converted to SMILES strings with default settings.
    """
    if not all(isinstance(x, (Mol, PropertyMol, str)) for x in X):
        types = {type(x) for x in X}
        raise TypeError(f"Passed values must be SMILES strings, got types: {types}")

    X = [MolToSmiles(x) if isinstance(x, Mol) else x for x in X]
    return X


def require_mols(X: Sequence[Any]) -> None:
    """
    Check that all inputs are RDKit ``Mol`` objects, raises TypeError otherwise.
    """
    for idx, x in enumerate(X):
        if not isinstance(x, (Mol, PropertyMol)):
            raise TypeError(
                f"Passed values must be RDKit Mol objects, got type {type(x)} at index {idx}"
            )


def require_mols_with_conf_ids(X: Sequence[Any]) -> Sequence[Mol]:
    """
    Check that all inputs are RDKit ``Mol`` objects with ``"conf_id"`` property
    set, i.e. with conformers computed and properly identified. Raises TypeError
    otherwise.
    """
    if not all(isinstance(x, (Mol, PropertyMol)) and x.HasProp("conf_id") for x in X):
        raise TypeError(
            "Passed data must be molecules (RDKit Mol objects) "
            "and each must have conf_id property set. "
            "You can use ConformerGenerator to add them."
        )
    return X


def require_strings(X: Sequence[Any]) -> None:
    """
    Check that all inputs are strings, raises TypeError otherwise.
    """
    for idx, x in enumerate(X):
        if not isinstance(x, str):
            raise TypeError(
                f"Passed values must be strings, got type {type(x)} at index {idx}"
            )


def require_atoms(min_atoms: int = 1, only_explicit=True) -> Callable:
    """
    Ensure molecule is nonempty or has at least min_atoms atoms.

    Used as a decorator for functions operating on a single molecule.
    Raises ValueError if conditions are not met.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(mol: Mol, *args, **kwargs):
            if mol.GetNumAtoms(onlyExplicit=only_explicit) < min_atoms:
                raise ValueError(
                    f"The molecule must have at least {min_atoms} atom(s), "
                    f"{func.__name__} cannot be calculated."
                )
            return func(mol, *args, **kwargs)

        return wrapper

    return decorator
