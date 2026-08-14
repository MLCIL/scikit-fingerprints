import numpy as np
from rdkit.Chem import Mol, MolToSmiles, rdPartialCharges
from rdkit.Chem.rdForceFieldHelpers import MMFFGetMoleculeProperties


def atomic_partial_charges(
    mol: Mol,
    partial_charge_model: str = "formal",
    charge_errors: str = "raise",
) -> np.ndarray:
    """
    Atomic partial charges.

    Calculate the atomic partial charges for all atoms in the molecule, using a given
    computational model. Note that it may fail for some molecules, e.g. Gasteiger model
    cannot compute charges for metals.

    Parameters
    ----------
    mol : RDKit ``Mol`` object
        The molecule for which to calculate the atomic partial charges.

    partial_charge_model : {"Gasteiger", "MMFF94", "formal", "precomputed"}, default="formal"
        Which model to use to compute atomic partial charges. Default ``"formal"``
        computes formal charges, and is the simplest and most error-resistant one.
        ``"precomputed"`` assumes that every atom of the input molecule has a
        "charge" float property set, e.g. with ``atom.SetDoubleProp("charge", value)``.

    charge_errors : {"raise", "ignore", "zero"}, default="raise"
        How to handle errors during calculation of atomic partial charges. ``"raise"``
        immediately raises any errors. ``"ignore"`` returns ``np.nan`` for all atoms that
        failed the computation. ``"zero"`` uses default value of 0 to fill all problematic
        charges.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors import atomic_partial_charges
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> atomic_partial_charges(mol)
    array([0., 0., 0., 0., 0., 0.])
    """
    atoms = mol.GetAtoms()

    if partial_charge_model == "Gasteiger":
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in atoms]
    elif partial_charge_model == "MMFF94":
        values = MMFFGetMoleculeProperties(mol)
        charges = [
            values.GetMMFFPartialCharge(i) if values else None
            for i in range(len(atoms))
        ]
    elif partial_charge_model == "formal":
        charges = [atom.GetFormalCharge() for atom in atoms]
    elif partial_charge_model == "precomputed":
        # RDKit raises a bare KeyError if the property is missing, which gives no
        # hint about what the caller was supposed to set
        if not all(atom.HasProp("charge") for atom in atoms):
            raise ValueError(
                'partial_charge_model="precomputed" requires every atom to have a '
                '"charge" float property set, e.g. with '
                'atom.SetDoubleProp("charge", value)'
            )
        charges = [atom.GetDoubleProp("charge") for atom in atoms]
    else:
        raise ValueError(
            f'Partial charge model "{partial_charge_model}" is not supported'
        )

    charges_problem = any(charge is None or np.isnan(charge) for charge in charges)
    if charge_errors == "raise" and charges_problem:
        smiles = MolToSmiles(mol)
        raise ValueError(
            f"Failed to compute at least one atom partial charge for {smiles}"
        )
    elif charge_errors == "zero":
        charges = np.nan_to_num(np.array(charges, dtype=float), nan=0)
    else:  # "ignore"
        charges = np.nan_to_num(
            np.array(charges, dtype=float), nan=np.nan, posinf=np.nan, neginf=np.nan
        )

    return np.asarray(charges, dtype=float)
