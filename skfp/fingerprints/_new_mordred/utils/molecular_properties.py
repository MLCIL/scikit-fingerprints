from rdkit.Chem import Crippen, Descriptors, Mol, rdMolDescriptors

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


class MolecularProperties:
    """
    Whole-molecule RDKit property values shared by several descriptor modules.

    The Crippen and hydrogen bond values come from SMARTS matching over the whole
    molecule, which costs enough to be worth doing once for all the descriptors
    that read them, rather than once per descriptor.

    Note that from_mol() should be used to calculate this.
    """

    def __init__(
        self,
        *,
        log_p: float,
        molar_refractivity: float,
        exact_mol_wt: float,
        num_atoms: int,
        num_h_bond_acceptors: int,
        num_h_bond_donors: int,
    ):
        # Wildman-Crippen LogP and molar refractivity
        self.log_p = log_p
        self.molar_refractivity = molar_refractivity
        self.exact_mol_wt = exact_mol_wt
        # hydrogens included, whether they are explicit atoms or not
        self.num_atoms = num_atoms
        self.num_h_bond_acceptors = num_h_bond_acceptors
        self.num_h_bond_donors = num_h_bond_donors

    @classmethod
    def from_mol(cls, mol: Mol) -> "MolecularProperties":
        """
        Read every property off a molecule.
        """
        return cls(
            log_p=Crippen.MolLogP(mol),
            molar_refractivity=Crippen.MolMR(mol),
            exact_mol_wt=Descriptors.ExactMolWt(mol),
            num_atoms=rdMolDescriptors.CalcNumAtoms(mol),
            num_h_bond_acceptors=rdMolDescriptors.CalcNumHBA(mol),
            num_h_bond_donors=rdMolDescriptors.CalcNumHBD(mol),
        )
