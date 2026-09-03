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
    """

    def __init__(self, mol: Mol):
        # Wildman-Crippen LogP and molar refractivity
        self.log_p = Crippen.MolLogP(mol)
        self.molar_refractivity = Crippen.MolMR(mol)
        self.exact_mol_wt = Descriptors.ExactMolWt(mol)
        # hydrogens included, whether they are explicit atoms or not
        self.num_atoms = rdMolDescriptors.CalcNumAtoms(mol)
        self.num_h_bond_acceptors = rdMolDescriptors.CalcNumHBA(mol)
        self.num_h_bond_donors = rdMolDescriptors.CalcNumHBD(mol)
