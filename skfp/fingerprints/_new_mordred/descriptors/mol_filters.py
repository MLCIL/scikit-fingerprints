import numpy as np

from skfp.fingerprints._new_mordred.utils.molecular_properties import (
    MolecularProperties,
)

"""
Drug-likeness rule descriptors: the Lipinski rule of five and the Ghose filter.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Lipinski", "GhoseFilter"]


def calc(mol_properties: MolecularProperties) -> np.ndarray:
    """
    Compute the Mordred drug-likeness rule descriptors, as 1.0 for a molecule
    passing the rule and 0.0 for one failing it.

    ``Lipinski`` is the rule of five: at most 5 hydrogen bond donors, at most 10
    acceptors, molecular weight up to 500 and Wildman-Crippen LogP up to 5.

    ``GhoseFilter`` bounds molecular weight to [160, 480], atom count (hydrogens
    included) to [20, 70], LogP to [-0.4, 5.6] and molar refractivity to [40, 130].
    """
    lipinski_rule_of_five = (
        mol_properties.num_h_bond_donors <= 5
        and mol_properties.num_h_bond_acceptors <= 10
        and mol_properties.exact_mol_wt <= 500
        and mol_properties.log_p <= 5
    )
    ghose_filter = (
        160 <= mol_properties.exact_mol_wt <= 480
        and 20 <= mol_properties.num_atoms <= 70
        and -0.4 <= mol_properties.log_p <= 5.6
        and 40 <= mol_properties.molar_refractivity <= 130
    )

    return np.asarray([lipinski_rule_of_five, ghose_filter], dtype=np.float32)
