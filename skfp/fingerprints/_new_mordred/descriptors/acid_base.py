import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["nAcid", "nBase"]

_ACID_SMARTS = (
    "[O;H1]-[C,S,P]=O",
    "[*;-;!$(*~[*;+])]",
    "[NH](S(=O)=O)C(F)(F)F",
    "n1nnnc1",
)
_BASE_SMARTS = (
    "[NH2]-[CX4]",
    "[NH](-[CX4])-[CX4]",
    "N(-[CX4])(-[CX4])-[CX4]",
    "[*;+;!$(*~[*;-])]",
    "N=C-N",
    "N-C=N",
)

_ACID_PATTERN = Chem.MolFromSmarts(
    "[" + ",".join(f"$({smarts})" for smarts in _ACID_SMARTS) + "]"
)
_BASE_PATTERN = Chem.MolFromSmarts(
    "[" + ",".join(f"$({smarts})" for smarts in _BASE_SMARTS) + "]"
)


def calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Acidic/basic group count descriptors.

    Based on CDK AcidicGroupCountDescriptor and BasicGroupCountDescriptor.
    https://cdk.github.io/cdk/1.5/docs/api/org/openscience/cdk/qsar/descriptors/molecular/AcidicGroupCountDescriptor.html
    https://cdk.github.io/cdk/1.5/docs/api/org/openscience/cdk/qsar/descriptors/molecular/BasicGroupCountDescriptor.html
    """
    values = [
        len(mol.GetSubstructMatches(_ACID_PATTERN)),
        len(mol.GetSubstructMatches(_BASE_PATTERN)),
    ]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
