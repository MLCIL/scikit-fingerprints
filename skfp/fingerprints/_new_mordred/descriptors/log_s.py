import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol

"""
Filter-it(TM) LogS descriptor, an aqueous solubility estimate.

http://silicos-it.be.s3-website-eu-west-1.amazonaws.com/software/filter-it/1.0.2/filter-it.html

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["FilterItLogS"]

_INTERCEPT = 0.89823
_MOL_WEIGHT_COEF = -0.10369

# SMARTS of the contributing groups -> their regression coefficient; each group
# contributes its coefficient once per match
_GROUP_COEFS = {
    "[NH0;X3;v3]": 0.71535,
    "[NH2;X3;v3]": 0.41056,
    "[nH0;X3]": 0.82535,
    "[OH0;X2;v2]": 0.31464,
    "[OH0;X1;v2]": 0.14787,
    "[OH1;X2;v2]": 0.62998,
    "[CH2;!R]": -0.35634,
    "[CH3;!R]": -0.33888,
    "[CH0;R]": -0.21912,
    "[CH2;R]": -0.23057,
    "[ch0]": -0.37570,
    "[ch1]": -0.22435,
    "F": -0.21728,
    "Cl": -0.49721,
    "Br": -0.57982,
    "I": -0.51547,
}

_GROUP_PATTERNS = [Chem.MolFromSmarts(smarts) for smarts in _GROUP_COEFS]
_GROUP_COEF_VALUES = np.asarray(list(_GROUP_COEFS.values()))


def calc(mol: Mol) -> np.ndarray:
    r"""
    Compute the Mordred Filter-it(TM) LogS descriptor.

    A linear model in the square root of the average molecular weight and the
    number of matches of each contributing group:

    .. math::

        \text{logS} = 0.89823 - 0.10369 \sqrt{MW} + \sum_i n_i c_i

    Note that the molecular weight here is the average one, unlike the exact
    weight the ``MW`` and ``AMW`` descriptors report.
    """
    group_counts = np.fromiter(
        (len(mol.GetSubstructMatches(pattern)) for pattern in _GROUP_PATTERNS),
        dtype=np.intp,
        count=len(_GROUP_PATTERNS),
    )
    log_s = (
        _INTERCEPT
        + _MOL_WEIGHT_COEF * np.sqrt(Descriptors.MolWt(mol))
        + group_counts @ _GROUP_COEF_VALUES
    )

    return np.asarray([log_s], dtype=np.float32)
