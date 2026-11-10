import numpy as np
from rdkit.Chem import Atom, HybridizationType, Mol
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_SP = int(HybridizationType.SP)
_SP2 = int(HybridizationType.SP2)
_SP3 = int(HybridizationType.SP3)
# hybridizations counted as SP3, as in Mordred
_SP3_HYBRIDIZATIONS = np.array(
    [_SP3, int(HybridizationType.SP3D), int(HybridizationType.SP3D2)]
)
_FEATURE_TO_COUNTS = {
    "sp carbons bonded to 1 carbon": (1, _SP),
    "sp carbons bonded to 2 carbons": (2, _SP),
    "sp2 carbons bonded to 1 carbon": (1, _SP2),
    "sp2 carbons bonded to 2 carbons": (2, _SP2),
    "sp2 carbons bonded to 3 carbons": (3, _SP2),
    "sp3 carbons bonded to 1 carbon": (1, _SP3),
    "sp3 carbons bonded to 2 carbons": (2, _SP3),
    "sp3 carbons bonded to 3 carbons": (3, _SP3),
    "sp3 carbons bonded to 4 carbons": (4, _SP3),
}

FEATURE_NAMES = [*_FEATURE_TO_COUNTS, "HybRatio", "Fsp3"]


def calc(props: AtomicProperties, mol_hydrogens: Mol) -> np.ndarray:
    """
    Count carbon atoms by carbon-neighbor degree and hybridization.

    Mordred counts these on the kekulized molecule, which has the same atoms,
    neighbors and hybridizations as the hydrogen-suppressed one ``props`` holds,
    since kekulization only changes bond types.

    ``Fsp3`` is RDKit's fraction of carbons with four neighbors, hydrogens
    included, on the hydrogen-explicit molecule, as in Mordred. That differs from
    the share of SP3-hybridized carbons for charged carbons, e.g. ``[CH2-]``.
    """
    is_carbon = props.is_carbon
    hybridizations = atoms_apply_func(Atom.GetHybridization, props.mol, np.intp)
    hybridizations[np.isin(hybridizations, _SP3_HYBRIDIZATIONS)] = _SP3
    hybridizations = hybridizations[is_carbon]
    carbon_neighbors = props.count_neighbors(is_carbon)[is_carbon]

    values: list[float] = [
        np.count_nonzero((carbon_neighbors == num_neighbors) & (hybridizations == hyb))
        for num_neighbors, hyb in _FEATURE_TO_COUNTS.values()
    ]

    # HybRatio uses aggregate hybridization counts, not neighbor counts
    num_sp3 = np.count_nonzero(hybridizations == _SP3)
    num_sp2 = np.count_nonzero(hybridizations == _SP2)
    values.append(
        np.nan if num_sp2 == 0 and num_sp3 == 0 else num_sp3 / (num_sp2 + num_sp3)
    )
    values.append(CalcFractionCSP3(mol_hydrogens))

    return np.asarray(values, dtype=np.float32)
