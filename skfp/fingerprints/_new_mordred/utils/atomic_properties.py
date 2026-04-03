import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond

from .periodic_table import (
    ALLRED_ROCOW_EN,
    IONIZATION_POTENTIAL,
    MASS,
    MC_GOWAN_VOLUME,
    PAULING_EN,
    PERIOD,
    POLARIZABILITY_94,
    SANDERSON_EN,
    VDW_VOLUME,
)

_TABLE = Chem.GetPeriodicTable()


def get_element_symbol(atomic_num: int) -> str:
    return _TABLE.GetElementSymbol(atomic_num)


def get_atomic_number_from_symbol(symbol: str) -> int:
    return _TABLE.GetAtomicNumber(symbol)


def get_mass(atom: Atom) -> float:
    return MASS[atom.GetAtomicNum()]


def get_vdw_volume(atom: Atom) -> float:
    return VDW_VOLUME[atom.GetAtomicNum()]


def get_sanderson_en(atom: Atom) -> float:
    return SANDERSON_EN[atom.GetAtomicNum()]


def get_pauling_en(atom: Atom) -> float:
    return PAULING_EN[atom.GetAtomicNum()]


def get_allred_rocow_en(atom: Atom) -> float:
    return ALLRED_ROCOW_EN[atom.GetAtomicNum()]


def get_polarizability(atom: Atom) -> float:
    return POLARIZABILITY_94[atom.GetAtomicNum()]


def get_ionization_potential(atom: Atom) -> float:
    return IONIZATION_POTENTIAL[atom.GetAtomicNum()]


def get_mc_gowan_volume(atom: Atom) -> float:
    return MC_GOWAN_VOLUME[atom.GetAtomicNum()]


def get_gasteiger_charge(atom: Atom) -> float:
    return (
        atom.GetDoubleProp("_GasteigerCharge") + atom.GetDoubleProp("_GasteigerHCharge")
        if atom.HasProp("_GasteigerHCharge")
        else 0.0
    )


def get_sigma_electrons(atom: Atom) -> int:
    return sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() != 1)


def get_valence_electrons(atom: Atom) -> float:
    """http://dx.doi.org/10.1002%2Fjps.2600721016"""
    N = atom.GetAtomicNum()
    if N == 1:
        return 0.0
    Zv = _TABLE.GetNOuterElecs(N) - atom.GetFormalCharge()
    Z = N - atom.GetFormalCharge()
    h = atom.GetTotalNumHs() + sum(
        1 for a in atom.GetNeighbors() if a.GetAtomicNum() == 1
    )
    return (Zv - h) / (Z - Zv - 1)


def get_intrinsic_state(atom: Atom) -> float:
    """http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html p. 283"""
    d = get_sigma_electrons(atom)
    if d == 0:
        return np.nan
    dv = get_valence_electrons(atom)
    return ((2.0 / PERIOD[atom.GetAtomicNum()]) ** 2 * dv + 1) / d


def get_core_count(atom: Atom) -> float:
    Z = atom.GetAtomicNum()
    if Z == 1:
        return 0.0
    Zv = _TABLE.GetNOuterElecs(Z)
    PN = PERIOD[Z]
    return (Z - Zv) / (Zv * (PN - 1))


def get_eta_epsilon(atom: Atom) -> float:
    Zv = _TABLE.GetNOuterElecs(atom.GetAtomicNum())
    return 0.3 * Zv - get_core_count(atom)


def get_eta_beta_sigma(atom: Atom) -> float:
    e = get_eta_epsilon(atom)
    return sum(
        0.5 if abs(get_eta_epsilon(a) - e) <= 0.3 else 0.75
        for a in atom.GetNeighbors()
        if a.GetAtomicNum() != 1
    )


def _get_other_atom(bond: Bond, atom: Atom) -> Atom:
    begin = bond.GetBeginAtom()
    if atom.GetIdx() != begin.GetIdx():
        return begin
    return bond.GetEndAtom()


def get_eta_nonsigma_contribute(bond: Bond) -> float:
    if bond.GetBondType() is Chem.BondType.SINGLE:
        return 0.0

    f = 1.0
    if bond.GetBondTypeAsDouble() == Chem.BondType.TRIPLE:
        f = 2.0

    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()
    dEps = abs(get_eta_epsilon(a) - get_eta_epsilon(b))

    if bond.GetIsAromatic():
        y = 2.0
    elif dEps > 0.3:
        y = 1.5
    else:
        y = 1.0

    return y * f


def get_eta_beta_delta(atom: Atom) -> float:
    if (
        atom.GetIsAromatic()
        or atom.IsInRing()
        or _TABLE.GetNOuterElecs(atom.GetAtomicNum()) - atom.GetTotalValence() <= 0
    ):
        return 0.0

    for b in atom.GetNeighbors():
        if b.GetIsAromatic():
            return 0.5

    return 0.0


def get_eta_beta_non_sigma(atom: Atom) -> float:
    return sum(
        get_eta_nonsigma_contribute(b)
        for b in atom.GetBonds()
        if _get_other_atom(b, atom).GetAtomicNum() != 1
    )


def get_eta_gamma(atom: Atom) -> float:
    beta = (
        get_eta_beta_sigma(atom)
        + get_eta_beta_non_sigma(atom)
        + get_eta_beta_delta(atom)
    )
    if beta == 0:
        return np.nan
    return get_core_count(atom) / beta
