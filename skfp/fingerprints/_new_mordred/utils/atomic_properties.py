import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond

from .periodic_table import (
    ALLRED_ROCOW_EN,
    IONIZATION_POTENTIAL,
    MC_GOWAN_VOLUME,
    PAULING_EN,
    PERIOD,
    POLARIZABILITY_94,
    SANDERSON_EN,
    mass,
    vdw_volume,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


_TABLE = Chem.GetPeriodicTable()


def get_element_symbol(atomic_num: int) -> str:
    return _TABLE.GetElementSymbol(atomic_num)


def get_atomic_number_from_symbol(symbol: str) -> int:
    return _TABLE.GetAtomicNumber(symbol)


def get_mass(atom: Atom) -> float:
    return mass(atom.GetAtomicNum())


def get_vdw_volume(atom: Atom) -> float:
    return vdw_volume(atom.GetAtomicNum())


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
    """
    Return the number of sigma (single-bond framework) electrons on an atom,
    approximated as the count of its non-hydrogen neighbors.
    """
    return sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() != 1)


def get_valence_electrons(atom: Atom) -> float:
    """
    Valence delta-value used in molecular connectivity indices.

    Based on Kier, L. B., & Hall, L. H. (1983). General definition of
    valence delta-values for molecular connectivity. Journal of
    Pharmaceutical Sciences, 72(10), 1170-1173.
    https://doi.org/10.1002/jps.2600721016
    """
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
    """
    Intrinsic state value used in electrotopological-state (E-state) indices.

    See the Molconn-Z 4.00 manual, chapter 2, p. 283:
    http://www.edusoft-lc.com/molconn/manuals/400/chaptwo.html.
    """
    d = get_sigma_electrons(atom)
    if d == 0:
        return np.nan
    dv = get_valence_electrons(atom)
    return ((2.0 / PERIOD[atom.GetAtomicNum()]) ** 2 * dv + 1) / d


def get_core_count(atom: Atom) -> float:
    """
    Atomic core-count term (alpha) used as a building block of ETA indices.
    Reflects the relative number of non-valence (core) electrons, scaled by period.
    """
    Z = atom.GetAtomicNum()
    if Z == 1:
        return 0.0
    Zv = _TABLE.GetNOuterElecs(Z)
    PN = PERIOD[Z]
    return (Z - Zv) / (Zv * (PN - 1))


def get_eta_epsilon(atom: Atom) -> float:
    """
    ETA electronegativity-like measure (epsilon) for a single atom.
    Differences in epsilon between bonded atoms encode bond polarity.
    """
    Zv = _TABLE.GetNOuterElecs(atom.GetAtomicNum())
    return 0.3 * Zv - get_core_count(atom)


def get_eta_beta_sigma(atom: Atom) -> float:
    """
    Sigma-bond contribution to an atom's ETA beta index, summed over
    non-hydrogen neighbors and weighted by similarity of their epsilon values.
    """
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
    """
    Non-sigma (pi, aromatic) contribution of a single bond to the ETA beta index.
    Weighted by bond order, aromaticity, and the epsilon difference of its endpoints.
    """
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
    """
    Lone-pair (delta) contribution to an atom's ETA beta index.
    Nonzero only for acyclic atoms with lone pairs adjacent to an aromatic neighbor.
    """
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
    """
    Total non-sigma (pi, aromatic) bond contribution to an atom's ETA beta index,
    summed over all bonds to non-hydrogen neighbors.
    """
    return sum(
        get_eta_nonsigma_contribute(b)
        for b in atom.GetBonds()
        if _get_other_atom(b, atom).GetAtomicNum() != 1
    )


def get_eta_gamma(atom: Atom) -> float:
    """
    ETA gamma index for an atom: core count divided by total beta contribution.
    Represents an atom's topochemical "hardness" in the ETA framework.
    """
    beta = (
        get_eta_beta_sigma(atom)
        + get_eta_beta_non_sigma(atom)
        + get_eta_beta_delta(atom)
    )
    if beta == 0:
        return np.nan
    return get_core_count(atom) / beta
