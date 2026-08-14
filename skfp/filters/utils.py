import functools
from collections.abc import Iterable

from rdkit.Chem import Atom, Mol, MolFromSmarts, rdMolDescriptors


def get_num_carbon_atoms(mol: Mol) -> int:
    """
    Calculate the number of carbon atoms in a molecule.
    """
    return sum(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def get_num_heavy_metals(mol: Mol) -> int:
    """
    Calculate the number of heavy metals in a molecule.

    Heavy metals are defined as metals other than ["Li", "Be", "K", "Na", "Ca", "Mg"].
    """
    # non-metals and non-heavy metals
    # fmt: off
    not_heavy_metals = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Si", "P",
        "S", "Cl", "Ar", "K", "Ca", "As", "Se", "Br", "Kr", "Te", "I", "Xe", "At", "Rn"
    }
    # fmt: on

    # easier to reject than to list all heavy metals
    return sum(atom.GetSymbol() not in not_heavy_metals for atom in mol.GetAtoms())


def get_num_charged_atoms(mol: Mol) -> int:
    """
    Calculate the number of charged atoms in a molecule.
    """
    return sum(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms())


def get_num_rigid_bonds(mol: Mol) -> int:
    """
    Calculate the number of rigid bonds in a molecule.
    """
    total_bonds = mol.GetNumBonds()
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return total_bonds - rotatable_bonds


def get_num_aromatic_rings(mol: Mol) -> int:
    """
    Calculate the number of aromatic rings in a molecule.
    """
    return sum(is_ring_aromatic(mol, ring) for ring in mol.GetRingInfo().AtomRings())


def get_max_num_fused_aromatic_rings(mol: Mol) -> int:
    """
    Calculate the number of rings in the largest system of fused
    aromatic rings in a molecule.
    """
    ring_info = mol.GetRingInfo()

    fused_aromatic_ring_idxs = []
    for idx, ring in enumerate(mol.GetRingInfo().AtomRings()):
        if is_ring_aromatic(mol, ring) and ring_info.IsRingFused(idx):
            fused_aromatic_ring_idxs.append(idx)

    # find the largest fused ring system by exhaustively checking
    # all combinations

    largest_size = 0
    for i, ring_idx in enumerate(fused_aromatic_ring_idxs):
        fused_rings = [
            ring_idx_2
            for ring_idx_2 in fused_aromatic_ring_idxs[i + 1 :]
            if ring_info.AreRingsFused(ring_idx, ring_idx_2)
        ]
        if fused_rings:
            # original ring and fused ones
            largest_size = max(largest_size, 1 + len(fused_rings))

    return largest_size


def is_ring_aromatic(mol: Mol, ring_atoms: Iterable[Atom]) -> bool:
    """
    Check whether a ring is aromatic.
    """
    return all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atoms)


def get_max_ring_size(mol: Mol) -> int:
    """
    Calculate maximum ring size in a molecule.
    """
    rings = mol.GetRingInfo().AtomRings()
    return max(len(ring) for ring in rings) if rings else 0


def get_non_carbon_to_carbon_ratio(mol: Mol) -> float:
    """
    Calculate ratio of non-carbon to carbon atoms.
    """
    num_carbons = get_num_carbon_atoms(mol)
    num_non_carbons = mol.GetNumAtoms() - num_carbons
    return num_non_carbons / num_carbons if num_carbons > 0 else 0.0


def get_num_charged_functional_groups(mol: Mol) -> int:
    """
    Calculate the number of charged functional groups in a molecule, i.e. having
    a non-zero total charge.

    Since determining what is a functional group is an arguable topic with no
    definite answers, we use a list of CXSMARTS by ChemAxon:
    https://docs.chemaxon.com/display/docs/attachments/attachments_1829721_1_functionalgroups.cxsmi.

    Phosphine and sulfoxide patterns could not be parsed by RDKit, so we manually fixed
    them.
    """
    atomic_charges = {atom.GetIdx(): atom.GetFormalCharge() for atom in mol.GetAtoms()}
    fragment_smarts = _get_functional_groups_smarts_patterns()

    matched_groups: list[tuple[int]] = []
    for smarts in fragment_smarts:
        # note that this returns lists of atom indexes
        matches = mol.GetSubstructMatches(smarts)
        matches = list(matches) if matches else []
        matched_groups.extend(matches)

    num_charged_groups = sum(
        sum(atomic_charges[atom_idx] for atom_idx in group) != 0
        for group in matched_groups
    )

    return num_charged_groups


@functools.cache
def _get_functional_groups_smarts_patterns() -> list[Mol]:
    # using cached function compiles SMARTS patterns only once, on first usage
    fragment_smarts = [
        "[O&X2:2][C&X4&!$(CO~[!#6]):1][O&X2:3]",  # acetal
        "[C&X3:1](=[O&X1:2])[F,Cl,Br,I:3]",  # acid halide
        "[C:1][O&H1&!$(OC~[!#1&!#6]):2]",  # alcohol
        "[C&X3&H1,H2:1]=[O&X1&!$(O=C~[!#1&!#6]):2]",  # aldehyde
        "[C&H1:1]=[N:2][O&H1:3]",  # aldoxime
        "[A:1]",  # aliphatic
        "[C&X4:1][N&X3&!$(NC~[!#1&!#6&!F&!Cl&!Br&!I])&!$(N~[!#1&!#6])&!$(Nc):2]",  # aliphatic amine
        "[!#1&!#6&A:1]",  # aliphatic hetero
        "[#3,#11,#19,#37,#55,#87:1]",  # alkali metal
        "[#4,#12,#20,#38,#56,#88:1]",  # alkaline earth metal
        "[C:1]=[C:1]",  # alkene
        "[C&X2:1](=[C:2])=[C:2]",  # allene
        "[C&H2:3]=[C&H1:2][C&H2:1]",  # allyl
        "[C:1][O&-&!$(OC~[!#1&!#6]):2]",  # alkoxide
        "[C:1][O&X2&H0&!$(OC~[!#1&!#6]):2]",  # alkoxy
        "[C&X2:1]#[C&X2:1]",  # alkyne
        "[C&X4:1][F,Cl,Br,I;!$([F,Cl,Br,I]C~[!#1&!#6]):2]",  # alkyl halide
        "[N&X3:2][C&X3:1]=[O&X1:3]",  # amide
        "[N&X3:3][C:1]=[N&X2:2]",  # amidine
        "[#6:1][N&X3&!$(NC~[!#1&!#6&!F&!Cl&!Br&!I])&!$(N~[!#1&!#6]):2]",  # amine
        "[#6:1][N&X4&+:2]",  # ammonium
        "[C&X3:1](=[O&X1:2])[O&X2:3][C&X3:1]=[O&X1:2]",  # anhydride
        "[c:1][N&X3&!$(NC~[!#1&!#6])&!$(N~[!#1&!#6]):2]",  # aniline
        "[*:1]",  # any atom
        "[a:1]",  # aromatic
        "[!#1&!#6&a:1]",  # aromatic hetero
        "[c&X3:1][F,Cl,Br,I:2]",  # aryl halide
        "[C:1]1[C:1][N:2]1",  # aziridine
        "[#6:2]/[N&X2:1]=[N&X2:1]/[#6:2]",  # azo
        "[N&X2:2]=[N&X2:3][S&X4:1](=[O:5])=[O:4]",  # azosulfone
        "[#6:1][$([N&X2:2]=[N&X3&+:3]([O&-:4])[#6:5]),$([N&X2:2]=[N&X3&+0:3](=[O:4])[#6:5])]",  # azoxy
        "[C&H2]-c1[c&H1][c&H1][c&H1][c&H1][c&H1]1",  # benzyl
        "[Br&-:1]",  # bromide
        "[N&X3,N&X4&+:2][C&X3:1](=[O&X1:3])[O&X2,O&X1&-:4]",  # carbamate
        "[C:1](=[O&X1:2])([O:3][#6:4])[O:3][#6:4]",  # carbonate
        "[C&X3:1]=[O&X1:2]",  # oxo
        "[C&X3:1](=[O:2])[O&X2&H1:3]",  # carboxylic acid
        "[C&X3:1](=[O:2])[O&-:3]",  # carboxylate
        "[R0:1]",  # chain
        "[Cl&-:1]",  # chloride
        # CoA
        "[#1]OC([#1])(C(=O)N([#1])C([#1])([#1])C([#1])([#1])C(=O)N([#1])C([#1])([#1])[C:2]([#1])([#1])[S:1])C(C([#1])([#1])[#1])(C([#1])([#1])[#1])C([#1])([#1])OP(=O)(O[#1])OP(=O)(O[#1])OC([#1])([#1])C1([#1])OC([#1])(n2c([#1])nc3c(nc([#1])nc23)N([#1])[#1])C([#1])(O[#1])C1([#1])OP(=O)(O[#1])O[#1]",
        "[N&X3:3][C&X2:1]#[N&X1:2]",  # cyanamide
        "[N&X3:3][C&X2:1]#[N&X1:2]",  # cyanimide
        "[$([#6:1]=[N&+:2]=[N&-:3]),$([#6&-:1]-[N&+:2]#[N:3])]",  # diazo
        "[#6&+0:1]-[N&+:2]#[N:3]",  # diazonium
        "[#6:1][S&X2&H0:2][S&X2&H0:2][#6:1]",  # disulfide
        "[N&X3&!$(NC~[!#1&!#6]):3][C&X3:1]=[C&X3:2]",  # enamine
        "[O&X2&H1:3][C&X3:2]=[C:1]",  # enol
        "[C:1]1[C:1][O:2]1",  # epoxide
        "[#6:5][O&X2:2][C&X3:1](=[O:3])[#6:4]",  # ester
        "[O&D2&!$(OC~[!#1&!#6]):2]([#6:1])[#6:1]",  # ether
        "[F&-:1]",  # fluoride
        "[F,Cl,Br,I:1]",  # halogen
        "[C:4][O&X2:3][C&H1:1]([#6:2])[O&H1&X2:5]",  # hemiacetal
        "[!#1&!#6:1]",  # hetero
        "[#6:1][N&X3:2][N&X3:3]",  # hydrazine
        "[N&X3:3][N&X2:2]=[C:1]",  # hydrazone
        "[#6:1][O&H1&!$(O[#6]~[!#1&!#6]):2]",  # hydroxyl
        "[N&X2:2][O&H1:3]",  # hydroxylamine
        "[C&X3:1](=[O&X1:3])[N&X3:2][C&X3:1]=[O&X1:3]",  # imide
        "[C&X3:1]=[N&X2&!$(N~C~[!#1&!#6]):3][#6,#1:2]",  # imine
        "[N&X3&+:2]=[C&X3:1]",  # iminium
        "[I&-:1]",  # iodide
        "[N&X2:2]=[C:1]=[O&X1:3]",  # isocyanate
        "[N&+:2]#[C&-:1]",  # isocyanide
        "[N&+:2]#[C&-:1]",  # isonitrile
        "[N&X2:2]=[C:1]=[S&X1:3]",  # isothiocyanate
        "[O&X2:2][C&H0&X4&!$(CO~[!#6]):1][O&X2:3]",  # ketal
        "[C:1]=[C:2]=[O&X1:3]",  # ketene
        "[#6:2][C&X3:1](=[O:3])[#6:2]",  # ketone
        "[C:1](=[N:2][O&H1:3])([#6:4])[#6:4]",  # ketoxime
        # metal
        "[!#1&!#2&!#5&!#6&!#7&!#8&!#9&!#10&!#14&!#15&!#16&!#17&!#18&!#32&!#33&!#34&!#35&!#36&!#51&!#52&!#53&!#54&!#85&!#86:1]",
        "[B,#14,#32,#33,#51,#52,#85:1]",  # metalloid
        "[#7:1][O&X1&!$(O[#7]~[!#1&!#6]):2]",  # N oxide
        "[N&X1:2]#[C&X2:1]",  # nitrile
        "[#6:1][$([N&X3:2](=[O:3])=[O:3]),$([N&X3&+:1](=[O:3])[O&-:3])]",  # nitro
        "[N&X2:1]=[O&X1:2]",  # nitroso
        "[#1,#2,#6,#7,#8,F,#10,#15,#16,Cl,#18,#34,Br,#36,I,#54,#86:1]",  # nonmetal
        "[C:1]=[N:2][O&H1:3]",  # oxime
        "[O&X2,O&X1&-][O&X2,O&X1&-]",  # peroxide
        "[c:1][O&X2&H1:2]",  # phenol
        "[c:1][O&X1&-:2]",  # phenoxide
        "[c:1][O&X2&H0&!$(OC~[!#1&!#6]):2]",  # phenoxy
        "[c&H1]1[c&H1][c&H1][c&H1][c&H1]c1",  # phenyl
        "[PX3;$([H3]),$([H2][#6]),$([H1]([#6])[#6]),$([H0]([#6])([#6])[#6])]",  # phosphine
        "[N&X3&H2:2][C&X3:1]=[O&X1:3]",  # primary amide
        "[#6:1][#7&A&H2&X3&!$(NC~[!#1&!#6&!F&!Cl&!Br&!I])&!$(N~[!#1&!#6]):2]",  # primary amine
        "[H1,H2,H3,H4&+:1]",  # protonated
        "[#6&X4:1][N&X4&H0&!$(NC~[!#1&!#6]):2]",  # quaternary amino
        "[R:1]",  # ring
        "[N&X3&H1:2][C&X3:1]=[O&X1:3]",  # secondary amide
        "[#6:1][N&X3&H1&!$(NC~[!#1&!#6&!F&!Cl&!Br&!I])&!$(N~[!#1&!#6]):2]",  # secondary amine
        "[#6:1][S&X2&H0&!$(SC~[!#1&!#6]):2][#6:1]",  # sulfide
        # sulfonamide
        "[#6:1][$([S&X4:2]([N&X3:3])(=[O&X1:4])=[O&X1:4]),$([S&X4&+2:2]([N&X3:3])([O&X1&-:4])[O&X1&-:4])]",
        # sulfone
        "[$([S&X4:2](=[O&X1:3])(=[O&X1:3])([#6:1])[#6:1]),$([S&X4&+2:2]([O&X1&-:3])([O&X1&-:3])([#6:1])[#6:1])]",
        # sulfonic acid
        "[#6:1][$([S&X4:2](=[O&X1:4])(=[O&X1:4])[O&X2&H1,O&X1&H0&-:3]),$([S&X4&+2:2]([O&X1&-:4])([O&X1&-:4])[O&X2&H1,O&X1&H0&-:3])]",
        "[$([SX3](=[OX1])([#6])[#6]),$([SX3+]([OX1-])([#6])[#6])]",  # sulfoxide
        "[N&X3&H0:2][C&X3:1]=[O&X1:3]",  # tertiary amide
        "[#6:1][N&X3&H0&!$(NC~[!#1&!#6&!F&!Cl&!Br&!I]):2]",  # tertiary amine
        "[#6:1][N&X3&H0&!$(NC~[!#1&!#6&!F&!Cl&!Br&!I]):2]",  # tertiary amino
        "[N&X3:2][C&X3:1]=[S&X1:3]",  # thioamide
        "[C&X3:1](=[S:2])[S&X2&H1:3]",  # thiocarboxide
        "[C&X3:1](=[S:2])[S&-:3]",  # thiocarboxylate
        "[#6:5][S&X2:2][C&X3:1](=[O:3])[#6:4]",  # thioester
        "[#6:1][S&H1&!$(S[#6]~[!#1&!#6]):2]",  # thiol
        "[N&X3:2][C&X3:1](=[S&X1:3])[N&X3:2]",  # thiourea
        "[N&X3:2][C&X3:1](=[O&X1:3])[N&X3:2]",  # urea
        "[C&X3&H2:1]=[C&X3&H1:1]",  # vinyl
    ]
    fragment_smarts = [MolFromSmarts(smarts) for smarts in fragment_smarts]
    return fragment_smarts
