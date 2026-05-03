import numpy as np
from rdkit.Chem import Mol, MolFromSmarts

from skfp.bases.base_filter import BaseFilter


class NIBRFilter(BaseFilter):
    """
    NIBR filter.

    Designed by Novartis Institutes for BioMedical Research (NIBR) for building
    screening decks following their diversity-driven subset screening experiments [1]_.
    Primarily targeted toward hit finding, i.e. keeping lead-like molecules.

    Rule definitions from the original publication have been published as part of
    RDKit [2]_. They are divided by severity into "exclude", "flag" and "annotation".
    Filter removes molecules which fulfill any "exclude" criterion, and those with
    at lest 10 flags (this can be controlled with ``severity`` parameter). "annotation"
    rules are not used. Note that some rules may need to be fulfilled many times to
    fulfill the condition.

    Note that this filter can be considerably slower than others due to a large
    number of rules checked with SMARTS patterns.

    Parameter
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the "exclude" rules for a molecule, which
        makes the filter less restrictive.

    severity : int, default=10
        Molecules with at least this much total severity from "flag" rules are removed.
        Lower value makes the filter more restrictive.

    return_type : {"mol", "indicators", "condition_indicators"}, default="mol"
        What values to return as the filtering result.

        - ``"mol"`` - return a list of molecules remaining in the dataset after filtering
        - ``"indicators"`` - return a binary vector with indicators which molecules pass
          the filter (1) and which would be removed (0)
        - ``"condition_indicators"`` - return a Pandas DataFrame with molecules in rows,
          filter conditions in columns, and 0/1 indicators whether a given condition was
          fulfilled by a given molecule

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `Ansgar Schuffenhauer et al.
        "Evolution of Novartis’ Small Molecule Screening Deck Design"
        J. Med. Chem. 2020, 63, 23, 14425-14447
        <https://doi.org/10.1021/acs.jmedchem.0c01332>`_

    .. [2] `RDKit NIBR filter definitions
        <https://github.com/rdkit/rdkit/blob/0de215a1f8baf3dc0ac48597dcd68299498d0526/Contrib/NIBRSubstructureFilters/SubstructureFilter_HitTriaging_wPubChemExamples.csv>`_

    Examples
    --------
    >>> from skfp.filters import NIBRFilter
    >>> smiles = ["C", "C1=CC2(C=CC1=O)C=C2	"]
    >>> filt = NIBRFilter()
    >>> filt
    NIBRFilter()

    >>> filtered_mols = filt.transform(smiles)
    >>> filtered_mols
    ['C']
    """

    def __init__(
        self,
        allow_one_violation: bool = False,
        severity: int = 10,
        return_type: str = "mol",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int = 0,
    ):
        filters, condition_names = self._load_filters()
        super().__init__(
            condition_names=condition_names,
            allow_one_violation=allow_one_violation,
            return_type=return_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.severity = severity
        self._filters = filters

    def _apply_mol_filter(self, mol: Mol) -> bool | np.ndarray:
        # note that this is rejection filter, trying to return False and remove
        # molecule as fast as possible

        # add condition indicators
        condition_indicators = np.zeros(len(self.condition_names), dtype=bool)
        exclusions = 0
        flags_counter = 0
        for idx, (smarts, min_count, exclude) in enumerate(self._filters):
            num_matches = len(mol.GetSubstructMatches(smarts, maxMatches=min_count))
            if num_matches < min_count:
                continue
            elif exclude:
                exclusions += 1
                condition_indicators[idx] = True
                if (
                    exclusions >= self.allow_one_violation
                    and self.return_type != "condition_indicators"
                ):
                    return False
            else:  # flag
                flags_counter += 1
                condition_indicators[idx] = True
                if (
                    flags_counter >= self.severity
                    and self.return_type != "condition_indicators"
                ):
                    return False

        if self.return_type == "condition_indicators":
            return condition_indicators
        else:
            return True

    def _load_filters(self) -> tuple[list[tuple[Mol, int, bool]], list[str]]:
        # SMARTS, minimal count, exclude (otherwise flag)
        filters = [
            ("S=S", 1, True),
            ("P~P", 1, True),
            ("Cl~O", 1, True),
            ("[S-]", 1, True),
            ("[3#1]", 1, True),
            ("C=C=O", 1, True),
            ("N=C=N", 1, True),
            ("[#15]", 2, True),
            ("[16#7]", 1, True),
            ("[17#8]", 1, True),
            ("[15#7]", 1, True),
            ("[14#6]", 1, True),
            ("[18#8]", 1, True),
            ("[13#6]", 1, True),
            ("[35#16]", 1, True),
            ("A-O-O-A", 1, True),
            ("[Si]~N ", 1, True),
            ("[125#53]", 1, True),
            ("O=S(*)OF", 1, True),
            ("COC(=S)C", 1, True),
            ("N#CccC#N", 1, True),
            ("O=C-N=!@N", 1, True),
            ("C1NNC=NN1", 1, True),
            ("N=NC(=S)N", 1, True),
            ("P(=S)(S)S", 1, True),
            ("A-O-[O;H1]", 1, True),
            ("S1C=CSC1=S", 1, True),
            ("S1SC=CC1=S", 1, True),
            ("NP(=O)(N)N", 1, True),
            ("*1:,=*~*~1", 1, True),
            ("A-[S;X2]-O", 1, True),
            ("SC(=[!r])S", 1, True),
            ("[SX2H0][N]", 1, True),
            ("S(O)(O)(O)", 1, True),
            ("C1~[O,S]~N1", 1, True),
            ("S(~O)(~O)~O", 1, True),
            ("SC(=O)[O,S]", 1, True),
            ("[#16]~[#16]", 1, True),
            ("C#[N+]-[O-]", 1, True),
            ("OCl(O)(O)(O)", 1, True),
            ("O1C=CC=CC=C1", 1, True),
            ("O=COn1cncc1 ", 1, True),
            ("S(=O)(=O)C#N", 1, True),
            ("[C;$(C(F)F)]", 5, True),
            ("C[N+](=O)[O-]", 1, True),
            ("N=[N+]([O-])C", 1, True),
            ("C(=O)C[N+,n+]", 1, True),
            ("C#C-[Cl,Br,I]", 1, True),
            ("C=C-C=!@C-C=C", 1, True),
            ("[N;R0](~N)~O ", 1, True),
            ("P(=O)(a)(a)(a)", 1, True),
            ("[#6]-C(=O)[OH]", 3, True),
            ("[#7]O[#6,#16]=O", 2, True),
            ("*~[F,Cl,Br,I]~*", 1, True),
            ("[#16]-[#6]#[#7]", 1, True),
            ("C#N.C#N.C#N.C#N", 1, True),
            ("C1(=O)C=CCSC=C1", 1, True),
            ("[#6]-[S;X2]-[#6]", 2, True),
            ("C1C(=O)C=CC=CC=1", 1, True),
            ("C(F)(F)(F)C(=O)S", 1, True),
            ("[!#6]1~[!#6]~*~1", 1, True),
            ("[N;R0]=[N;R0]C#N", 1, True),
            ("C[Mg][F,Cl,Br,I]", 1, True),
            ("[N+]([O-])(=C)-* ", 1, True),
            ("[C,c][S;X3](~O)-S", 1, True),
            ("[#7]-[C;H2]-[O;H]", 1, True),
            ("[NH2]-c1sccc1-C=O", 1, True),
            ("[O,S]=C-[F,Cl,Br,I]", 1, True),
            ("s1ccc(c1)-[#8]-[#1]", 1, True),
            ("c12ccccc1(SC(=S)N2)", 1, True),
            ("c12ccccc1(SC(S)=N2)", 1, True),
            ("[#8]=[#15]-[#6]#[#7]", 1, True),
            ("[#6]~[#7;X2]~[#7;X1]", 1, True),
            ("c1(c)c2c(c)cccc2ccc1", 1, True),
            ("[N;!R]-[N;!R]-[N;!R]", 1, True),
            ("c12ccccc1cc3ccccc3c2", 1, True),
            ("S!@C(=!@[O,S])!@[#7]", 1, True),
            ("[#6;-1;X1]#[#7;+1;X2]", 1, True),
            ("[#6]-[O]-!@C(=O)-[#6]", 5, True),
            ("c12ccccc1ccc3ccccc23 ", 1, True),
            ("[!#6]=[*]1-[!#6]*[!#6]1", 1, True),
            ("[#8;X1]~[#16]-[#6]#[#7]", 1, True),
            ("[#6,#8,#16]-[CH1]=[NH1]", 1, True),
            ("O1CCCCC1OC2CCC3CCCCC3C2", 1, True),
            ("NC(C(C)OP(O)(O)=O)C(O)=O", 1, True),
            ("[OH]-c1cccc2ccc[n;X2]c12", 1, True),
            ("[#7]-N=!@C-c1ccccc1-[OH]", 1, True),
            ("O=C(N1)[C;H1]=[C;H1]C1=O", 1, True),
            ("[#6;!$([#6]=[!#6])]-[OH]", 6, True),
            ("[CX4]O-S(=O)(=O)-C(F)(F)F", 1, True),
            ("[O;$(O1OCOC1),$(O1OOCC1)]", 1, True),
            ("[#6;-1;!$([#6;-1]~[*;+1])]", 1, True),
            ("[S;D3](-N)(-[c,C])(-[c,C])", 1, True),
            ("[#7]-N=!@C-c1ccc(-[OH])cc1", 1, True),
            ("S(=O)(=O)-[N;R0]-S(=O)(=O)", 1, True),
            ("[#6]C(=O)!@OC(=!@[N,O])[#6]", 1, True),
            ("[!#6]-[CH2]-N1C(=O)CCC(=O)1", 1, True),
            ("[#6]-[N;X3](~[O;X1])~[O;X1]", 3, True),
            ("[#8]-[#7](~[#8;X1])~[#8;X1]", 1, True),
            ("C(=O)Oc1ccc([N+](=O)[O-])cc1", 1, True),
            ("C1~[O,S]~[C,N,O,S]1[a,N,O,S]", 1, True),
            ("[#7;X2;!$([#7]-[#7v3])]=[#8]", 1, True),
            ("O=C1-[#6]~[#6]-C(=O)N1-O-C=O", 1, True),
            ("[#16&v2,#15]-!@[#7&X3,#15&X3]", 1, True),
            ("C1(C)(C)SC(C(=O)O)=C(C(=O)O)S1", 1, True),
            ("[c;$(c1ccccc1),$(c1ccac1)][OH]", 3, True),
            ("NC(C[S;D1])C([N;H1]([O;D1]))=O", 1, True),
            ("[CH2]1-[#7&v3&H1,#16&v2]-[C;X4]1", 1, True),
            ("C1(Cl)(Cl)C(Cl)C(Cl)=C(Cl)C1(Cl)", 1, True),
            ("[#6]=,:[#6](C#N)-,:[#6]=,:[#6]C#N", 1, True),
            ("O=[#6]-,:[#8]-,:[#6]=,:[#7&X2,+1]", 1, True),
            ("O=C1-[#6]~[#6]-C(=O)N1-O-c2naaaa2", 1, True),
            ("[#15]1~[#7]~[#15]~[#7]~[#15]~[#7]1", 1, True),
            ("c12c3cccc1ncc2CC4NC[#6]~[#6]-,=C43", 1, True),
            ("C(=O)-[CH;$(C=O),$(C(-[OH])-[OH])]", 1, True),
            ("[OH]-c1ccccc1-[C;X4;R0](-[N;v3])-c", 1, True),
            ("N=C1[#6]:,=[#6]C(=[C,N])[#6]:,=[#6]1", 1, True),
            ("c12cccc(C(=O)N(-&!@C)C(=O)3)c2c3ccc1", 1, True),
            ("O=C(-[!N])O[$(nnn),$([#7]-[#7]=[#7])]", 1, True),
            ("[$(Sc1nnn[nH,n-]1),$(Sc1nn[nH,n-]n1)]", 1, True),
            ("[#7]1~[#7]~[#6]~[#7](-C(=O)[!N])~[#6]1", 1, True),
            ("[$(C(=O)),$(S(=O)(=O))][O,S](S(=O)(=O))", 1, True),
            ("[Cl,Br,I][CX4][CX4][$([O,S,N]*),Cl,Br,I]", 1, True),
            ("c1(-c2ccccc2)c3ccccc3c(-c4ccccc4)c5ccccc51", 1, True),
            ("[#7;R1]1~[#7;R1]~[#7;R1](-C(=O))~[#6]~[#6]1", 1, True),
            ("[#6]-1(=[#6])-[#6](-[#7]=[#6]-[#16]-1)=[#8]", 1, True),
            ("[Cl,Br,I]-[CX4;!$(C([F,Cl,Br,I])[F,Cl,Br,I])]", 2, True),
            ("[#6]-[#6]=[#6](-[F,Cl,Br,I])-[#6](=[#8])-[#6]", 1, True),
            ("[#7]-1-[#6](=[#16])-[#16]-[#6;X4]-[#6]-1=[#8]", 1, True),
            ("O=[C,P!$(P(~O)(~O)(~O)O),N,S]!@O!@[C,P,N,S]=O", 1, True),
            ("[#16]=[#6](-[#16])-[#8,#16;H1,R0,$(*-[C;X4])]", 1, True),
            ("[$(C(=O));!$(C-N);!$(C-O);!$(C-S)]C(Cl)(Cl)(Cl)", 1, True),
            ("[#16]=[#6]-1-[#6]=,:[#6]-[!#6&!#1]-[#6]=,:[#6]-1", 1, True),
            ("c:1:c:2:c(:c:c:c:1):n:c:3:c(:c:2-[#7]):c:c:c:c:3", 1, True),
            ("[#16;H1;X2;!$([#16]-[#6](-,:[#7,#8,#16])=,:[#7])]", 2, True),
            ("[#6;r16,r17,r18]~[#6]1~[#6]~[#6]~[#6](~[#6])~[#7]1", 1, True),
            ("[#7](-[#1])-[#7]=[#6](-[#6]#[#7])-[#6]=[!#6&!#1;!R]", 1, True),
            ("[#6]=[#6]1-[#16&v2]-[C;X4]-[#7&v3]-[C;X4]-[#7&v3]-1", 1, True),
            ("[Cl,Br,I]-[CH2,$([CH](-[Cl,Br,I])-[Cl,Br,I])]-[#16]", 1, True),
            ("P(=S)(-[S;H1,H0$(S(P)C)])(-[O;H1,H0$(O(P)C)])(-N(C)C)", 1, True),
            ("[CH2]-[CH]1-[S&v2]-[CH2]-[CH]2-[NH]-C(=O)-[NH]-[CH]12", 1, True),
            ("[#8,#7&v3,#16&v2]-[CH2]-[CH2]-[CH2]-[#8,#7&v3,#16&v2]", 3, True),
            ("N-!@C(CC1-!@N)C(-!@OC2OCCCC2)C(-!@O)C1-!@OC3CCC-,=CO3", 1, True),
            ("[#6]1~[$(C(=O)),$(S(=O))]~[O,S,N]~[$(C(=O)),$(S(=O))]1", 1, True),
            ("c:1-2:c(:c:c:c:c:1)-[#6](=[#8])-[#6](=[#6])-[#6]-2=[#8]", 1, True),
            ("O=[#6,P!$(P(~O)(~O)(~O)O),#7,#16]@[#8]@[#6,#15,#7,#16]=O", 1, True),
            ("[#8]=[#6]-,:[#6](=[#8])-,:[#7]-,:[#6](=[#8])-,:[#6]=[#8]", 1, True),
            ("[#6]-1(-[#6](=[#8])-[#7]-[#6](=[#8])-[#7]-[#6]-1=[#8])=[#7]", 1, True),
            ("[#6](=O)-O-[#6]=[#6;$([#6]-C#N),$([#6]-[#6]=O),$([#6]-S=O)]", 1, True),
            ("[#6]=!@[#6](-[!#1])-@[#6](=!@[!#6&!#1])-@[#6](=!@[#6])-[!#1]", 1, True),
            ("[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35;!#5;!#14;!#53;!#34]", 1, True),
            ("c:1:c:c-2:c(:c:c:1)-[#6]-3-[#6](-[#6]-[#7]-2)-[#6]-[#6]=[#6]-3", 1, True),
            (
                "[C,S,P](=O)[OH].[C,S,P](=O)[OH].[C,S,P](=O)[OH].[C,S,P](=O)[OH]",
                1,
                True,
            ),
            (
                "[$(N(-C(=O))(-C(=O))(-C(=O))),$(n([#6](=O))([#6](=O))([#6](=O)))]",
                1,
                True,
            ),
            (
                "c(C([C;H3])([C;H3])[C;H3])1c([OH1])c(C([C;H3])([C;H3])[C;H3])ccc1",
                1,
                True,
            ),
            (
                "[c]1:[c]2:[c](-[#6](=[#8])-[#6]=[#6](-[#16])-[#16]2):[c]:[c]:[c]1",
                1,
                True,
            ),
            (
                "[#15!$([#15](~[#8;X1])(-[#7])(-[#8])-[#8])](~[#8;X1])-[#16,#7&H1]",
                1,
                True,
            ),
            (
                "c12ccccc1C=,-C-,=C-,=3-,=C-,=2-,=C-,=C-,=C4-,=C-,=C-,=C-,=C-,=4-,=3",
                1,
                True,
            ),
            (
                "C(=O)Oc1c([Cl,F])[cH1,$(c[F,Cl])]c([F,Cl])[cH1,$(c[F,Cl])]c1([F,Cl])",
                1,
                True,
            ),
            (
                "[Cl,Br,I]-[CH2,$([CH](-[Cl,Br,I])-[Cl,Br,I])]-[#7&X3,#15&X3,#5&X3,#8]",
                1,
                True,
            ),
            (
                "a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a",
                1,
                True,
            ),
            (
                "[C;R0;X4]!@[CH]!@[CH]!@[CH]!@[CH]!@[CH]!@[CH]!@[CH]!@[CH]!@[CH]!@[CH] ",
                1,
                True,
            ),
            (
                "[#6](-c1[cH][cH]c[cH][cH]1)(-c1[cH][cH]c[cH][cH]1)-c1[cH][cH]c[cH][cH]1",
                1,
                True,
            ),
            (
                "c(-[O,N,S;!$(N[C,S]=O);H1,H2;R0])1c(-[O,N,S;!$(N[C,S]=O);H1,H2;R0])cccc1",
                1,
                True,
            ),
            (
                "[#6]1-,:[#6](=[#8,#7,#16])-,:[#6]=,:[#6]-,:[#6](=[#8,#7,#16])-,:[#6]=,:1",
                1,
                True,
            ),
            (
                "c(-[O,N,S;!$(N[C,S]=O);H1,H2;R0])1ccc(-[O,N,S;!$(N[C,S]=O);H1,H2;R0])cc1",
                1,
                True,
            ),
            (
                "[#7;+1;X3;!$([#7;+1](=[#8])-[#8;-1]);!$([#7;+1](#[#6])-[#8;-1])]-[#8;-1]",
                2,
                True,
            ),
            (
                "c1c([F,Br,Cl,I])c([F,Br,Cl,I])c([F,Br,Cl,I])c([F,Br,Cl,I])c1([F,Br,Cl,I])",
                1,
                True,
            ),
            (
                "[!#6;!Cl;!F;!#1]-[#6](-[!#6;!F;!C;!#1])(-[!#6;!F;!Cl;!#1])-[!#6;!F;!C;!#1]",
                1,
                True,
            ),
            (
                "[CH0](=O)[#7;X2;$([#7]=[#7;X2;+1]=[#7;X1;-1]),$([#7;-1]-[#7;X2;+1]#[#7;X1])]",
                1,
                True,
            ),
            (
                "[#7]=,#[#6;$([#6](#[#7+])-[#8-,#16-]),$([#6](=[#7])=[#8,#16])]=,-[#8,#16;X1]",
                1,
                True,
            ),
            (
                "[#8,#7,#15,#16,#14,#5]-[F!$(F-[#14])!$(F-S(=O)(=O)-[#6])!$(F-S(F)(F)),Cl,Br,I]",
                1,
                True,
            ),
            (
                "[#6]1-,:[#6](=;!@[#8,#7,#16])-,:[#6](=;!@[#8,#7,#16])-,:[#6]=,:[#6]-,:[#6]=,:1",
                1,
                True,
            ),
            (
                "[C;X4](-[R0;Cl,Br,I,$(O(S(=O)(=O)[!$(N);!$([O&D1])]))])(-[c,C])(-[c,C])(-[c,C])",
                1,
                True,
            ),
            (
                "[#6]1-,:[#6](=[#6!R!$(C-[O,N,S])])-,:[#6](=[#8,#7,#16])-,:[#6]=,:[#6]-,:[#6]=,:1",
                1,
                True,
            ),
            (
                "[#6]1-,:[#6](=[#6!R!$(C-[O,N,S])])-,:[#6]=,:[#6]-,:[#6](=[#8,#7,#16])-,:[#6]=,:1",
                1,
                True,
            ),
            (
                "[#6]1(-O-[C;!R](=[O,N;!R]))[#6]([$(N(=O)(=O)),$([N+](=O)[O-])])[#6][#6][#6][#6]1",
                1,
                True,
            ),
            (
                "c1c([O;D1])c(-[Cl,Br,I])cc(-[Cl,Br,I])c1.c1c([O;D1])c(-[Cl,Br,I])cc(-[Cl,Br,I])c1",
                1,
                True,
            ),
            (
                "c(-[Cl,Br,I])1c([O;D1])c(-[Cl,Br,I])ccc1.c(-[Cl,Br,I])1c([O;D1])c(-[Cl,Br,I])ccc1",
                1,
                True,
            ),
            (
                "c1c([O;D1])ccc(-[Cl,Br,I])c(-[Cl,Br,I])1.c1c([O;D1])ccc(-[Cl,Br,I])c(-[Cl,Br,I])1",
                1,
                True,
            ),
            (
                "[OX1]=[CX3]1C([F,Cl,Br,I])=C([F,Cl,Br,I])[CX3](=[OX1])C([F,Cl,Br,I])=C1[F,Cl,Br,I]",
                1,
                True,
            ),
            (
                "C1[C!r3!r4!r5!r6!r7!r8](=O)-NC[C!r3!r4!r5!r6!r7!r8](=O)NC[C!r3!r4!r5!r6!r7!r8](=O)N1",
                1,
                True,
            ),
            (
                "C12-,=C-,=C-,=C-,=C-,=C-,=1-,=C=,-C-,=C3-,=C-,=2-,=C-,=C-,=C4-,=C-,=C-,=C-,=C-,=4-,=3",
                1,
                True,
            ),
            (
                "[$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-])]-[CH,CH2]-[Cl,Br,I,$(O(S(=O)(=O)))]",
                1,
                True,
            ),
            (
                "[#6]-,:1(=,:[#6](-,:[#6](=[#8])-,:[#7]-,:[#6](=,:[#7]-,:1)-,:[!#6&!#1])-[#6]#[#7])-[#6]",
                1,
                True,
            ),
            (
                "[#6]1(-O(-[C;!R](-[!N])(=[O,N;!R])))[#6][#6][#6]([$(N(=O)(=O)),$([N+](=O)[O-])])[#6][#6]1",
                1,
                True,
            ),
            (
                "[CH3]-[c;$(cc([CH3])c([CH3])c[CH3]),$(cc([CH3])c([CH3])cc[CH3]),$(cc([CH3])cc([CH3])c[CH3])]",
                1,
                True,
            ),
            (
                "C[C!r3!r4!r5!r6!r7!r8!r9](=O)-NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC",
                1,
                True,
            ),
            (
                "C[C!r3!r4!r5!r6!r7!r8!r9](=O)-NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC",
                1,
                True,
            ),
            (
                "[N;!$(N-S(=O)(=O));!$(N-C=O)]-[N;!r3;!$(N-S(=O)(=O));!$(N-C=O)]-[N;!$(N-S(=O)(=O));!$(N-C=O)]",
                1,
                True,
            ),
            (
                "C[C!r3!r4!r5!r6!r7!r8!r9](=O)-NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC",
                1,
                True,
            ),
            (
                "C[C!r3!r4!r5!r6!r7!r8!r9](=O)-NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC",
                1,
                True,
            ),
            (
                "[Cl,F,Br,I]C([Cl,F,Br,I])([Cl,F,Br,I])C([N,O,S])([N,O,S])C([Cl,F,Br,I])([Cl,F,Br,I])[Cl,F,Br,I]",
                1,
                True,
            ),
            (
                "C-[CH]=C(-[CH3])-[CH]=[CH]-[CH]=C(-[CH3])-[CH]=[CH]-[#6]1:,=[#6]:,-[#6]:,-[#6]:,-[#6]:,-[#6]-,=1",
                1,
                True,
            ),
            (
                "O=C(NCC(NC(C(C(CC=,-C)C)=,-[O,N])C(NCC(NCC(NCC(NCC(NCC(NCC(NCC(NC1)=O)=O)=O)=O)=O)=O)=O)=O)=O)CNC1=O",
                1,
                True,
            ),
            (
                "[C!R]-[C!R]-[C!R]-[C!R]-[C!R]-[C!R]-[C!R]-,=[C!R]-C1~C~C~C~C~1-C-C-,=[C!R]-[C!R]-[C!R]-[C!R]-C(=O)-O",
                1,
                True,
            ),
            (
                "[!#6]~[#6](~[!#6])=[#6;!$([#6]1=[#6]-[#7,#8,#16&v2]~[#7,#8,#16&v2,#6&X3]~[#7,#8,#16&v2]1)](~[!#6])~[!#6]",
                1,
                True,
            ),
            (
                "C1(=O)~[#6]~[#6]N1([$(S(=O)(=O)[C,c,O&D2]),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)[C,c,O&D2])])",
                1,
                True,
            ),
            (
                "[#7;R1]1[#6][#6]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6][#7]1",
                1,
                True,
            ),
            (
                "[#7](-[#1])(-[#1])-[#6]-1=[#6](-[#6]#[#7])-[#6](-[#1])(-[#6]:[#6])-[#6](=[#6](-[#7](-[#1])-[#1])-[#16]-1)-[#6]#[#7]",
                1,
                True,
            ),
            (
                "[#7;R1]1[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]([F,Cl,Br,I])[#6][#6][#7]1",
                1,
                True,
            ),
            (
                "[OX1,N]=[#6]1-,:[#6]=,:[#6]2-,:[#16X2,#8X2]-,:[#6]3=,:[#6]-,:[#6](=,:[#6]-,:[#6]=,:[#6]3-,:[#7]=,:[#6]2-,:[#6]=,:[#6]1)",
                1,
                True,
            ),
            (
                "[#6;!$([#6](-[N,O,S]))]1=[#6;!$([#6](-[N,O,S]))][#6](=[#6])[#6;!$([#6](-[N,O,S]))]=[#6;!$([#6](-[N,O,S]))][#6]1(=[O,N,S])",
                1,
                True,
            ),
            (
                "[#16&v2&R0]=,-[#6;$([#6](=[#16&v2])-[#7]-[#16&v2]),$([#6](-[#16&v2])=[#7]-[#16&v2]),$(c1nsaa1),$(c1nsaaa1)]~[#7]-,:[#16&v2]",
                1,
                True,
            ),
            (
                "[CH2]=C(-[$([#6]=[#8,#16]),$([#7,#15,#16]=[#8]),$(C#N),$(C(F)(F)F)])-[$([#6]=[#8,#16]),$([#7,#15,#16]=[#8]),$(C#N),$(C(F)(F)F)]",
                1,
                True,
            ),
            (
                "[#6]2=,:[#6]3-,:[#6]4=,:[#6](-,:[#6]=,:[#6]-,:[#6]=,:[#6]4)-,:[#6](=O)-,:[#6]4=,:[#6]3-,:[#6](=,:[#6]-,:[#6]=,:[#6]4)[#7]-,:[#6]2=O",
                1,
                True,
            ),
            (
                "N#C-[#6]1~[#6;X3](~[#16&v2;$(S=C-[NH]),$([SH]-C=N),$([SH]-cn),$(S=c[nH])])~[#7]~[*;a,$(*(=*)-*)]~[*;a,$(*(=*)-*)]~[*;a,$(*(=*)-*)]~1",
                1,
                True,
            ),
            (
                "[#7&v3,#8,#16&v2]=,-[#6;X3]1-,:[#6]=,:[#6]-,:[#6]2=,:[#6]-,:[#6]3:[#6]:[#6]:[#6](-[#7&v3,#8,#16&v2]):[#6]:[#6]:3-,:[#7&v3,#8,#16&v2]-,:[#6]-,:2=,:[#6]-,:1",
                1,
                True,
            ),
            (
                "C([Cl,Br,I,$(O(S(=O)(=O)))])=C([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])",
                1,
                True,
            ),
            (
                "O=[C,S]Oc1aaa([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])aa([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C(=O)O),$(C(=O)N)])1",
                1,
                True,
            ),
            (
                "[C;!R]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])=[C;!R]([C;!R](=O))([!$([#8]);!$([#7])])",
                1,
                True,
            ),
            (
                "[#6;$(C14(ccC(=O)O4)c2ccc(-[O,S,N])cc2Oc3ccccc31),$(C1(ccC(=O)O)=C2C=CC(=[O,N,S])C=C2Oc3ccccc31),$(c1(-ccC(=O)O)c2ccc(=[O,N,S])cc2oc3ccccc31)]1(ccC(=O)O)~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]2~[#8]~c3ccccc31",
                1,
                True,
            ),
            (
                "[CH3]-[CH,$(C=C),$(C-[OH])](-[CH3])=,-[CH2,CH&X3,$([CH]-[OH])]-[CH2]-[CH2]-[CH,$(C=C),$(C-[OH])](-[CH3])=,-[CH2,CH&X3,$([CH]-[OH])]-[CH2]-[CH2]-[CH,$(C=C),$(C-[OH])](-[CH3])=,-[CH2,CH&X3,$([CH]-[OH])]-[CH2]",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH2][CH](-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH2][CH](-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH2][CH](-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O)-2",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH2]C(-O2)(-O)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-2",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH](-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH](-[CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH](-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-O2)[CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH2][CH](-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O)-2",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH](-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH2]C(-O2)(-O)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-2",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH](-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-O2)[CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH](-[CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-O2)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]2-O",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH](-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-O2)-[CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O)-2",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH](-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-O2)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O)-2",
                1,
                True,
            ),
            (
                "[O,S][C!R]C(-[CX4&R1,O]1)[CX4][CX4][CX4&R1]1-[#7;$([#7;R1]2-,:[#6;R1](=O)-,:[#7,R1]=,:[#6,R1](-[N&H2,N&H1,O&H1])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2),$([#7;R1]2-,:[#6;R1](=O)-,:[#7,R1;H1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2),$([#7;R1]2-,:[#6;R1](-[OH])=,:[#7,R1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2)]",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH](-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-O2)-[CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O)-2",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH](-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-O2)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O)-2",
                1,
                True,
            ),
            (
                "C~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@C(=O)-[O,N!$(N-a)]",
                1,
                True,
            ),
            (
                "[O,S][C!R]C(-[CX4&R1,O][CX4&R1]1)[CX4&R1][CX4&R1][CX4&R1]1-[#7;$([#7;R1]2-,:[#6;R1](=O)-,:[#7,R1]=,:[#6,R1](-[N&H2,N&H1,O&H1])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2),$([#7;R1]2-,:[#6;R1](=O)-,:[#7,R1;H1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2),$([#7;R1]2-,:[#6;R1](-[OH])=,:[#7,R1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2)]",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CH]1-O-[CH](-O2)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-2",
                1,
                True,
            ),
            (
                "C~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@O-[CH2]-[CH](-O)-[CH2]-O",
                1,
                True,
            ),
            (
                "O=P1O[CX4&R1][CX4&R2](-[CX4&R1,O]2)[CX4&R2](-O1)[CX4&R1][CX4&R1]2-[#7;$([#7;R1]3-,:[#6;R1](=O)-,:[#7,R1]=,:[#6,R1](-[N&H2,N&H1,O&H1])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:3),$([#7;R1]3-,:[#6;R1](=O)-,:[#7,R1;H1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:3),$([#7;R1]3-,:[#6;R1](-[OH])=,:[#7,R1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:3)]",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CH](-O2)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)]-2",
                1,
                True,
            ),
            (
                "C~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@C-OP(=O)(-[O-,OH])-O-[CH2]-[CH2]-N",
                1,
                True,
            ),
            (
                "[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]-C(=O)-O)](-O1)[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)]1-O-[CX4;H1,$([C]-[CH3]),$([C]-[CH2]-O),$(CC(=O)-O)](-O2)-[CX4;$([CH]-O),$([CH]-[N!H0]),H2][CX4;$([CH]-O),$([CH]-[N!H0]),H2]-[CX4;H2,$([CH]-[CH3]),$([CH]-[CH2]-O),$([CH]C(=O)-O)]-2",
                1,
                True,
            ),
            (
                "C~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH](-O)-[CH](-[N!$(N~[O,N,S])])-[CH2]-[OH]",
                1,
                True,
            ),
            (
                "C~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@[CH2,CH&X3,CX2,$([CH]-[OH])]~!@C-O-[CH2]-!@[CH](-O)-!@[CH2]-OP(=O)(-[O-,OH])-O-[CH2]-[CH2]-N",
                1,
                True,
            ),
            (
                "[O,S][C!R][CX4;R1;r6,r7][OR1;r6,r7][CX4;R1;r6,r7;$(C1OCCCC1),$(C1OCC=CC1),$(C1OCCC=C1),$(C1OCCCCC1)]!@[#7;$([#7;R1]2-,:[#6;R1](=O)-,:[#7,R1]=,:[#6,R1](-[N&H2,N&H1,O&H1])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2),$([#7;R1]2-,:[#6;R1](=O)-,:[#7,R1;H1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2),$([#7;R1]2-,:[#6;R1](-[OH])=,:[#7,R1]-,:[#6,R1](=[N,O])-,:[#6,#7;R1]=,:[#6,#7;R1]-,:2)]",
                1,
                True,
            ),
            (
                "[n;R1;X2;!$([n;R1;X2]1:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]):[n;R1;X2]aa1);!$([n;R1;X2]1:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]):[n;R1;X2]a1);!$([n;R1;X2]1:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]a1):[n;R1;X2]);!$([n;R1;X2]1:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]aa1):[n;R1;X2])]:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]):[n;R1;X2;!$([n;R1;X2]1:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]a1):[n;R1;X2]);!$([n;R1;X2]1:[a;R2](:[a;R2,R3]):[a;R2](:[a;R2,R3]aa1):[n;R1;X2])]",
                1,
                True,
            ),
            (
                "[#7;v3;!$(N1N=N[#6]=,:[#6]1);!$(N1N=N[#7]=,:[#6]1);!$(N1N=N[#6]=,:[#6]-[#6](=[#8,#7&v3,#16&v2])1);!$(N1N=N[#6](=[#8,#7&v3,#16&v2])-[#6]=,:[#6]1);!$(N1N=N[#8,#7&v3,#16&v2]C1=[#8,#7&v3,#16&v2]);!$(N1N=NC(=[#8,#7&v3,#16&v2])[#8,#7&v3,#16&v2]1);!$(N1N=NC(=[#8,#7&v3,#16&v2])[#8,#7&v3,#16&v2]C(=[#8,#7&v3,#16&v2])1);!$(N1N=NC(=[#8,#7&v3,#16&v2])C(=[#8,#7&v3,#16&v2])[#8,#7&v3,#16&v2]1);!$(N1N=N[#8,#7&v3,#16&v2]C(=[#8,#7&v3,#16&v2])C1(=[#8,#7&v3,#16&v2]))]-!@[#7;v3]=!@[#7;v3]",
                1,
                True,
            ),
            (
                "[O,S][C!R]C(-[CX4&R1,O]1)[CX4][CX4][CX4&R1]1-[#7;$([#7]2-,:[#6]3=,:[#7,#6;R1]-,:[#7,#6;R1]=,:[#7]-,:[#6](-[NH2,NH,OH])=,:[#6]-,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]=,:[#7,#6;R1]-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7]=,:[#6](-[NH2,NH,OH])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2)]",
                1,
                True,
            ),
            (
                "[O,S][C!R]C(-[CX4&R1,O][CX4&R1]1)[CX4&R1][CX4&R1][CX4&R1]1-[#7;$([#7]2-,:[#6]3=,:[#7,#6;R1]-,:[#7,#6;R1]=,:[#7]-,:[#6](-[NH2,NH,OH])=,:[#6]-,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]=,:[#7,#6;R1]-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7]=,:[#6](-[NH2,NH,OH])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2)]",
                1,
                True,
            ),
            (
                "O=P1O[CX4&R1][CX4&R2](-[CX4&R1,O]2)[CX4&R2](-O1)[CX4&R1][CX4&R1]2-[#7;$([#7]3-,:[#6]4=,:[#7,#6;R1]-,:[#7,#6;R1]=,:[#7]-,:[#6](-[NH2,NH,OH])=,:[#6]-,:4-,:[#7,#6;R1]=,:[#6,#7;R1]-,:3),$([#7]3-,:[#6]4-,:[#7,#6;R1]=,:[#7,#6;R1]-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:4-,:[#7,#6;R1]=,:[#6,#7;R1]-,:3),$([#7]3-,:[#6]4-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:4-,:[#7,#6;R1]=,:[#6,#7;R1]-,:3),$([#7]3-,:[#6]4-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7]=,:[#6](-[NH2,NH,OH])-,:[#6]=,:4-,:[#7,#6;R1]=,:[#6,#7;R1]-,:3)]",
                1,
                True,
            ),
            (
                "[O,S][C!R][CX4;R1;r6,r7][OR1;r6,r7][CX4;R1;r6,r7;$(C1OCCCC1),$(C1OCC=CC1),$(C1OCCC=C1),$(C1OCCCCC1)]!@[#7;$([#7]2-,:[#6]3=,:[#7,#6;R1]-,:[#7,#6;R1]=,:[#7]-,:[#6](-[NH2,NH,OH])=,:[#6]-,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]=,:[#7,#6;R1]-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7;H1]-,:[#6](=[NH,O])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2),$([#7]2-,:[#6]3-,:[#7,#6;R1]-,:[#6](=[O,NH])-,:[#7]=,:[#6](-[NH2,NH,OH])-,:[#6]=,:3-,:[#7,#6;R1]=,:[#6,#7;R1]-,:2)]",
                1,
                True,
            ),
            (
                "[F,Cl,Br,I]-[C;!$(C1=N[#7]-,:[#6](=,:[#8,#7&v3,#16&v2])-,:[#7&v3,#6&X3]=,:[#7&v3,#6&X3]-1);!$(C1=N[#7]-,:[#6](=,:[#8,#7&v3,#16&v2])-,:[#7&v3,#16&v2,#8]-1);!$(C1=N-[#8,#7&v3,#16&v2]-,:[#8,#7&v3,#16&v2]-,:[#6]1=,:[#8,#7&v3,#16&v2]);!$(C1=N-[#6](=,:[#8,#7&v3,#16&v2])-,:[#8,#7&v3,#16&v2]-,:[#8,#7&v3,#16&v2]-1);!$(C1=N-[#6](=,:[#8,#7&v3,#16&v2])-,:[#8,#7&v3,#16&v2]-,:[#7&v3,#6&X3]=,:[#7&v3,#6&X3]-1);!$(C1=N-[#6](=,:[#8,#7&v3,#16&v2])-,:[#7&v3,#6&X3]=,:[#7&v3,#6&X3]-,:[#8,#7&v3,#16&v2]-1);!$(C1=N-[#6]=,:[#6]-[#8,#7&v3,#16&v2]-,:[#6]1=,:[#8,#7&v3,#16&v2])]=N",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);$(N-;!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);$(N-;!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);$(N-;!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N);$(N-;!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(N-;!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            (
                "[$([O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8]),$([O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8]),$([O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][CH,CH2;!r3;!r4;!r5;!r6;!r7;!r8][O,S,#7;R1;!r3;!r4;!r5;!r6;!r7;!r8])]",
                1,
                True,
            ),
            (
                "[a;$(a1:[a;!$(a=*)]:[a;!$(a=*)]:[a;!$(a=*)]:[a;!$(a=*)]:[a;!$(a=*)]:1),$(a1:[a;!$(a=*)]:[a;!$(a=*)]:[a;!$(a=*)]:[a;!$(a=*)]:1)]-[C;!$(C~[#7,#8,#16,F,Cl,Br,I]);!$([#6]1=[#6]-[#6,#7&v3]=,:[#6,#7&v3]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6,#7&v3]=,:[#6,#7&v3]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#6,#7&v3]=,:[#6,#7&v3]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6](=[#8,#7&v3,#16&v2])-[#8,#7&v3,#16&v2]-1)]=[C;!$([#6]1=[#6]-[#6,#7&v3]=,:[#6,#7&v3]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6,#7&v3]=,:[#6,#7&v3]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#6,#7&v3]=,:[#6,#7&v3]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6](=[#8,#7&v3,#16&v2])-[#8,#7&v3,#16&v2]-1)](-[$([#6;H1]=[#8,#16]),$([#7,#15,#16]=[#8])])-[$([#6;H1]=[#8,#16]),$([#7,#15,#16]=[#8])]",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC);$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O))]",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC);$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O))]",
                1,
                True,
            ),
            (
                "[C!r3!r4!r5!r6!r7!r8!r9](=O)-NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)[N;!$([NH2]);!$(NC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC);!$(NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)NC);$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)),$(N([C!r3!r4!r5!r6!r7!r8!r9](=O)CCN-[C!r3!r4!r5!r6!r7!r8!r9](=O))-;!@A!@A!@A!@A!@A!@A!@A-;!@N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O))]",
                1,
                True,
            ),
            (
                "N[C!r3!r4!r5!r6!r7!r8!r9](=O)CN[C!r3!r4!r5!r6!r7!r8!r9](=O)-;!@[C;$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            (
                "N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)-;!@[C;$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            (
                "N[C!r3!r4!r5!r6!r7!r8!r9](=O)CCN[C!r3!r4!r5!r6!r7!r8!r9](=O)-;!@[C;$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N),$(C(-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)!@A!@A!@A!@A!@A!@A!@A!@C-;!@[C!r3!r4!r5!r6!r7!r8!r9](=O)NCC[C!r3!r4!r5!r6!r7!r8!r9](=O)N)]",
                1,
                True,
            ),
            ("C=C=C", 1, False),
            ("O=CC#N", 1, False),
            ("C#CC#C", 1, False),
            ("[#6]=S", 1, False),
            ("N#CCC#N", 1, False),
            ("C1O[C;H2]1", 1, False),
            ("C(=O)N[OH]", 1, False),
            ("c1nnnn1C=O", 1, False),
            ("O=C1C=COC=C1", 1, False),
            ("[C;$(C(F)F)]", 3, False),
            ("c1cnnn1-O-C=O", 1, False),
            ("s1c(S)nnc1NC=O", 1, False),
            ("C=[C!R]-[C!R]=C", 1, False),
            ("[C;X4]-O-[C;X4]", 5, False),
            ("[S;X2;H0]C=[O,N]", 1, False),
            ("C=!@C-[F,Cl,Br,I]", 1, False),
            ("[#6]-C(=O)-[CH2]-F", 1, False),
            ("[P;!$(P(~O)(~O)~O)]", 1, False),
            ("C(=O)!@N!@C(=O)[#7]", 1, False),
            ("C([CH3])([CH3])[CH3]", 3, False),
            ("C1=CC2c3ccccc3NCC2C1", 1, False),
            ("[CH1;R0]=[N;R0][N;R0]", 1, False),
            ("[#6]-[O]-!@C(=O)-[#6]", 3, False),
            ("[+1;H0;!$([+1]~[-1])]", 1, False),
            ("O=C-!@[O]-!@[N]=!@[#6]", 1, False),
            ("c-O-[#6;!$([#5]=[!#6])]", 5, False),
            ("O=C-N-c1scc(-c)c1-C(=O)-O", 1, False),
            ("[C!X4]O-S(=O)(=O)-C(F)(F)F", 1, False),
            ("[#6]-[N;X3](~[O;X1])~[O;X1]", 2, False),
            ("[S;X2]1[C;X4;R1]N[C;R1][C;R1]1", 1, False),
            ("[c;$(c1ccccc1),$(c1ccac1)][OH]", 2, False),
            ("c1ccc(n[o,s]n2)c2c1[N+](=O)[O-]", 1, False),
            ("[#6,#8,#16]-C(=[NH1])[#6,#8,#16]", 1, False),
            ("c1(N)ccc(C(=O)NC3(=O))c(c3ccc2)c21", 1, False),
            ("[C;H1&$([#6]-[#6]),H2]=!@N-[*;!#7;!#8]", 1, False),
            ("a1(a2aa(a3aaaaa3)aa(a4aaaaa4)a2)aaaaa1", 1, False),
            ("FC(F)(F)-[CH2]-[O;$(O-C=O),$(O-[Si]),H1]", 1, False),
            ("[C;!$(C(Cl)(Cl)Cl);!$(C(F)F)]-[F,Cl,Br,I]", 3, False),
            ("[!#1;!#6;!#7;!#8;!#9;!#15;!#16;!#17;!#35]", 1, False),
            ("C14-c2c(cccc2)-C(-[#6]~[#6]-1)-c3c(cccc3)-4", 1, False),
            ("[Cl,Br,I]-[CX4;!$(C([F,Cl,Br,I])[F,Cl,Br,I])]", 1, False),
            ("[O]~N([c;R1]1[c;H1;R1][c;H1][c;R1][n,o]1)~[O]", 1, False),
            ("[#7](-[#6](-[#8])=[#6]1)-[#6](=[#7])-[#6]1=[#7]", 1, False),
            ("O=[N+](-[O-])-caac-[$(N(C)C),$([NH]C),$([NH2])]", 1, False),
            ("C=!@[#6]1~[#6;X3](~[#16&v2])~[#7]~[#7]=,:[#6]~1", 1, False),
            ("[#8;H1,$([#8]-[#6,#16,#15]=[#8])]-[#6;X4]-[#6]#[#7]", 1, False),
            ("[#8]=[#6]-2-[#6](=!@[#7]-[#7])-c:1:c:c:c:c:c:1-[#7]-2", 1, False),
            ("[O;!$(O=C1C=C2CCCCC2CC1)]=C1C=C2C=,-CC3C4CCCC4CCC3-,=C2CC1", 1, False),
            ("a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a@;:a", 1, False),
            (
                "[S&v2]-[#6]1-,:[#16&v2]-,:[#6]=,:[#6]-,:[#6](=[#8])-,:[#6]=,:1",
                1,
                False,
            ),
            (
                "[nv3X2]c([c,s,n])[CX4][NX3]([H,C])[CX4]c([c,s,n])[nv3X2,nv3X3H1]",
                1,
                False,
            ),
            (
                "[CH2;R0]-[CH2;R0]-[CH2;R0]-[CH2;R0]-[CH2;R0]-[CH2;R0]-[CH2,CH3;R0]",
                1,
                False,
            ),
            (
                "[$(N(-C(=O))(-C(=O))(-S(=O))),$(n([#6](=O))([#6](=O))([#16](=O)))]",
                1,
                False,
            ),
            (
                "[#8;$([#8]=[#6]),$([#8]-[#6]=,:[#7])]~[#6]1~[#7]-,:[#16&X2]-,:c2ccccc12",
                1,
                False,
            ),
            (
                "c:1:c:c:c:c(:c:1-[#7&!H0;!H1,!$([#7]-[#6]=[#8])])-[#6](-[#6]:[#6])=[#8]",
                1,
                False,
            ),
            (
                "[#6]1=,:[#6]([OH1])-,:[#7X3]-,:[#6X3](=[OX1])-,:[#7X3]-,:[#6X3]-,:1(=[OX1])",
                1,
                False,
            ),
            (
                "[#6]-[#6](-[#6]-[#8,#16]-[c]1:[c](-[#17,#9]):[c]:[c]:[c]:[c]1-[#17,#9])=[#8]",
                1,
                False,
            ),
            (
                "[$(O=C[CH](C=O)C=O),$(O=CC(C=O)=C[OH]),$(N#C[CH](-C=O)-C=O),$(N#CC(-C=O)=C[OH])] ",
                1,
                False,
            ),
            (
                "C=!@[#6]1~[#6;X3](~[#7,#8,#16v2])~[#7;X2]~[#6;X3;H1,$([#6]-[#6])]~[#8,#7&v3,#16v2]~1",
                1,
                False,
            ),
            (
                "[#8,#7;!$([#7]=[#6](-[#7])-[#7]);!$([#7]-[#6](=[#7])-[#7]);!$([#7]:[#6](:[#7]):[#7])]-C#N",
                1,
                False,
            ),
            (
                "[C;!$(C-[#6;X3,X2]);!$(C-[#7,#8,#16])]=!@[C;!$(C-[#6;X3,X2]);!$(C-[#8,#16]);!$(C(-[#7])[#7])]-N",
                1,
                False,
            ),
            (
                "C=!@[#6]1~[#6;X3](~[#8;$(O=C),$(O-C=N),$(O-cn)])~[#7]-[#7]~[#6;X3](~[#8;$(O=C),$(O-C=N),$(O-cn)])~1",
                1,
                False,
            ),
            (
                "c:1(:c(:c(:c(:c(:c:1-[#1])-[#1])-[$([#8]),$([#7]),$([#6](-[#1])-[#1])])-[#1])-[#1])-[#7](-[#1])-[#1]",
                1,
                False,
            ),
            (
                "[#6&X3]1(~[#8])~[#7]~[#6&X3](~[#8])~[#7]~[#6&X3]2~[#6&X3]~1~[#7&X2]~[#6&X3]~[#6&x3,#7&X2]~[#7,#16&v2]~2",
                1,
                False,
            ),
            (
                "[O,S&v2,N&v3;R0]-[C;X3;R0](=[O,S&v2,N&v3;R0])-[O,S&v2,N&v3;R0]-,=[C;X3;R0](=,-[O,S&v2,N&v3;R0])-[O,S&v2,N&v3;R0]",
                1,
                False,
            ),
            (
                "[#7;R1]1[#6]([F,Cl,Br,I])[#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#7][#6][#6]1",
                1,
                False,
            ),
            (
                "[#7;R1]1[#6]([F,Cl,Br,I])[#7][#6][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]1",
                1,
                False,
            ),
            (
                "[c;$(cc(-[F,Cl,Br,I])c-[F,Cl,Br,I]),$(cc(-[F,Cl,Br,I])cc-[F,Cl,Br,I]),$(ccc(-[F,Cl,Br,I])cc-[F,Cl,Br,I])]-[F,Cl,Br,I]",
                1,
                False,
            ),
            (
                "C=!@[#6]1~[#6;X3](~[#8;$(O=C),$(O-C=N),$(O-cn)])~[#7]~[#6;X3](~[#8,#16&v2;$(A=C),$(A-C=N),$(A-cn)])~[#7]~[#6;X3](~[#8;$(O=C),$(O-C=N),$(O-cn)])1",
                1,
                False,
            ),
            (
                "[#7;R1]1[#6;!$(c=O);!$(c-N)][#6]([$(S(=O)(=O)),$(C(F)(F)(F)),$(C#N),$(N(=O)(=O)),$([N+](=O)[O-]),$(C=O)])[#6]([F,Cl,Br,I])[#6][#6;!$(c=O);!$(c-N)]1",
                1,
                False,
            ),
            (
                "[#7,#16&v2;$([#7,#16]=[#6]1-[#7]-[#6]-[#7,#6]-[#16]-1),$([#7,#16;H1,H2]-[#6]1=[#7]-[#6]-[#7,#6]-[#16]-1)]=,-[#6]1=,-[#7]-[#6&X4]-[#7,#6&X4]-[#16&v2]-1",
                1,
                False,
            ),
            (
                "[F,Cl,Br,I]C([F,Cl,Br,I])([F,Cl,Br,I])[#6;$([#6](=[#8,#7&v3,#16&v2])[#8,#16&v2]),$([#6;H1]=[#8,#7&v3,#16&v2]),$([#6](-[#8,#7&v3,#16&v2,#15v3])-[#8,#7&v3,#15v3])]",
                1,
                False,
            ),
            (
                "[F,Cl,Br,I]-[#6;$(c1[n;X2]aa[a;!c]1),$(C1=N[#7]-,:[#6](=,:[#8,#7&v3,#16&v2])-,:[#7&v3,#16&v2,#8]-1),$(C1=N-[#8,#7&v3,#16&v2]-,:[#8,#7&v3,#16&v2]-,:[#6]1=,:[#8,#7&v3,#16&v2]),$(C1=N-[#6](=,:[#8,#7&v3,#16&v2])-,:[#8,#7&v3,#16&v2]-,:[#8,#7&v3,#16&v2]-1)]=,:[#7;X2]",
                1,
                False,
            ),
            (
                "[#8,#7v3]=!@[#6;$([#6](-[#6])-[#6]),$([#6;H1]=[#7]);!$([#6]1(=O)-,:[#6](=O)-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1);!$([#6]1(=O)-[#6](=O)-[#6]=,:[#6]1)]~[#6;$([#6](-[#6])-[#6]),$([#6;H1]=[#7]);!$([#6]1(=O)-,:[#6](=O)-,:[#6;X3]=,:[#6;X3]-,:[#6;X3]=,:[#6;X3]-,:1);!$([#6]1(=O)-[#6](=O)-[#6]=,:[#6]1)]=!@[#8,#7v3]",
                1,
                False,
            ),
            (
                "[#16&v2]~[#6&X3&R1;!$([#6]1(-A):[a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])]1);!$([#6]1(-A):[a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])][a;!$(a=*);!$(nc-[#8,#16&v2;H1]);!$(nc=[#8,#16&v2])]1)](~[#7&v3])~@[#7&v3,#6]",
                1,
                False,
            ),
            (
                "[#6]-[C!$(C1-,:[#6&X3,#7&v3]=,:[#6&X3,#7&v3]-,:[#7&v3,#16&v2,#8]-,:[#6&X3,#7&v3]=,:[#6&X3,#7&v3]-,:1);!$(C1-,:[#6&X3,#7&v3]=,:[#6&X3,#7&v3]-,:[#6&X3,#7&v3]=,:[#6&X3,#7&v3]-,:[#7&v3,#16&v2,#8]-,:1);!$(C1-,:[#7&v3,#16&v2,#8]-C(=[#7&v3,#16&v2,#8])-,:[#6&X3,#7&v3]=,:[#6&X3,#7&v3]-,:[#7&v3,#16&v2,#8]-,:1);!$(C1-,:[#7&v3,#16&v2,#8]-C(=[#7&v3,#16&v2,#8])-,:[#7&v3,#16&v2,#8]-,:[#6&X3,#7&v3]=,:[#6&X3,#7&v3]-,:1);!$(C1-,:[#7&v3,#16&v2,#8]-C(=[#7&v3,#16&v2,#8])-,:[#7&v3,#16&v2,#8]-,:[#7&v3,#16&v2,#8]-C(=[#7&v3,#16&v2,#8])-,:1)](-[#6])=!@[N;!$(N~[#7,#8])]",
                1,
                False,
            ),
            (
                "[C;!$([#6]1=[#6]-[#6,#7&v3]=,:[#6,#7&v3]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6,#7&v3]=,:[#6,#7&v3]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#6,#7&v3]=,:[#6,#7&v3]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6](=[#8,#7&v3,#16&v2])-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#8]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-1)]=[C;!$([#6]1=[#6]-[#6,#7&v3]=,:[#6,#7&v3]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6,#7&v3]=,:[#6,#7&v3]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#6,#7&v3]=,:[#6,#7&v3]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-[#6](=[#8,#7&v3,#16&v2])-[#8,#7&v3,#16&v2]-1);!$([#6]1=[#6]-[#8]-[#6](=[#8,#7,#16&v2])-[#8,#7&v3,#16&v2]-1)]-O-[#6]",
                1,
                False,
            ),
        ]
        condition_names = [smarts for smarts, min_count, exclude in filters]
        filters = [
            (MolFromSmarts(smarts), min_count, exclude)
            for smarts, min_count, exclude in filters
        ]
        return filters, condition_names
