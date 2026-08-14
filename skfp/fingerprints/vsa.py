from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class VSAFingerprint(BaseFingerprintTransformer):
    """
    VSA (Van der Waals Surface Area) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, which
    calculates the atomic contributions to the approximate van der Waals surface
    area (VSA / ASA), based on given atomic properties [1]_.

    To calculate VSA, one gets the contribution of each atom in the molecule to
    a molecular property (e.g. SLogP) along with the contribution of each atom to
    the approximate molecular surface area (VSA), assign the atoms to bins based on
    the property contributions, and then sum up the VSA contributions for each atom
    in a bin.

    Features based on atomic contributions can be used:

    - SlogP, calculated water-octanol partition coefficient, measures lipophilicity [2]_
    - SMR, molar refractivity, measures polarizability [2]_
    - PEOE, Gasteiger partial charges, measure direct electrostatic interactions [3]_
    - EState, electrotopological states, encode information about both the topological
      environment of that atom and the electronic interactions [4]_

    Histogram bins are based on feature distributions in molecular datasets. RDKit uses
    slightly different bins from the original paper, see [5]_ [6]_ for exact values.
    See also [7]_ for methods of interpreting the values.

    Parameters
    ----------
    variant : {"all", "all_original", "SlogP", "SMR", "PEOE", "EState"}, default="all_original"
        Which features to use for calculating VSA bins. "all_original" uses three
        features from the original paper [1]_, i.e. SlogP, SMR and PEOE. "all" also
        adds EState.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int
        Number of output features. Depends on variant: "SlogP" (12), "SMR" (10),
        "PEOE" (14), "EState" (11), "all_original" (36) or "all" (47).

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    References
    ----------
    .. [1] `Paul Labute
        "A widely applicable set of descriptors"
        Journal of Molecular Graphics and Modelling, Volume 18, Issues 4-5, 2000, Pages 464-477
        <https://www.sciencedirect.com/science/article/pii/S1093326300000681>`_

    .. [2] `Scott A. Wildman and Gordon M. Crippen
        "Prediction of Physicochemical Parameters by Atomic Contributions"
        J. Chem. Inf. Comput. Sci. 1999, 39, 5, 868-873
        <https://pubs.acs.org/doi/10.1021/ci990307l>`_

    .. [3] `Johann Gasteiger and Mario Marsili
        "Iterative partial equalization of orbital electronegativity — a rapid access to atomic charges"
        Tetrahedron, Volume 36, Issue 22, 1980, Pages 3219-3228
        <https://www.sciencedirect.com/science/article/abs/pii/0040402080801682>`_

    .. [4] `Lowell H. Hall and Lemont B. Kier
        "Electrotopological State Indices for Atom Types: A Novel Combination of Electronic,
        Topological, and Valence State Information"
        J. Chem. Inf. Comput. Sci. 1995, 35, 6, 1039-1045
        <https://pubs.acs.org/doi/10.1021/ci00028a014>`_

    .. [5] `RDKit SlogP, SMR and PEOE bin values
        <https://github.com/rdkit/rdkit/blob/41f5c82ccf36a1b339dbf00440d823379decc2c3/rdkit/Chem/MolSurf.py>`_

    .. [6] `RDKit EState bin values
        <https://github.com/rdkit/rdkit/blob/41f5c82ccf36a1b339dbf00440d823379decc2c3/rdkit/Chem/EState/EState_VSA.py>`_

    .. [7] `Gregory Landrum
        "What are the VSA Descriptors?"
        <https://greglandrum.github.io/rdkit-blog/posts/2023-04-17-what-are-the-vsa-descriptors.html>`_

    Examples
    --------
    >>> from skfp.fingerprints import VSAFingerprint
    >>> smiles = ["CCO", "CCN"]
    >>> fp = VSAFingerprint()
    >>> fp
    VSAFingerprint()

    >>> fp.transform(smiles)  # doctest: +SKIP
    array([[ 0.        , 11.71340936,  0.        ,  ...,  0.        ,  0.        , 0.        ],
           [ 5.73366748,  6.54475641,  0.        ,  ...,  0.        ,  0.        , 0.        ]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "variant": [
            StrOptions({"all", "all_original", "SlogP", "SMR", "PEOE", "EState"})
        ],
    }

    def __init__(
        self,
        variant: str = "all_original",
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=self._get_n_features_out(variant),
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.variant = variant

    def _get_n_features_out(self, variant: str) -> int:
        n_features_out = {
            "SlogP": 12,
            "SMR": 10,
            "PEOE": 14,
            "EState": 11,
            "all_original": 36,  # SlogP + SMR + PEOE
            "all": 47,  # all Labute + Estate
        }
        try:
            return n_features_out[variant]
        except KeyError as err:
            raise ValueError(f'Variant "{variant}" not recognized') from err

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to histogram
        bins for descriptors.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            VSA feature names.
        """
        from rdkit.Chem.EState.EState_VSA import estateBins as EState_bins
        from rdkit.Chem.MolSurf import chgBins as PEOE_bins
        from rdkit.Chem.MolSurf import logpBins as SlogP_bins
        from rdkit.Chem.MolSurf import mrBins as SMR_bins

        group_feature_names = {}
        for group_name, bins in [
            ("SlogP", SlogP_bins),
            ("SMR", SMR_bins),
            ("PEOE", PEOE_bins),
            ("EState", EState_bins),
        ]:
            less_than_name = f"{group_name} < {bins[0]}"
            greater_than_name = f"{group_name} >= {bins[-1]}"
            # e.g. "0.1<SlogP<0.2"
            bin_names = [
                f"{bins[i]} <= {group_name} < {bins[i + 1]}"
                for i in range(len(bins) - 1)
            ]
            group_feature_names[group_name] = [
                less_than_name,
                *bin_names,
                greater_than_name,
            ]

        if self.variant in group_feature_names:
            feature_names = group_feature_names[self.variant]
        elif self.variant == "all_original":
            feature_names = (
                group_feature_names["SlogP"]
                + group_feature_names["SMR"]
                + group_feature_names["PEOE"]
            )
        else:  # "all"
            feature_names = (
                group_feature_names["SlogP"]
                + group_feature_names["SMR"]
                + group_feature_names["PEOE"]
                + group_feature_names["EState"]
            )

        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute VSA fingerprints. Output shape depends on ``variant``
        parameter.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.n_features_out)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem.EState.EState_VSA import EState_VSA_
        from rdkit.Chem.rdMolDescriptors import PEOE_VSA_, SMR_VSA_, SlogP_VSA_

        mols = ensure_mols(X)

        vsa_funcs = {
            "SlogP": lambda mols: np.array([SlogP_VSA_(mol) for mol in mols]),
            "SMR": lambda mols: np.array([SMR_VSA_(mol) for mol in mols]),
            "PEOE": lambda mols: np.array([PEOE_VSA_(mol) for mol in mols]),
            "EState": lambda mols: np.array([EState_VSA_(mol) for mol in mols]),
        }

        if self.variant in vsa_funcs:
            X = vsa_funcs[self.variant](mols)
        elif self.variant == "all_original":
            X_slogp = vsa_funcs["SlogP"](mols)
            X_smr = vsa_funcs["SMR"](mols)
            X_peoe = vsa_funcs["PEOE"](mols)
            X = np.column_stack((X_slogp, X_smr, X_peoe))
        else:  # "all"
            X_slogp = vsa_funcs["SlogP"](mols)
            X_smr = vsa_funcs["SMR"](mols)
            X_peoe = vsa_funcs["PEOE"](mols)
            X_estate = vsa_funcs["EState"](mols)
            X = np.column_stack((X_slogp, X_smr, X_peoe, X_estate))

        return csr_array(X) if self.sparse else X
