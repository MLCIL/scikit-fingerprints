from collections.abc import Sequence
from numbers import Real

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import get_conf_id, require_mols_with_conformations


class WHIMFingerprint(BaseFingerprintTransformer):
    """
    WHIM (Weighted Holistic Invariant Molecular descriptors) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, where
    bits measure rotation-invariant 3D information regarding size, shape, symmetry
    and atom distributions.

    Features are based on the principal component analysis (PCA) on the centered
    cartesian coordinates of a molecule by using a weighted covariance matrix. There are
    two groups of features, each one measuring size, shape, symmetry and density of atoms:
    - 11 directional, using scores of individual principal axes
    - 7 global, aggregating information about the whole molecule

    Additionally, all directional features and 5 of the global ones are computed
    using unweighted distances matrix, as well as 6 weighted variants, using atomic
    features: atomic mass, van der Waals volume, electronegativity, polarizability,
    ion polarity, and IState [1]_ [2]_. They are relative to the carbon, e.g. molecular
    weight is: MW(atom_type) / MW(carbon).

    This gives 114 features: 11 * 7 weighted directional + 5 * 7 weighted global
    + 2 unweighted global. See [3]_ [4]_ [5]_ [6]_ for details.

    Typical correct values should be small, but can result in NaN or infinity for some
    molecules. Value clipping with ``clip_val`` parameter, feature selection, and/or
    imputation should be used.

    Parameters
    ----------
    clip_val : float or None, default=2147483647
        Value to clip results at, both positive and negative ones.The default value is
        the maximal value of 32-bit integer, but should often be set lower, depending
        on the application. ``None`` means that no clipping is applied.

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
    n_features_out : int = 114
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D-based. It always requires
        molecules with conformers as input. If the ``conf_id`` property is set, it
        will be used to determine the conformation. You can use
        :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

    References
    ----------
    .. [1] RDKit IState discussion
        https://github.com/rdkit/rdkit/discussions/6122

    .. [2] RDKit IState implementation
        https://github.com/rdkit/rdkit/blob/df2b6c8a8c775748c1dcac83af0f92af434bab81/Code/GraphMol/Descriptors/MolData3Ddescriptors.cpp#L127

    .. [3] `Rajarshi Guha
        "Weighted Holistic Invariant Molecular Descriptors"
        <http://www.rguha.net/writing/notes/desc/node7.html>`_

    .. [4] `Roberto Todeschini and Viviana Consonni
        "Molecular Descriptors for Chemoinformatics"
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    .. [5] `Guillaume Godin
        "3D	descriptors in RDKit"
        UGM 2017
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf>`_

    .. [6] `Roberto Todeschini and Paola Gramatica
        "The WHIM Theory: New 3D Molecular Descriptors for QSAR in Environmental Modelling"
        SAR and QSAR in Environmental Research, 7(1-4), 89-115
        <https://www.tandfonline.com/doi/abs/10.1080/10629369708039126>`_

    Examples
    --------
    >>> from skfp.fingerprints import WHIMFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = WHIMFingerprint()
    >>> fp
    WHIMFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[0.44 , 0.067, 0.   , ..., 0.514, 0.537, 0.537],
           [1.17 , 0.395, 0.393, ..., 2.266, 3.38 , 2.542],
           [0.329, 0.   , 0.   , ..., 0.329, 0.329, 0.329],
           [1.196, 0.507, 0.242, ..., 2.183, 3.285, 3.129]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "clip_val": [None, Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        clip_val: float = 2147483647,  # max int32 value
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=114,
            requires_conformers=True,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.clip_val = clip_val

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to various
        descriptors derived from weighted covariance matrix. See references
        given in main body for explanations.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            WHIM feature names.
        """
        weighting_variants = [
            "unweighted",
            "atomic mass",
            "van der Waals volume",
            "electronegativity",
            "polarizability",
            "ion polarity",
            "IState",
        ]

        # first 77 directional features
        directional_feature_names = [
            f"{weight} {feature_name}"
            for weight in weighting_variants
            for feature_name in [
                "axis 1 directional WHIM size",
                "axis 2 directional WHIM size",
                "axis 3 directional WHIM size",
                "axis 1 directional WHIM shape",
                "axis 2 directional WHIM shape",
                "axis 1 directional WHIM symmetry",
                "axis 2 directional WHIM symmetry",
                "axis 3 directional WHIM symmetry",
                "axis 1 directional WHIM density",
                "axis 2 directional WHIM density",
                "axis 3 directional WHIM density",
            ]
        ]
        global_whim_size_feature_names = [
            f"{weight} global WHIM size" for weight in weighting_variants
        ]
        global_whim_size_cross_sums_feature_names = [
            f"{weight} global WHIM size cross sums" for weight in weighting_variants
        ]
        global_whim_axial_shape_feature_names = [
            f"{weight} global WHIM axial shape" for weight in weighting_variants[:2]
        ]
        global_whim_shape_feature_names = [
            f"{weight} global WHIM shape" for weight in weighting_variants
        ]
        global_whim_density_feature_names = [
            f"{weight} global WHIM density" for weight in weighting_variants
        ]
        global_whim_symmetry_feature_names = [
            f"{weight} global WHIM symmetry" for weight in weighting_variants
        ]

        feature_names = (
            directional_feature_names
            + global_whim_size_feature_names
            + global_whim_size_cross_sums_feature_names
            + global_whim_axial_shape_feature_names
            + global_whim_shape_feature_names
            + global_whim_density_feature_names
            + global_whim_symmetry_feature_names
        )

        return np.asarray(feature_names, dtype=object)

    def transform(self, X: Sequence[Mol], copy: bool = False) -> np.ndarray | csr_array:
        """
        Compute WHIM fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 114)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem.rdMolDescriptors import CalcWHIM

        X = require_mols_with_conformations(X)
        X = [CalcWHIM(mol, confId=get_conf_id(mol)) for mol in X]
        if self.clip_val is not None:
            X = np.clip(X, -self.clip_val, self.clip_val)

        return csr_array(X) if self.sparse else np.array(X)
