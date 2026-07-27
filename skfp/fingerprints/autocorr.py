from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols, get_conf_id, require_mols_with_conformations


class AutocorrFingerprint(BaseFingerprintTransformer):
    """
    Autocorrelation fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, where
    bits measure strength of autocorrelation of molecular properties between atoms
    with different shortest path distances. For 3D variant, those distances are
    further weighted by Euclidean distance between atoms in a given conformation.

    It uses a hydrogen-depleted molecule, and for each heavy atom computes 6 features:
    atomic mass, van der Waals volume, electronegativity, polarizability, ion polarity,
    and IState [1]_ [2]_. They are then made relative to the carbon, e.g. molecular
    weight is: MW(atom_type) / MW(carbon).

    Four autocorrelation measures are used: Moreau-Broto, centered (average) Moreau-Broto,
    Moran and Geary [3]_ [4]_. They are calculated using topological distances (shortest
    paths), with distance between 1 and 8 (inclusive). This results in 192 features:
    6 atom features * 4 autocorrelations * 8 distances.

    3D variant has the following differences:

    - requires passing molecules with conformers and ``conf_id`` integer property set
    - weights topological distances by Euclidean distance between atoms
    - uses 2 additional features: constant 1 (which measures Euclidean distance
      autocorrelation due to weighting) and covalent radius (RCov)
    - uses shortest paths distances between 1 and 9 (inclusive)
    - uses only Moreau-Broto autocrrelation
    - results in 80 features: 8 atom features * 10 distances

    For more details on autocorrelation descriptors, see [5]_.

    Parameters
    ----------
    use_3D : bool, default=False
        Whether to use 3D Euclidean distance matrix. If False, only uses topological
        distances on molecular graph.

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
    n_features_out : int = 192 or 80
        Number of output features. Equal to 192 for 2D and 80 for 3D, which depends
        on the ``use_3D`` parameter.

    requires_conformers : bool
        Whether the fingerprint is 3D-based and requires molecules with conformers as
        inputs. This depends on the ``use_3D`` parameter, and has the same value. If the
        ``conf_id`` property is set, it will be used to determine the conformation.
        You can use :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

    References
    ----------
    .. [1] RDKit IState discussion
        https://github.com/rdkit/rdkit/discussions/6122

    .. [2] RDKit IState implementation
        https://github.com/rdkit/rdkit/blob/df2b6c8a8c775748c1dcac83af0f92af434bab81/Code/GraphMol/Descriptors/MolData3Ddescriptors.cpp#L127

    .. [3] `Rajarshi Guha
        "Autocorrelation Descriptors"
        <http://www.rguha.net/writing/notes/desc/node2.html>`_

    .. [4] `Roberto Todeschini and Viviana Consonni
        "Molecular Descriptors for Chemoinformatics"
        <https://onlinelibrary.wiley.com/doi/book/10.1002/9783527628766>`_

    .. [5] `Guillaume Godin
        "3D	descriptors in RDKit"
        UGM 2017
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Godin_3D_Descriptors.pdf>`_

    Examples
    --------
    >>> from skfp.fingerprints import AutocorrFingerprint
    >>> smiles = ["CCO", "CCN"]
    >>> fp = AutocorrFingerprint()
    >>> fp
    AutocorrFingerprint()

    >>> fp.transform(smiles)  # doctest: +ELLIPSIS
    array([[ 1.204,  0.847,  0.   ,  ...,  0.   ,  0.   ,  0.   ],
           [ 1.153,  0.773,  0.   ,  ...,  0.   ,  0.   ,  0.   ]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        use_3D: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        n_features_out = 80 if use_3D else 192
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.use_3D = use_3D

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names.

        They differ depending on ``use_3D`` parameter:

        - 2D: features correspond to 4 autocorrelation functions, 6 atomic descriptors
          and 8 distance buckets
        - 3D: features correspond to 8 atomic descriptors and 10 distance buckets

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Autocorrelation feature names.
        """
        if not self.use_3D:
            autocorr_funcs = ["Moreau-Broto", "Centered Moreau-Broto", "Moran", "Geary"]
            descriptors = [
                "mass",
                "van der Waals volume",
                "electronegativity",
                "polarizability",
                "ion polarity",
                "IState",
            ]
            feature_names = [
                f"{autocorr_func} {descriptor} distance {distance}"
                for autocorr_func in autocorr_funcs
                for descriptor in descriptors
                for distance in range(8)
            ]
        else:
            descriptors = [
                "unweighted",
                "mass",
                "van der Waals volume",
                "electronegativity",
                "polarizability",
                "ion polarity",
                "IState",
                "covalent radius",
            ]
            feature_names = [
                f"Moreau-Broto {descriptor} distance {distance}"
                for descriptor in descriptors
                for distance in range(10)
            ]

        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute Autocorrelation fingerprints. Output shape depends on ``use_3D``
        parameter.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects. If ``use_3D``
            is True, only ``Mol`` objects with computed conformations and with
            ``conf_id`` property are allowed.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 192) or (n_samples, 80)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR2D, CalcAUTOCORR3D

        if not self.use_3D:
            X = ensure_mols(X)
            X = [CalcAUTOCORR2D(mol) for mol in X]
        else:
            X = require_mols_with_conformations(X)
            X = [CalcAUTOCORR3D(mol, confId=get_conf_id(mol)) for mol in X]

        return np.array(X)
