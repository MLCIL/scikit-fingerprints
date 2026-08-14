from collections.abc import Sequence

import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class MordredFingerprint(BaseFingerprintTransformer):
    """
    Mordred fingerprint.

    The implementation uses ``mordredcommunity`` [3]_ library. This is a descriptor-based
    fingerprint, implementing a very large number of 2D, and optionally 3D, molecular
    descriptors, originally implemented in the Mordred [2]_ library.

    Descriptors include simple counts (e.g. atom types, rings), topological indices,
    computed properties (e.g. ClogP, polarizability), and more. For a full list, see
    Supplementary File 3 in the original publication [1]_.

    Many descriptors are often undefined for a given molecule and result in NaN, e.g.
    ring descriptors for acyclic structures. Feature selection and/or imputation must
    often be used before passing the result to downstream estimators.

    Parameters
    ----------
    use_3D : bool, default=False
        Whether to include 3D (conformer-based) descriptors. Using this option results
        in longer computation time.

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
        Number of output features, size of fingerprints: 1613 for 2D-only variant,
        and 1826 if using 3D descriptors.

    requires_conformers : bool = False
        This fingerprint in the 3D variant computes conformers internally, and
        therefore does not require conformers.

    References
    ----------
    .. [1] `Moriwaki, Hirotomo, et al.
        "Mordred: a molecular descriptor calculator"
        J Cheminform 10, 4 (2018)
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y>`_

    .. [2] https://github.com/mordred-descriptor/mordred

    .. [3] https://github.com/JacksonBurns/mordred-community

    Examples
    --------
    >>> from skfp.fingerprints import MordredFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MordredFingerprint()
    >>> fp
    MordredFingerprint()

    >>> fp.transform(smiles)
    array([[0.       , 0.       , 0.       , ..., 0.       ,       nan,
            0.       ],
           [0.       , 0.       , 0.       , ..., 1.       , 2.       ,
            1.       ],
           [0.       , 0.       , 1.       , ..., 1.       , 2.       ,
            1.       ],
           [1.4142135, 1.4142135, 0.       , ..., 4.       , 2.25     ,
            1.       ]], shape=(4, 1613), dtype=float32)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "use_3D": ["boolean"],
    }

    def __init__(
        self,
        use_3D: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        n_features_out = 1826 if use_3D else 1613
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=use_3D,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.use_3D = use_3D

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to descriptor
        names used by Mordred descriptor calculator, used in this fingerprint.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Mordred feature names.
        """
        calc = Calculator(descriptors)
        if not self.use_3D:
            feature_names = [str(d) for d in calc.descriptors if d.require_3D is False]
        else:
            feature_names = [str(d) for d in calc.descriptors]
        return np.asarray(feature_names, dtype=object)

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute Mordred fingerprints.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, 1613) or (n_samples, 1826)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = ensure_mols(X)

        calc = Calculator(descriptors, ignore_3D=not self.use_3D)
        X = [calc(mol) for mol in X]

        return (
            csr_array(X, dtype=np.float32)
            if self.sparse
            else np.array(X, dtype=np.float32)
        )
