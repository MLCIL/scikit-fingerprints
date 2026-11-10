import warnings
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.fingerprints._mordred_implementations.base import BaseMordredImplementation
from skfp.fingerprints._mordred_implementations.optimized import (
    OptimizedMordredImplementation,
)
from skfp.fingerprints._new_mordred.descriptors.detour_matrix import (
    DEFAULT_MAX_RING_SYSTEM_RANK,
)
from skfp.fingerprints._new_mordred.utils.descriptor_evaluation import DEFAULT_TIMEOUT
from skfp.utils import ensure_mols, require_mols_with_conformations


class MordredFingerprint(BaseFingerprintTransformer):
    """
    Mordred fingerprint.

    This is a descriptor-based fingerprint, implementing many 2D, and optionally 3D,
    molecular descriptors, originally implemented in the Mordred [2]_ library. By
    default, they are computed by an optimized implementation, built from scratch. The
    ``mordredcommunity`` [3]_ library can be used instead, with the ``implementation``
    parameter.

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

    implementation : {"optimized", "mordredcommunity"}, default="optimized"
        Which implementation computes the descriptors. ``"optimized"`` is the
        implementation built into scikit-fingerprints. ``"mordredcommunity"`` uses the
        ``mordredcommunity`` library, as earlier scikit-fingerprints versions did, and
        requires the ``mordredcommunity`` optional dependency, installed with
        ``pip install "scikit-fingerprints[mordredcommunity]"``. Both implementations
        compute the same set of descriptors, but under different feature names, and
        with 3D descriptors in a different order, so their outputs are not
        interchangeable.

    timeout : float, default=60.0
        Timeout in seconds for computing descriptors that can be very slow for
        molecules with many rings. Descriptors not computed within it are NaN.

    max_ring_system_rank : int, default=23
        Maximal number of independent cycles of a ring system, for which the detour
        matrix descriptors are computed. They are NaN for larger ring systems, since
        their computation time grows exponentially with it. Increase it to compute them
        for highly fused molecules, at the cost of much longer computation, limited by
        ``timeout``. Only used with ``implementation="optimized"``.

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

    requires_conformers : bool
        Whether the fingerprint is 3D-based and requires molecules with conformers as
        inputs. This depends on the ``use_3D`` parameter, and has the same value. If the
        ``conf_id`` property is set, it will be used to determine the conformation.
        You can use :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

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
        "implementation": [StrOptions({"optimized", "mordredcommunity"})],
        "timeout": [Interval(Real, 0, None, closed="neither")],
        "max_ring_system_rank": [Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        use_3D: bool = False,
        implementation: str = "optimized",
        timeout: float = DEFAULT_TIMEOUT,
        max_ring_system_rank: int = DEFAULT_MAX_RING_SYSTEM_RANK,
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
        self.implementation = implementation
        self.timeout = timeout
        self.max_ring_system_rank = max_ring_system_rank

    def __setstate__(self, state: dict) -> None:
        """
        Restore a pickled object. Objects pickled before the ``"optimized"``
        implementation existed computed their descriptors with ``mordredcommunity``,
        so they keep using it, to give the same values as before.
        """
        state.setdefault("implementation", "mordredcommunity")
        state.setdefault("timeout", DEFAULT_TIMEOUT)
        state.setdefault("max_ring_system_rank", DEFAULT_MAX_RING_SYSTEM_RANK)
        super().__setstate__(state)

    def _validate_params(self) -> None:
        super()._validate_params()
        # built once, in the main process, and shared by all batches; this also makes
        # a missing mordredcommunity fail before joblib splits the molecules
        self._implementation = self._get_implementation()
        if (
            self.implementation == "mordredcommunity"
            and self.max_ring_system_rank != DEFAULT_MAX_RING_SYSTEM_RANK
        ):
            warnings.warn(
                "max_ring_system_rank is only used with implementation='optimized', "
                "ignoring it for implementation='mordredcommunity'"
            )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. Depends on ``implementation``
        parameter, including both order and names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Mordred feature names.
        """
        return self._get_implementation().get_feature_names()

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute Mordred fingerprints.

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
        X : {ndarray, sparse matrix} of shape (n_samples, 1613) or (n_samples, 1826)
            Array with fingerprints.
        """
        return super().transform(X, copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = require_mols_with_conformations(X) if self.use_3D else ensure_mols(X)
        X = self._implementation.compute(X)
        return (
            csr_array(X, dtype=np.float32)
            if self.sparse
            else np.array(X, dtype=np.float32)
        )

    def _get_implementation(self) -> BaseMordredImplementation:
        if self.implementation == "optimized":
            return OptimizedMordredImplementation(
                use_3D=self.use_3D,
                timeout=self.timeout,
                max_ring_system_rank=self.max_ring_system_rank,
            )

        if self.implementation == "mordredcommunity":
            # mordredcommunity is an optional dependency, imported only when chosen
            try:
                from skfp.fingerprints._mordred_implementations.mordred_community import (
                    MordredCommunityImplementation,
                )
            except ModuleNotFoundError as e:
                # other errors, e.g. from a broken mordredcommunity installation, are
                # not solved by installing it
                if e.name != "mordred":
                    raise
                raise ImportError(
                    "MordredFingerprint with implementation='mordredcommunity' "
                    "requires the mordredcommunity optional dependency. Install it "
                    'with: pip install "scikit-fingerprints[mordredcommunity]"'
                ) from e
            return MordredCommunityImplementation(
                use_3D=self.use_3D, timeout=self.timeout
            )

        raise ValueError(
            "implementation must be 'optimized' or 'mordredcommunity', "
            f"got: {self.implementation!r}"
        )
