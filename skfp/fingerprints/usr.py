from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import get_conf_id, require_mols_with_conformations


class USRFingerprint(BaseFingerprintTransformer):
    """
    USR (Ultrafast Shape Recognition) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, which
    characterizes the shape of the molecule by encoding the relative positions of its
    atoms [1]_ [2]_.

    Four points are considered: molecular centroid (ctd), the closest atom to centroid (catc),
    the farthest atom from centroid (fact), and atom the fartest from fct (fatf). Distances
    from all atoms to each of those four points are computed, and each of those distributions
    is summarized its first three moments: mean, variance, and skewness. Concretely, standard
    deviation and cubic root of skewness are used to keep the same unit. This results in
    12 descriptors.

    This is a 3D fingerprint, and requires molecules with ``conf_id`` integer property
    set. They can be generated with :class:`~skfp.preprocessing.ConformerGenerator`.
    Furthermore, only molecules with 3 or more atoms are allowed, to allow computation
    of all three moments.

    **Warning:** parallel computation results in values slightly different from sequential
    one, but is still fully deterministic.

    Parameters
    ----------
    errors : {"raise", "NaN", "ignore"}, default="raise"
        How to handle errors during fingerprint calculation. ``"raise"`` immediately
        raises any errors. ``"NaN"`` returns NaN values for molecules which resulted in
        errors. ``"ignore"`` suppresses errors and does not return anything for
        molecules with errors. This potentially results in fewer output vectors than
        input molecules, and should be used with caution.

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
    n_features_out : int = 12
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D-based. It always requires
        molecules with conformers as input. If the ``conf_id`` property is set, it
        will be used to determine the conformation. You can use
        :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

    See Also
    --------
    :class:`USRCAT` : Related fingerprint, which additionally uses pharmacophoric
        atom types.

    :class:`ElectroShape` : Related fingerprint, which additionally uses atomic
        partial charges.

    References
    ----------
    .. [1] `Pedro J. Ballester and W. Graham Richards
        "Ultrafast shape recognition for similarity search in molecular databases"
        Proceedings of the Royal Society A: Mathematical, Physical and Engineering
        Sciences 463.2081 (2007): 1307-1321
        <https://royalsocietypublishing.org/doi/full/10.1098/rspa.2007.1823>`_

    .. [2] `Pedro J. Ballester
        "Ultrafast shape recognition: method and applications"
        Future Medicinal Chemistry 3.1 (2011): 65-78
        <https://www.future-science.com/doi/abs/10.4155/fmc.10.280>`_

    Examples
    --------
    >>> from skfp.fingerprints import USRFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["CC=O"]
    >>> fp = USRFingerprint()
    >>> fp
    USRFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[ 1.33723405,  0.39526642, ...        ,  0.94315243, -0.78539494]])
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "errors": [StrOptions({"raise", "NaN", "ignore"})],
    }

    def __init__(
        self,
        errors: str = "raise",
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=12,
            requires_conformers=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.errors = errors

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to aggregates
        of atomic distances to 4 centroid-based points.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            USR feature names.
        """
        feature_names = [
            "centroid_dists_mean",
            "centroid_dists_std",
            "centroid_dists_skewness_cubic_root",
            "closest_atom_to_centroid_mean",
            "closest_atom_to_centroid_std",
            "closest_atom_to_centroid_skewness_cubic_root",
            "farthest_atom_from_centroid_mean",
            "farthest_atom_from_centroid_std",
            "farthest_atom_from_centroid_skewness_cubic_root",
            "farthest_atom_from_farthest_to_centroid_mean",
            "farthest_atom_from_farthest_to_centroid_std",
            "farthest_atom_from_farthest_to_centroid_cubic_root",
        ]
        return np.asarray(feature_names, dtype=object)

    def transform(self, X: Sequence[Mol], copy: bool = False) -> np.ndarray | csr_array:
        """
        Compute USR fingerprints. If ``errors`` is set to ``"ignore"``, then in
        case of errors less than n_samples values may be returned.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.
        """
        y = np.empty(len(X))
        X, _ = self.transform_x_y(X, y, copy=copy)
        return X

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[np.ndarray | csr_array, np.ndarray]:
        """
        Compute USR fingerprints. If ``errors`` is set to ``"ignore"``, then in
        case of errors less than n_samples values may be returned. The returned
        values for X and y are properly synchronized.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects, with conformers generated and
            ``conf_id`` integer property set.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=False
            Copy the input labels y or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Array with fingerprints.

        y : np.ndarray of shape (n_samples,)
            Array with labels for molecules.
        """
        if copy:
            y = deepcopy(y)

        X = super().transform(X)

        if self.errors == "ignore":
            # errors are marked as NaN rows
            idxs_to_keep = [idx for idx, x in enumerate(X) if not np.any(np.isnan(x))]
            X = X[idxs_to_keep]
            y = y[idxs_to_keep]

        return X, y

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> np.ndarray | csr_array:
        from rdkit.Chem.rdMolDescriptors import GetUSR

        X = require_mols_with_conformations(X)

        if self.errors == "raise":
            fps = [GetUSR(mol, confId=get_conf_id(mol)) for mol in X]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = GetUSR(mol, confId=get_conf_id(mol))
                except ValueError:
                    fp = np.full(self.n_features_out, np.nan)
                fps.append(fp)

        return np.array(fps)
