from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import get_conf_id, require_mols_with_conformations


class USRCATFingerprint(BaseFingerprintTransformer):
    """
    USRCAT (Ultrafast Shape Recognition with CREDO Atom Types) fingerprint.

    The implementation uses RDKit. This is a descriptor-based fingerprint, extending
    the USR fingerprint by additionally considering pharmacophoric atom types [1]_.

    Firstly, the four reference points are computed, and they are used in all five
    feature groups. Each group is processed exactly the same, i.e. distance distributions
    to reference points are calculated, and features from the first three moments
    (mean, standard deviation, cubic root of skewness) are computed. This results in
    12 features.

    USRCAT expands on USR by considering 4 additional subsets of atoms, based on their
    pharmacophoric types: hydrophobic, aromatic, hydrogen bond donor or acceptor atoms.
    For each atoms subset, distance distribution and moments are calculated like in
    the original USR. This results in 5 * 12 = 60 features.

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
    n_features_out : int = 60
        Number of output features, size of fingerprints.

    requires_conformers : bool = True
        Value is always True, as this fingerprint is 3D-based. It always requires
        molecules with conformers as input. If the ``conf_id`` property is set, it
        will be used to determine the conformation. You can use
        :class:`~skfp.preprocessing.ConformerGenerator` to generate them.

    See Also
    --------
    :class:`USR` : Related fingerprint, which USRCAT expands.

    :class:`USRCAT` : Related fingerprint, which expands USR with atomic partial charges,
        instead of pharmacophotic atom types.

    References
    ----------
    .. [1] `Adrian M. Schreyer and Tom Blundell
        "USRCAT: real-time ultrafast shape recognition with pharmacophoric constraints"
        Journal of Cheminformatics 4, 27 (2012)
        <https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-4-27#citeas>`_

    Examples
    --------
    >>> from skfp.fingerprints import USRCATFingerprint
    >>> from skfp.preprocessing import MolFromSmilesTransformer, ConformerGenerator
    >>> smiles = ["CC=O"]
    >>> fp = USRCATFingerprint()
    >>> fp
    USRCATFingerprint()

    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mols = mol_from_smiles.transform(smiles)
    >>> conf_gen = ConformerGenerator()
    >>> mols = conf_gen.transform(mols)
    >>> fp.transform(mols)  # doctest: +SKIP
    array([[ 1.33723405,  0.39526642, ...        ,  0.        ,  0.        ]])
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
            n_features_out=60,
            requires_conformers=True,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.errors = errors

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        """
        Get fingerprint output feature names. They correspond to aggregates
        of atomic distances to 4 centroid-based points, for each of 5 atom
        type subsets.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            USRCAT feature names.
        """
        base_feature_names = [
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
        atom_types = [
            "all",
            "hydrophobic",
            "aromatic",
            "hydrogen_bond_donor",
            "hydrogen_bond_acceptor",
        ]
        feature_names = [
            f"{atom_type}_{base_feature_name}"
            for atom_type in atom_types
            for base_feature_name in base_feature_names
        ]

        return np.asarray(feature_names, dtype=object)

    def transform(self, X: Sequence[Mol], copy: bool = False) -> np.ndarray | csr_array:
        """
        Compute USRCAT fingerprints. If ``errors`` is set to ``"ignore"``, then in
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
        Compute USRCAT fingerprints. If ``errors`` is set to ``"ignore"``, then in
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
        from rdkit.Chem.rdMolDescriptors import GetUSRCAT

        X = require_mols_with_conformations(X)

        if self.errors == "raise":
            fps = [GetUSRCAT(mol, confId=get_conf_id(mol)) for mol in X]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = GetUSRCAT(mol, confId=get_conf_id(mol))
                except ValueError:
                    fp = np.full(self.n_features_out, np.nan)
                fps.append(fp)

        return np.array(fps)
