from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles
from sklearn.utils import _safe_indexing

from skfp.bases import BasePreprocessor
from skfp.utils import (
    get_data_from_indices,
    no_rdkit_logs,
    require_mols,
    require_strings,
)


class MolFromSmilesTransformer(BasePreprocessor):
    """
    Creates RDKit ``Mol`` objects from SMILES strings.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization [1]_, i.e. basic validity checks, on created
        molecules.

    replacements : dict, default=None
        If provided, will be used to do string substitution of abbreviations in the
        input SMILES.

    valid_only: bool, default=False
        Whether to return only molecules that were successfully loaded. By default,
        returns ``None`` for molecules that got errors.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    suppress_warnings: bool, default=False
        Whether to suppress warnings and errors on loading molecules.

    verbose : int or dict, default=0
        Controls the verbosity when processing molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `Gregory Landrum
        "The RDKit Book: Molecular Sanitization"
        <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSmilesTransformer
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mol_from_smiles
    MolFromSmilesTransformer()

    >>> mol_from_smiles.transform(smiles)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        **BasePreprocessor._parameter_constraints,
        "sanitize": ["boolean"],
        "replacements": [dict, None],
        "valid_only": ["boolean"],
        "suppress_warnings": ["boolean"],
    }

    def __init__(
        self,
        sanitize: bool = True,
        replacements: dict | None = None,
        valid_only: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        suppress_warnings: bool = False,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            batch_size=batch_size,
            suppress_warnings=suppress_warnings,
            verbose=verbose,
        )
        self.sanitize = sanitize
        self.replacements = replacements
        self.valid_only = valid_only

    def transform(self, X, copy: bool = False) -> list[Mol]:
        """
        Create RDKit ``Mol`` objects from SMILES strings. If ``valid_only`` is set to
        True, returns only a subset of molecules which could be successfully loaded.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples,)
            List with RDKit ``Mol`` objects.
        """
        y = np.empty(len(X))
        X, _ = self.transform_x_y(X, y, copy=copy)
        return X

    def transform_x_y(self, X, y, copy: bool = False) -> tuple[list[Mol], Any]:
        """
        Create RDKit ``Mol`` objects from SMILES strings. If ``valid_only`` is set to
        True, returns only a subset of molecules and labels which could be successfully
        loaded.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings.

        y : {list, np.ndarray, pd.Series, pd.DataFrame}
            Labels for molecules.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples,)
            List with RDKit ``Mol`` objects.

        y : {list, np.ndarray, pd.Series, pd.DataFrame}
            Labels for the returned molecules, preserving the input type.
        """
        X = super().transform(X, copy)
        if self.valid_only:
            idxs_to_keep = [idx for idx, mol in enumerate(X) if mol is not None]
            X = get_data_from_indices(X, idxs_to_keep)
            y = _safe_indexing(y, idxs_to_keep)

        return X, y

    def _transform_batch(self, X: Sequence[str]) -> list[Mol]:
        with no_rdkit_logs() if self.suppress_warnings else nullcontext():
            require_strings(X)
            replacements = self.replacements or {}
            return [
                MolFromSmiles(smi, sanitize=self.sanitize, replacements=replacements)
                for smi in X
            ]


class MolToSmilesTransformer(BasePreprocessor):
    """
    Creates SMILES strings from RDKit ``Mol`` objects.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    isomeric_smiles : bool, default=True
        Whether to include information about stereochemistry.

    kekule_smiles : bool, default=False
        Whether to use the Kekule form (no aromatic bonds).

    canonical : bool, default=True
        Whether to canonicalize the molecule. This results in a reproducible
        SMILES, given the same input molecule (if ``do_random`` is not used).

    all_bonds_explicit : bool, default=False
        Whether to explicitly indicate all bond orders.

    all_hs_explicit : bool, default=False
        Whether to explicitly indicate all hydrogens.

    do_random : bool, default=False
        If True, randomizes the traversal of the molecule graph, generating
        random SMILES.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    suppress_warnings: bool, default=False
        Whether to suppress warnings and errors on saving molecules.

    verbose : int or dict, default=0
        Controls the verbosity when processing molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------
    .. [1] `RDKit MolToSmiles documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.MolToSmiles>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSmilesTransformer, MolToSmilesTransformer
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mol_to_smiles = MolToSmilesTransformer()
    >>> mol_to_smiles
    MolToSmilesTransformer()

    >>> mols = mol_from_smiles.transform(smiles)
    >>> mol_to_smiles.transform(mols)
    ['O', 'CC', '[C-]#N', 'CC=O']
    """

    _parameter_constraints: dict = {
        **BasePreprocessor._parameter_constraints,
        "isomeric_smiles": ["boolean"],
        "kekule_smiles": ["boolean"],
        "canonical": ["boolean"],
        "all_bonds_explicit": ["boolean"],
        "all_hs_explicit": ["boolean"],
        "do_random": ["boolean"],
        "suppress_warnings": ["boolean"],
    }

    def __init__(
        self,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        suppress_warnings: bool = False,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            batch_size=batch_size,
            suppress_warnings=suppress_warnings,
            verbose=verbose,
        )
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        self.canonical = canonical
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.do_random = do_random

    def _transform_batch(self, X: Sequence[Mol]) -> list[str]:
        with no_rdkit_logs() if self.suppress_warnings else nullcontext():
            require_mols(X)
            return [
                MolToSmiles(
                    mol,
                    isomericSmiles=self.isomeric_smiles,
                    kekuleSmiles=self.kekule_smiles,
                    canonical=self.canonical,
                    allBondsExplicit=self.all_bonds_explicit,
                    allHsExplicit=self.all_hs_explicit,
                    doRandom=self.do_random,
                )
                for mol in X
            ]
