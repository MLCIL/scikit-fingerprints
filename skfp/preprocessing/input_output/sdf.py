import os.path
import warnings
from collections.abc import Sequence
from numbers import Integral

from joblib import effective_n_jobs
from rdkit.Chem import Mol, SDMolSupplier, SDWriter
from rdkit.Chem.PropertyMol import PropertyMol

from skfp.bases import BasePreprocessor
from skfp.utils import require_mols
from skfp.utils.functions import _get_rdkit_version

_MIN_MULTITHREADED_SDF_VERSION = (2025, 9, 1)


class MolFromSDFTransformer(BasePreprocessor):
    """
    Creates RDKit ``Mol`` objects from SDF string or file.

    SDF (structure-data format) is processed for whole files, rather than individual
    molecules. For this reason ``.transform()`` either reads the SDF file directly
    from disk or takes a string input in that format.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization [1]_, i.e. basic validity checks, on created
        molecules.

    remove_hydrogens : bool, default=True
        Remove explicit hydrogens from the molecule where possible, using RDKit
        implicit hydrogens instead.

    n_jobs : int, default=None
        The number of jobs to use when reading molecules from an SDF file path.
        If ``n_jobs > 1`` and the installed RDKit version is at least ``2025.09.1``
        the file is read in parallel. Raw SDF text input is always processed sequentially.

    References
    ----------
    .. [1] `RDKit SDMolSupplier documentation
        <https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.SDMolSupplier>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSDFTransformer
    >>> sdf_file_path = "mols_in.sdf"
    >>> mol_from_sdf = MolFromSDFTransformer()  # doctest: +SKIP
    >>> mol_from_sdf  # doctest: +SKIP
    MolFromSDFTransformer()

    >>> mol_from_sdf.transform(sdf_file_path)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        "sanitize": ["boolean"],
        "remove_hydrogens": ["boolean"],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        sanitize: bool = True,
        remove_hydrogens: bool = True,
        n_jobs: int | None = None,
    ):
        super().__init__(n_jobs=n_jobs)
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens

    def transform(self, X: str, copy: bool = False) -> list[Mol]:  # type: ignore[override]    # noqa: ARG002
        """
        Create RDKit ``Mol`` objects from SDF file.

        Parameters
        ----------
        X : str
            Path to SDF file.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        X : list of shape (n_samples,)
            List with RDKit ``Mol`` objects.
        """
        self._validate_params()

        if X.endswith(".sdf"):
            if not os.path.exists(X):
                raise FileNotFoundError(f"SDF file at path '{X}' not found")

            mols = self._read_sdf_file(X)
        else:
            mols = self._read_sdf_text(X)

        if not mols:
            warnings.warn("No molecules detected in provided SDF file")

        return mols

    def _transform_batch(self, X):
        pass  # unused

    def _read_sdf_file(self, filepath: str) -> list[Mol]:
        n_jobs = effective_n_jobs(self.n_jobs)

        if n_jobs > 1:
            rdkit_version = _get_rdkit_version()
            if rdkit_version < _MIN_MULTITHREADED_SDF_VERSION:
                warnings.warn(
                    "Parallel SDF reading requires RDKit >= 2025.09.1. "
                    f"Installed version is {'.'.join(map(str, rdkit_version))}. "
                    "Falling back to sequential loading."
                )
            else:
                return self._read_sdf_file_parallel(filepath, n_jobs)

        return list(
            SDMolSupplier(
                filepath,
                sanitize=self.sanitize,
                removeHs=self.remove_hydrogens,
            )
        )

    def _read_sdf_file_parallel(self, filepath: str, n_jobs: int) -> list[Mol]:
        from rdkit.Chem import MultithreadedSDMolSupplier

        with MultithreadedSDMolSupplier(
            filepath,
            sanitize=self.sanitize,
            removeHs=self.remove_hydrogens,
            numWriterThreads=n_jobs,
        ) as supplier:
            mols_with_record_ids = [
                (supplier.GetLastRecordId(), mol)
                for mol in supplier
                if mol is not None  # multithreaded supplier may yield None duplicates
            ]

        mols_with_record_ids.sort(key=lambda item: item[0])
        return [mol for _, mol in mols_with_record_ids]

    def _read_sdf_text(self, sdf_text: str) -> list[Mol]:
        if effective_n_jobs(self.n_jobs) > 1:
            warnings.warn(
                "Parallel SDF reading requires a file path. Falling back to sequential "
                "loading for raw SDF text input."
            )

        supplier = SDMolSupplier()
        supplier.SetData(
            sdf_text,
            sanitize=self.sanitize,
            removeHs=self.remove_hydrogens,
        )
        return list(supplier)


class MolToSDFTransformer(BasePreprocessor):
    """
    Creates SDF file from RDKit ``Mol`` objects.

    SDF (structure-data format) is processed for whole files, rather than individual
    molecules. For this reason ``.transform()`` saves the results directly to file.

    If ``conf_id`` integer property is set for molecules, they are used to determine
    the conformer to save.

    For details see RDKit documentation [1]_.

    Parameters
    ----------
    filepath : string, default="mols.sdf"
        A string with file path location to save the SDF file. It should be a valid
        file path with ``.sdf`` extension.

    kekulize : bool, default=True
        Whether to kekulize molecules before writing them to SDF file.

    force_V3000 : bool, default=False
        Whether to force the V3000 format when writing to SDF file.

    References
    ----------
    .. [1] `RDKit SDWriter documentation
        <https://rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.SDWriter>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSDFTransformer, MolToSDFTransformer
    >>> sdf_file_path = "mols_in.sdf"
    >>> mol_from_sdf = MolFromSDFTransformer()
    >>> mol_to_sdf = MolToSDFTransformer(filepath="mols_out.sdf")
    >>> mol_to_sdf
    MolToSDFTransformer(filepath='mols_out.sdf')

    >>> mols = mol_from_sdf.transform(sdf_file_path)  # doctest: +SKIP
    >>> mol_to_sdf.transform(mols)  # doctest: +SKIP
    """

    _parameter_constraints: dict = {
        "filepath": [str],
        "kekulize": ["boolean"],
        "force_V3000": ["boolean"],
    }

    def __init__(
        self,
        filepath: str = "mols.sdf",
        kekulize: bool = True,
        force_V3000: bool = False,
    ):
        super().__init__()
        self.filepath = filepath
        self.kekulize = kekulize
        self.force_V3000 = force_V3000

    def transform(self, X: Sequence[Mol], copy: bool = False) -> None:  # noqa: ARG002
        """
        Write RDKit ``Mol`` objects to SDF file at location given by
        ``filepath`` parameter. File is created if necessary, and overwritten
        if it exists already.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects.

        copy : bool, default=False
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
            None
        """
        self._validate_params()
        require_mols(X)

        with open(self.filepath, "w") as file:
            writer = SDWriter(file)
            writer.SetKekulize(self.kekulize)
            writer.SetForceV3000(self.force_V3000)

            for mol in X:
                if isinstance(mol, PropertyMol) and mol.HasProp("conf_id"):
                    writer.write(mol, confId=mol.GetIntProp("conf_id"))
                else:
                    writer.write(mol)

            writer.flush()

    def _transform_batch(self, X):
        pass  # unused
