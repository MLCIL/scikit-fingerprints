import itertools
import struct
from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
from numbers import Integral

import numpy as np
from rdkit.Chem import Mol, MolToSmiles, PathToSubmol
from rdkit.Chem.rdmolops import FindAtomEnvironmentOfRadiusN, GetDistanceMatrix
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_random_state

from skfp.bases import BaseFingerprintTransformer
from skfp.utils import ensure_mols


class MAPFingerprint(BaseFingerprintTransformer):
    """
    MinHashed Atom Pair fingerprint (MAP).

    Implementation is based on the MAP4 and MAP4C papers [1]_ [2]_. This is a
    hashed fingerprint, using the ideas from Atom Pair and SECFP fingerprints.

    It computes fragments based on pairs of atoms, using circular substructures
    around each atom represented with SMILES (like SECFP) and length of shortest
    path between them (like Atom Pair), and then hashes the resulting triplet
    using MinHash algorithm into a resulting fingerprint.

    Subgraphs are created around each atom with increasing radius, starting
    with just an atom itself. It is then transformed into a canonical SMILES.
    In each iteration, it is increased by another atom (one "hop" on the graph).

    For each pair of atoms, length of the shortest path is created. Then, for
    each radius from 1 up to the given maximal radius, the triplet is created:
    (atom 1 SMILES, shortest path length, atom 2 SMILES). They are then hashed
    using MinHash algorithm.

    See also original MAP [3]_ and MAPC [4]_ implementations.

    Parameters
    ----------
    fp_size : int, default=2048
        Size of output vectors. Depending on the ``variant`` argument, those are either
        raw hashes, bits, or counts. Must be positive.

    radius : int, default=2
        Number of iterations performed, i.e. maximum radius of resulting subgraphs.
        Another common notation uses diameter, therefore MAP4 has radius 2.

    include_duplicated_shingles : bool, default=False
        Whether to include duplicated shingles in the final fingerprint.

    include_chirality : bool, default=False
        Whether to include chirality information when computing atom types. This is
        also known as MAPC fingerprint [3]_ [4]_.

    variant : {"binary", "count", "minhash"}, default="binary"
        Output fingerprint variant:
        - ``"binary"``: folded binary fingerprint,
        - ``"count"``: folded count fingerprint,
        - ``"minhash"``: MinHash sketch.

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

    random_state : int or None, default=0
        Controls the randomness of the MinHash computation. Pass an integer
        for reproducible results across multiple function calls. ``None`` means
        that no fixed seed is used.

    Attributes
    ----------
    n_features_out : int
        Number of output features. Equal to ``fp_size``.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require conformers.

    See Also
    --------
    :class:`AtomPairFingerprint` : Related fingerprint, which uses atom invariants and
        folding.

    :class:`SECFPFingerprint` : Related fingerprint, which only uses SMILES of circular
        subgraph around each atom.

    References
    ----------
    .. [1] `Alice Capecchi, Daniel Probst, Jean-Louis Reymond
        "One molecular fingerprint to rule them all: drugs, biomolecules, and the metabolome"
        J Cheminform 12, 43 (2020)
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00445-4>`_

    .. [2] `Markus Orsi, Jean-Louis Reymond
        "One chiral fingerprint to find them all"
        J Cheminform 16, 53 (2024)
        <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00849-6>`_

    .. [3] `https://github.com/reymond-group/map4`

    .. [4] `https://github.com/markusorsi/mapchiral/tree/main`

    Examples
    --------
    >>> from skfp.fingerprints import MAPFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = MAPFingerprint()
    >>> fp
    MAPFingerprint()

    >>> fp.transform(smiles)
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], shape=(4, 1024), dtype=uint8)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "radius": [Interval(Integral, 0, None, closed="left")],
        "include_duplicated_shingles": ["boolean"],
        "include_chirality": ["boolean"],
        "variant": [StrOptions({"binary", "count", "minhash"})],
    }

    _MINHASH_PRIME = np.uint64((1 << 61) - 1)

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 2,
        include_duplicated_shingles: bool = False,
        include_chirality: bool = False,
        variant: str = "binary",
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        random_state: int | None = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=(variant == "count"),
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.include_duplicated_shingles = include_duplicated_shingles
        self.include_chirality = include_chirality
        self.variant = variant

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute MAP fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, self.fp_size)
            Transformed data.
        """
        return super().transform(X, copy=copy)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        X = ensure_mols(X)

        if self.variant in {"minhash", "count"}:
            dtype = np.uint32
        else:
            dtype = np.uint8

        X = np.stack(
            [self._calculate_single_mol_fingerprint(mol) for mol in X],
            dtype=dtype,
        )

        return csr_array(X) if self.sparse else np.array(X)

    def _calculate_single_mol_fingerprint(self, mol: Mol) -> np.ndarray:
        from rdkit.Chem.rdCIPLabeler import AssignCIPLabels

        if self.include_chirality:
            AssignCIPLabels(mol)

        atoms_envs = self._get_atom_envs(mol)
        shinglings = self._get_atom_pair_shingles(mol, atoms_envs)
        hashed_shinglings = self._hash_shingles(shinglings)

        if self.variant == "minhash":
            return self._minhash(hashed_shinglings)

        return self._fold(hashed_shinglings)

    def _get_atom_envs(self, mol: Mol) -> dict[int, list[str | None]]:
        from rdkit.Chem import FindMolChiralCenters

        # for each atom get its environment, i.e. radius-hop neighborhood.
        atom_envs: dict[int, list[str | None]] = defaultdict(list)

        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                atom_env = self._find_env(mol, atom_idx, radius)
                atom_envs[atom_idx].append(atom_env)

        # in chiral MAP, Cahn-Ingold-Prelog (CIP) descriptor between $ signs replaces
        # the first (chiral center) atom identifier for largest radius in SMILES string
        if self.include_chirality:
            chiral_centers = FindMolChiralCenters(mol, includeUnassigned=True)
            for atom_idx, cip_label in chiral_centers:
                max_radius_smiles = atom_envs[atom_idx][-1]
                if max_radius_smiles:
                    atom_envs[atom_idx][-1] = f"${cip_label}${max_radius_smiles[1:]}"

        return atom_envs

    def _find_env(self, mol: Mol, atom_identifier: int, radius: int) -> str | None:
        # get SMILES of atom environment at given radius
        atom_identifiers_within_radius: list[int] = FindAtomEnvironmentOfRadiusN(
            mol=mol, radius=radius, rootedAtAtom=atom_identifier
        )
        atom_map: dict = {}

        sub_molecule: Mol = PathToSubmol(
            mol, atom_identifiers_within_radius, atomMap=atom_map
        )
        if atom_identifier not in atom_map:
            return None

        if self.include_chirality:
            smiles = MolToSmiles(
                sub_molecule,
                rootedAtAtom=atom_map[atom_identifier],
                isomericSmiles=True,
                canonical=True,
            )
            # preserve E/Z isomerism, remove chirality
            smiles = smiles.replace("[C@H]", "C").replace("[C@@H]", "C")
        else:
            smiles = MolToSmiles(
                sub_molecule,
                rootedAtAtom=atom_map[atom_identifier],
                isomericSmiles=False,  # following original MAP code
            )

        return smiles

    def _get_atom_pair_shingles(self, mol: Mol, atoms_envs: dict) -> set[bytes]:
        # get a list of atom shingles as SMILES, i.e. circular structures
        # around atom pairs and the length of shortest path
        atom_pairs: set[bytes] = set()

        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict: dict[str, int] = defaultdict(int)

        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = int(distance_matrix[idx1][idx2])

            for i in range(self.radius):
                env_a: str | None = atoms_envs[idx1][i]
                env_b: str | None = atoms_envs[idx2][i]

                shingle = self._make_shingle(env_a, env_b, dist)

                if self.include_duplicated_shingles:
                    shingle_dict[shingle] += 1
                    shingle += f"|{shingle_dict[shingle]}"

                atom_pairs.add(shingle.encode("utf-8"))

        return atom_pairs

    @staticmethod
    def _make_shingle(env_a: str | None, env_b: str | None, distance: int) -> str:
        env_a = env_a or ""
        env_b = env_b or ""

        if len(env_a) > len(env_b):
            larger_env: str = env_a
            smaller_env: str = env_b
        else:
            larger_env = env_b
            smaller_env = env_a

        shingle = f"{smaller_env}|{distance}|{larger_env}"
        return shingle

    @staticmethod
    def _hash_shingles(shinglings: set[bytes]) -> np.ndarray:
        if not shinglings:
            return np.empty(0, dtype=np.uint32)

        hashed_values = (
            struct.unpack("<I", sha256(shingling).digest()[:4])[0]
            for shingling in shinglings
        )

        return np.fromiter(
            hashed_values,
            dtype=np.uint32,
            count=len(shinglings),
        )

    def _fold(self, hashed_shinglings: np.ndarray) -> np.ndarray:
        folded = np.zeros(
            self.fp_size,
            dtype=np.uint32 if self.variant == "count" else np.uint8,
        )

        if hashed_shinglings.size == 0:
            return folded

        indices = hashed_shinglings % self.fp_size

        if self.variant == "count":
            np.add.at(folded, indices, 1)
        else:
            folded[indices] = 1

        return folded

    def _minhash(self, hashed_shinglings: np.ndarray) -> np.ndarray:
        # Return all-zero vector for empty shingle set
        if hashed_shinglings.size == 0:
            return np.zeros(self.fp_size, dtype=np.uint32)

        rng = np.random.default_rng(check_random_state(self.random_state))

        # Generate permutation parameters:
        # h_i(x) = (a_i * x + b_i) mod P
        a = rng.integers(
            1,
            self._MINHASH_PRIME,
            size=self.fp_size,
            dtype=np.uint64,
        )
        b = rng.integers(
            0,
            self._MINHASH_PRIME,
            size=self.fp_size,
            dtype=np.uint64,
        )

        x = hashed_shinglings.astype(np.uint64)

        # Apply all MinHash permutations to all hashed shingles at once.
        # Broadcasting yields an array of shape (n_shingles, fp_size), where
        # entry (j, i) is the value of permutation i applied to shingle j:
        #     h_i(x_j) = (a_i * x_j + b_i) mod P
        permuted = (
            x[:, np.newaxis] * a[np.newaxis, :] + b[np.newaxis, :]
        ) % self._MINHASH_PRIME
        mins = permuted.min(axis=0)

        # Store the sketch as uint32 to keep output compact and consistent.
        return mins.astype(np.uint32)
