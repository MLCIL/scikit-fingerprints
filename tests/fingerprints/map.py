import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem import MolFromSmiles
from scipy.sparse import csr_array

from skfp.fingerprints import MAPFingerprint


def test_map_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(n_jobs=-1)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [
            map_fp._calculate_single_mol_fingerprint(mol) > 0
            for mol in smallest_mols_list
        ],
    )

    assert_equal(X_skfp, X_map)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_map_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="count",
        include_duplicated_shingles=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
    )

    assert_equal(X_skfp, X_map)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_map_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(variant="minhash", n_jobs=-1, random_state=0)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=int,
    )

    assert_equal(X_skfp, X_map)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert np.issubdtype(X_skfp.dtype, np.integer)


def test_map_sparse_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(variant="binary", sparse=True, n_jobs=-1)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [
            map_fp._calculate_single_mol_fingerprint(mol) > 0
            for mol in smallest_mols_list
        ],
    )

    assert_equal(X_skfp.data, X_map.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_map_sparse_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="count",
        include_duplicated_shingles=True,
        sparse=True,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
    )

    assert_equal(X_skfp.data, X_map.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_map_sparse_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=int,
    )

    assert_equal(X_skfp.data, X_map.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert np.issubdtype(X_skfp.dtype, np.integer)


def test_map_sparse_minhash_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="minhash",
        sparse=True,
        n_jobs=-1,
        random_state=0,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=np.uint32,
    )

    assert_equal(X_skfp.data, X_map.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.issubdtype(X_skfp.dtype, np.integer)


def test_map_minhash_same_random_state_is_reproducible(smallest_smiles_list):
    map_fp_1 = MAPFingerprint(variant="minhash", random_state=123, n_jobs=-1)
    map_fp_2 = MAPFingerprint(variant="minhash", random_state=123, n_jobs=-1)

    X_1 = map_fp_1.transform(smallest_smiles_list)
    X_2 = map_fp_2.transform(smallest_smiles_list)

    assert_equal(X_1, X_2)


def test_map_minhash_different_random_state_changes_output(smallest_smiles_list):
    map_fp_1 = MAPFingerprint(variant="minhash", random_state=123, n_jobs=-1)
    map_fp_2 = MAPFingerprint(variant="minhash", random_state=456, n_jobs=-1)

    X_1 = map_fp_1.transform(smallest_smiles_list)
    X_2 = map_fp_2.transform(smallest_smiles_list)

    assert not np.array_equal(X_1, X_2)


def test_map_minhash_is_independent_of_input_order_and_batch_size():
    smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CCO",
        "c1ccccc1",
        "CCN(CC)CC",
    ]

    map_fp = MAPFingerprint(variant="minhash", random_state=123, n_jobs=-1)

    X_full = map_fp.transform(smiles)

    # same molecules, different order
    reordered_indices = [2, 0, 3, 1]
    reordered_smiles = [smiles[i] for i in reordered_indices]
    X_reordered = map_fp.transform(reordered_smiles)

    # compare molecule-by-molecule, not row-by-row
    for original_idx, reordered_idx in enumerate(reordered_indices):
        assert_equal(X_full[reordered_idx], X_reordered[original_idx])

    # same molecules, smaller subsets / singleton calls
    for i, smi in enumerate(smiles):
        X_single = map_fp.transform([smi])
        assert_equal(X_full[i], X_single[0])

    X_subset = map_fp.transform(smiles[:2])
    assert_equal(X_full[:2], X_subset)


def test_map_binary_ignores_random_state(smallest_smiles_list):
    map_fp_1 = MAPFingerprint(variant="binary", random_state=123, n_jobs=-1)
    map_fp_2 = MAPFingerprint(variant="binary", random_state=456, n_jobs=-1)

    X_1 = map_fp_1.transform(smallest_smiles_list)
    X_2 = map_fp_2.transform(smallest_smiles_list)

    assert_equal(X_1, X_2)


def test_map_count_ignores_random_state(smallest_smiles_list):
    map_fp_1 = MAPFingerprint(variant="count", random_state=123, n_jobs=-1)
    map_fp_2 = MAPFingerprint(variant="count", random_state=456, n_jobs=-1)

    X_1 = map_fp_1.transform(smallest_smiles_list)
    X_2 = map_fp_2.transform(smallest_smiles_list)

    assert_equal(X_1, X_2)


def test_map_chirality(smallest_mols_list):
    # smoke test, this should not throw an error
    map_fp = MAPFingerprint(include_chirality=True, n_jobs=-1)
    map_fp.transform(smallest_mols_list)


def test_map_chirality_uses_substructure():
    # L-alanine and D-alanine are enantiomers with different CIP labels
    l_ala = MolFromSmiles("N[C@@H](C)C(=O)O")
    d_ala = MolFromSmiles("N[C@H](C)C(=O)O")

    map_fp = MAPFingerprint(include_chirality=True)
    fp_l = map_fp._calculate_single_mol_fingerprint(l_ala)
    fp_d = map_fp._calculate_single_mol_fingerprint(d_ala)

    # with chirality enabled, enantiomers should produce different fingerprints
    assert not np.array_equal(fp_l, fp_d)


@pytest.mark.parametrize(
    "random_state",
    [0, np.random.RandomState(0), None],
    ids=["int", "RandomState", "None"],
)
def test_map_fp_random_state_types(smallest_smiles_list, random_state):
    """MAPFingerprint should accept int, RandomState, or None."""
    fp = MAPFingerprint(random_state=random_state, n_jobs=-1)
    X = fp.transform(smallest_smiles_list)
    assert X.shape[0] == len(smallest_smiles_list)
