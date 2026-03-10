import numpy as np
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
    map_fp = MAPFingerprint(verbose=0, n_jobs=-1)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
    )

    assert_equal(X_skfp, X_map)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp >= 0)


def test_map_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(n_jobs=-1)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=int,
    )

    assert_equal(X_skfp, X_map)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert np.issubdtype(X_skfp.dtype, np.integer)


def test_map_sparse_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(sparse=True, n_jobs=-1)
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
    map_fp = MAPFingerprint(include_duplicated_shingles=True, sparse=True, n_jobs=-1)
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
    )

    assert_equal(X_skfp.data, X_map.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), map_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data > 0)

    map_fp = MAPFingerprint(
        include_duplicated_shingles=True, sparse=True, count=True, n_jobs=-1
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


def test_map_chirality(smallest_mols_list):
    # smoke test, this should not throw an error
    map_fp = MAPFingerprint(include_chirality=True, n_jobs=-1)
    map_fp.transform(smallest_mols_list)


def test_map_chirality_uses_substructure():
    """Chirality mode should use sub_molecule SMILES, not full molecule SMILES."""
    # L-alanine and D-alanine are enantiomers with different CIP labels
    l_ala = MolFromSmiles("N[C@@H](C)C(=O)O")
    d_ala = MolFromSmiles("N[C@H](C)C(=O)O")

    map_fp = MAPFingerprint(include_chirality=True)
    fp_l = map_fp._calculate_single_mol_fingerprint(l_ala)
    fp_d = map_fp._calculate_single_mol_fingerprint(d_ala)

    # with chirality enabled, enantiomers should produce different fingerprints
    assert not np.array_equal(fp_l, fp_d)
