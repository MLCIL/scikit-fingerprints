import numpy as np
import pytest
from rdkit.Chem import FindAllSubgraphsOfLengthN, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.chi import (
    SUBGRAPH_TYPES,
    _class_mask,
)
from skfp.fingerprints._new_mordred.descriptors.path_count import _grow_paths
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.subgraphs import (
    SUBGRAPH_MAX_NUM_BONDS,
    Subgraphs,
)

# molecules whose subgraph counts stress the enumeration: empty and single-bond
# graphs, fused and bridged rings, spiro centres, and heavy branching
_SMILES = [
    "C",
    "[Na+]",
    "N#N",
    "CCO",
    "c1ccccc1",
    "C1CCCCC1",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "C1CC2CCC1CC2",
    "c1ccc2c(c1)ccc1ccccc12",
    "CC(C)(C)C(C)(C)CC(C)(C)C",
    "C1CCC2(CC1)CCCCC2",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "C1CC1CC1CC1",
]


def _subgraphs_of(smiles: str) -> Subgraphs:
    mol = preprocess_mol(MolFromSmiles(smiles))
    return Subgraphs(AtomicProperties.from_mol(mol))


def _paths_to_order(subgraphs: Subgraphs, order: int):
    """Paths of any order: enumerated up to the cap, grown by path_count past it."""
    if order <= SUBGRAPH_MAX_NUM_BONDS:
        return subgraphs.paths(order)
    paths = subgraphs.paths(SUBGRAPH_MAX_NUM_BONDS)
    for _ in range(SUBGRAPH_MAX_NUM_BONDS, order):
        paths = _grow_paths(paths, subgraphs)
    return paths


def _row_set(bond_idxs: np.ndarray) -> set[tuple[int, ...]]:
    bond_idxs = np.asarray(bond_idxs)
    if bond_idxs.size == 0:
        return set()
    return {tuple(row) for row in bond_idxs}


@pytest.mark.parametrize("smiles", _SMILES)
@pytest.mark.parametrize("order", range(1, SUBGRAPH_MAX_NUM_BONDS + 1))
def test_enumeration_matches_rdkit_per_order(smiles, order):
    """
    Cross-check the ESU recursion against RDKit, an independent implementation.

    This is the test that pins down ESU's correctness: getting the exclusive
    neighborhood or the root rule wrong loses or duplicates subgraphs, and either
    shows up here as a mismatch.
    """
    mol = preprocess_mol(MolFromSmiles(smiles))
    expected = {
        tuple(sorted(subgraph)) for subgraph in FindAllSubgraphsOfLengthN(mol, order)
    }
    assert _row_set(_subgraphs_of(smiles)._subgraph_bond_idxs(order)) == expected


@pytest.mark.parametrize("smiles", _SMILES)
@pytest.mark.parametrize("order", range(1, SUBGRAPH_MAX_NUM_BONDS + 1))
def test_subgraph_rows_are_ascending_and_correctly_shaped(smiles, order):
    """Rows must be sorted, since the path deduplication relies on it."""
    bond_idxs = np.asarray(_subgraphs_of(smiles)._subgraph_bond_idxs(order))
    assert bond_idxs.ndim == 2
    assert bond_idxs.shape[1] == order
    if bond_idxs.size:
        assert np.all(np.diff(bond_idxs, axis=1) > 0)


@pytest.mark.parametrize("smiles", _SMILES)
def test_classes_partition_the_subgraphs_of_each_order(smiles):
    """
    Every subgraph belongs to exactly one class, so the four class masks together
    account for each order's subgraphs, once each.
    """
    subgraphs = _subgraphs_of(smiles)
    for order in range(1, SUBGRAPH_MAX_NUM_BONDS + 1):
        num_subgraphs = len(np.asarray(subgraphs._subgraph_bond_idxs(order)))
        topology = subgraphs.topology(order)
        by_type = sum(
            int(_class_mask(topology, subgraph_type).sum())
            for subgraph_type in SUBGRAPH_TYPES
        )
        assert by_type == num_subgraphs, order


@pytest.mark.parametrize("smiles", _SMILES)
def test_atom_idxs_span_the_atoms_of_their_subgraphs(smiles):
    mol = preprocess_mol(MolFromSmiles(smiles))
    subgraphs = _subgraphs_of(smiles)
    for order in range(1, SUBGRAPH_MAX_NUM_BONDS + 1):
        num_subgraphs = len(np.asarray(subgraphs._subgraph_bond_idxs(order)))
        topology = subgraphs.topology(order)
        for subgraph_idx in range(num_subgraphs):
            atoms = topology.atom_idxs(subgraph_idx).tolist()
            assert len(set(atoms)) == len(atoms)  # no repeated atom
            assert all(0 <= atom < mol.GetNumAtoms() for atom in atoms)


@pytest.mark.parametrize("smiles", _SMILES)
def test_atom_products_multiply_over_each_subgraph(smiles):
    """The vectorized product must match multiplying a subgraph's atoms by hand."""
    mol = preprocess_mol(MolFromSmiles(smiles))
    subgraphs = _subgraphs_of(smiles)
    rng = np.random.default_rng(0)
    for order in range(1, SUBGRAPH_MAX_NUM_BONDS + 1):
        num_subgraphs = len(np.asarray(subgraphs._subgraph_bond_idxs(order)))
        topology = subgraphs.topology(order)
        values = rng.uniform(0.5, 2.0, size=(2, mol.GetNumAtoms()))
        products = topology.atom_products(values)
        assert products.shape == (2, num_subgraphs)
        for subgraph_idx in range(num_subgraphs):
            atoms = topology.atom_idxs(subgraph_idx)
            assert np.allclose(products[:, subgraph_idx], values[:, atoms].prod(axis=1))


@pytest.mark.parametrize("smiles", _SMILES)
def test_paths_are_self_avoiding_paths(smiles):
    """
    Paths above SUBGRAPH_MAX_NUM_BONDS are grown by :meth:`Subgraphs._grow_paths` rather than
    enumerated, so check the defining property directly: a path of ``order`` bonds
    spans ``order + 1`` distinct atoms, each interior atom joining exactly two of
    its bonds.
    """
    mol = preprocess_mol(MolFromSmiles(smiles))
    bond_ends = {
        bond.GetIdx(): (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    }
    subgraphs = _subgraphs_of(smiles)

    for order in range(1, SUBGRAPH_MAX_NUM_BONDS + 4):
        bond_idxs = np.asarray(_paths_to_order(subgraphs, order).bond_idxs)
        assert bond_idxs.ndim == 2
        assert bond_idxs.shape[1] == order

        rows = _row_set(bond_idxs)
        assert len(rows) == len(bond_idxs), f"duplicate paths at order {order}"

        for row in bond_idxs:
            degrees: dict[int, int] = {}
            for bond in row.tolist():
                for atom in bond_ends[bond]:
                    degrees[atom] = degrees.get(atom, 0) + 1
            # order + 1 atoms, and no atom used by more than two bonds: exactly a
            # self-avoiding, unbranched, acyclic path
            assert len(degrees) == order + 1, (order, row)
            assert max(degrees.values()) <= 2, (order, row)
            assert sorted(degrees.values()).count(1) == 2, (order, row)
