"""
Per-family speedup: time one descriptor family at a time on both sides.

Protocol is symmetric - each side computes the family *and its prerequisites* and
nothing else, so the ratio is not distorted by work shared with other families.
Single-threaded, BLAS pinned, no profiler.
"""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

import sys, time, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AddHs
RDLogger.DisableLog("rdApp.*")

N       = int(sys.argv[1]) if len(sys.argv) > 1 else 100
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DATASET = sys.argv[3] if len(sys.argv) > 3 else "bbbp"

if DATASET == "bbbp":
    from skfp.datasets.moleculenet import load_bbbp
    smiles, _ = load_bbbp()
else:
    from skfp.datasets.moleculenet import load_hiv
    smiles, _ = load_hiv()

rng = np.random.default_rng(0)
idx = rng.choice(len(smiles), size=min(N, len(smiles)), replace=False)
mols = [m for m in (Chem.MolFromSmiles(smiles[i]) for i in idx) if m is not None]
print(f"{len(mols)} molecules, mean atoms {np.mean([m.GetNumAtoms() for m in mols]):.1f}", flush=True)

# ---------------------------------------------------------------- ours
from skfp.fingerprints._new_mordred.descriptors import (
    autocorrelation, barysz_matrix, bcut, chi, detour_matrix, adjacency_matrix,
    extended_topochemical_atom, information_content, kappa_shape_index,
    molecular_distance_edge, molecular_id, path_count, walk_count, estate,
)
from skfp.fingerprints._new_mordred.descriptors import ring_count
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import AdjacencyMatrix, DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol, bonds_apply_func
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs
from rdkit.Chem import GetMolFrags
from rdkit.Chem.rdchem import Bond


def ours_chi(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return chi.calc(p, Subgraphs(p))

def ours_path_count(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return path_count.calc(p, Subgraphs(p))

def ours_kappa(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return kappa_shape_index.calc(p, Subgraphs(p))

def ours_ic(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    mh = AddHs(m); ph = AtomicProperties.with_hydrogens_added(mh, p)
    mk = preprocess_mol(Chem.Mol(m), kekulize=True, sanitize=False)
    bt = bonds_apply_func(Bond.GetBondType, mk, np.intp)
    return information_content.calc(mh, ph, bt)

def ours_molecular_id(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return molecular_id.calc(p, len(GetMolFrags(mol)))

def ours_autocorr(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    d = DistanceMatrix.from_mol(m)
    mh = AddHs(m); ph = AtomicProperties.with_hydrogens_added(mh, p)
    dh = DistanceMatrix.with_hydrogens_added(d, ph)
    return autocorrelation.calc(ph, dh)

def ours_detour(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    d = DistanceMatrix.from_mol(m)
    return detour_matrix.calc(p, d, ring_count.RingSets(m, p), len(GetMolFrags(mol)))

def ours_walk(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    a = AdjacencyMatrix(m)
    return walk_count.calc(p, np.linalg.eigh(a.matrix))

def ours_adjacency(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    a = AdjacencyMatrix(m)
    return adjacency_matrix.calc(p, len(GetMolFrags(mol)), a, np.linalg.eigh(a.matrix))

def ours_bcut(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return bcut.calc(p, len(GetMolFrags(mol)))

def ours_barysz(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return barysz_matrix.calc(p, len(GetMolFrags(mol)))

def ours_eta(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    d = DistanceMatrix.from_mol(m)
    mh = AddHs(m); ph = AtomicProperties.with_hydrogens_added(mh, p)
    mk = preprocess_mol(Chem.Mol(m), kekulize=True, sanitize=False)
    bt = bonds_apply_func(Bond.GetBondType, mk, np.intp)
    return extended_topochemical_atom.calc(bt, p, ph, d, ring_count.RingSets(m, p), len(GetMolFrags(mol)))

def ours_mde(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    return molecular_distance_edge.calc(p, AdjacencyMatrix(m), DistanceMatrix.from_mol(m))

def ours_estate(mol):
    m = preprocess_mol(mol); p = AtomicProperties.from_mol(m)
    d = DistanceMatrix.from_mol(m)
    return estate.calc(m, estate.calc_indices(p, d))

# ---------------------------------------------------------------- mordred
import mordred
from mordred import Calculator
from mordred import (Chi as MChi, PathCount as MPath, InformationContent as MIC,
                     MolecularId as MMolId, Autocorrelation as MAuto,
                     DetourMatrix as MDetour, WalkCount as MWalk,
                     AdjacencyMatrix as MAdj, BCUT as MBCUT,
                     BaryszMatrix as MBarysz, ExtendedTopochemicalAtom as META,
                     MolecularDistanceEdge as MMDE, EState as MEState,
                     KappaShapeIndex as MKappa)

FAMILIES = [
    ("Chi",                     MChi,    ours_chi),
    ("PathCount",               MPath,   ours_path_count),
    ("KappaShapeIndex",         MKappa,  ours_kappa),
    ("InformationContent",      MIC,     ours_ic),
    ("MolecularId",             MMolId,  ours_molecular_id),
    ("Autocorrelation",         MAuto,   ours_autocorr),
    ("DetourMatrix",            MDetour, ours_detour),
    ("WalkCount",               MWalk,   ours_walk),
    ("AdjacencyMatrix",         MAdj,    ours_adjacency),
    ("BCUT",                    MBCUT,   ours_bcut),
    ("BaryszMatrix",            MBarysz, ours_barysz),
    ("ExtendedTopochemicalAtom",METive := META, ours_eta),
    ("MolecularDistanceEdge",   MMDE,    ours_mde),
    ("EState",                  MEState, ours_estate),
]


def timeit(fn, mols, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for m in mols:
            fn(m)
        best = min(best, time.perf_counter() - t0)
    return best


rows = []
for name, mmod, ofn in FAMILIES:
    calc = Calculator(mmod, ignore_3D=True)
    n_desc = len(calc.descriptors)
    tm = timeit(lambda m: list(calc(m)), mols, repeats=REPEATS)
    to = timeit(ofn, mols, repeats=REPEATS)
    rows.append((name, n_desc, tm, to, tm / to))
    print(f"{name:26s} {n_desc:5d} feat   mordred {tm:8.3f} s   ours {to:8.3f} s   {tm/to:7.2f}x", flush=True)

# whole fingerprint, both sides
from skfp.fingerprints import NewMordredFingerprint
fp = NewMordredFingerprint(n_jobs=1, verbose=False)
to_all = timeit(lambda m: None, [], 1)
t0 = time.perf_counter(); fp.transform(mols); to_all = time.perf_counter() - t0
call = Calculator(mordred.descriptors, ignore_3D=True)
t0 = time.perf_counter()
for m in mols: list(call(m))
tm_all = time.perf_counter() - t0
print(f"\n{'WHOLE FINGERPRINT':26s} {len(call.descriptors):5d} feat   mordred {tm_all:8.3f} s   ours {to_all:8.3f} s   {tm_all/to_all:7.2f}x")
rows.append(("WHOLE FINGERPRINT", len(call.descriptors), tm_all, to_all, tm_all / to_all))

with open("family_speedup.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "n_features_mordred", "mordred_s", "ours_s", "speedup", "n_molecules", "dataset"])
    for r in rows:
        w.writerow([*r, len(mols), DATASET])
print("\nwritten family_speedup.csv")
