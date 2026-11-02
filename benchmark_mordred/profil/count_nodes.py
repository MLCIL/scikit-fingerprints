import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import warnings; warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
import mordred
from mordred import descriptors as MD
from mordred import Calculator
from mordred._base.calculator import Calculator as C

calc = Calculator(MD, ignore_3D=True)
print("registered 2D descriptors:", len(calc.descriptors))

orig = C._calculate_one
counter = {"n": 0}
def patched(self, cxt, desc, reset):
    counter["n"] += 1
    return orig(self, cxt, desc, reset)
C._calculate_one = patched

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
list(calc(mol))
print("aspirin: _calculate_one calls:", counter["n"], " unique cached nodes:", len(calc._cache))

counter["n"] = 0
mol2 = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")  # caffeine
list(calc(mol2))
print("caffeine: _calculate_one calls:", counter["n"], " unique cached nodes:", len(calc._cache))

counter["n"] = 0
mol3 = Chem.MolFromSmiles("c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1")  # pentacene
list(calc(mol3))
print("pentacene: _calculate_one calls:", counter["n"], " unique cached nodes:", len(calc._cache))
