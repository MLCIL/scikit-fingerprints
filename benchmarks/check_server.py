"""
Rozpoznanie serwera przed przebiegiem benchmarku.

Odpowiada na pytania, ktore trzeba znac PRZED odpaleniem 12-godzinnego liczenia:
  - ile rdzeni realnie mam do dyspozycji (nie ile widzi system!)
  - czy maszyna jest pusta, czy ktos juz na niej liczy
  - czy srodowisko jest kompletne (skfp na wlasciwym branchu, mordred, matplotlib)
  - jak szybki jest ten procesor na jednym rdzeniu

Nic nie zapisuje i nic nie psuje - tylko czyta i wypisuje.

    python check_server.py              # samo rozpoznanie, kilka sekund
    python check_server.py --probe      # + szybki pomiar wydajnosci (~1 min)
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def line(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def read(path: str) -> str:
    try:
        return Path(path).read_text()
    except Exception:
        return ""


# --------------------------------------------------------------------------
def cpu_info() -> dict:
    info = {"model": platform.processor() or "?", "logical": os.cpu_count() or 0}

    cpuinfo = read("/proc/cpuinfo")
    if cpuinfo:
        for ln in cpuinfo.splitlines():
            if ln.startswith("model name"):
                info["model"] = ln.split(":", 1)[1].strip()
                break
        # fizyczne rdzenie: unikalne pary (physical id, core id)
        phys, cur = set(), {}
        for ln in cpuinfo.splitlines():
            if ":" in ln:
                k, v = (x.strip() for x in ln.split(":", 1))
                if k in ("physical id", "core id"):
                    cur[k] = v
                    if len(cur) == 2:
                        phys.add((cur["physical id"], cur["core id"]))
                        cur = {}
        if phys:
            info["physical"] = len(phys)

    # ILE RDZENI NAPRAWDE MOGE UZYC - to jest wazniejsze niz cpu_count()
    try:
        info["affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity"] = info["logical"]

    return info


def cgroup_limit() -> str | None:
    """Kontener/cgroup potrafi ograniczyc CPU ponizej liczby rdzeni maszyny."""
    v2 = read("/sys/fs/cgroup/cpu.max").split()
    if len(v2) == 2 and v2[0] != "max":
        return f"{int(v2[0]) / int(v2[1]):.1f} rdzenia (cgroup v2)"
    quota = read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").strip()
    period = read("/sys/fs/cgroup/cpu/cpu.cfs_period_us").strip()
    if quota and period and quota != "-1":
        return f"{int(quota) / int(period):.1f} rdzenia (cgroup v1)"
    return None


def memory_info() -> str:
    meminfo = read("/proc/meminfo")
    if meminfo:
        total = avail = None
        for ln in meminfo.splitlines():
            if ln.startswith("MemTotal:"):
                total = int(ln.split()[1]) / 1024 / 1024
            elif ln.startswith("MemAvailable:"):
                avail = int(ln.split()[1]) / 1024 / 1024
        if total:
            return f"{total:.1f} GB lacznie, {avail:.1f} GB wolne" if avail else f"{total:.1f} GB"
    try:
        import psutil

        m = psutil.virtual_memory()
        return f"{m.total / 1e9:.1f} GB lacznie, {m.available / 1e9:.1f} GB wolne"
    except ImportError:
        return "nieznane (brak /proc/meminfo i psutil)"


def load_info(cores: int) -> tuple[str, bool]:
    """Zwraca opis obciazenia i czy maszyna wyglada na wolna."""
    try:
        l1, l5, l15 = os.getloadavg()
    except (AttributeError, OSError):
        return "nieznane (brak getloadavg - pewnie Windows)", True
    per_core = l1 / cores if cores else l1
    verdict = "PUSTA" if per_core < 0.15 else ("LEKKO OBCIAZONA" if per_core < 0.5 else "ZAJETA")
    return (f"load average {l1:.2f} / {l5:.2f} / {l15:.2f} "
            f"({per_core * 100:.0f}% na rdzen) -> {verdict}"), per_core < 0.15


def other_users() -> str:
    """Kto jeszcze liczy na tej maszynie."""
    if not shutil.which("ps"):
        return "nie sprawdzono (brak ps)"
    try:
        out = subprocess.run(
            ["ps", "-eo", "user,pcpu,comm", "--sort=-pcpu"],
            capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()[1:]
    except Exception:
        return "nie sprawdzono"

    me = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    heavy = []
    for ln in out[:40]:
        parts = ln.split(None, 2)
        if len(parts) == 3:
            user, pcpu, comm = parts
            try:
                if float(pcpu) > 20.0:
                    heavy.append((user, float(pcpu), comm.strip()))
            except ValueError:
                pass
    if not heavy:
        return "brak procesow zjadajacych >20% CPU"
    lines = [f"    {u:12s} {c:6.1f}% {cmd[:40]}{'  <- TY' if u == me else ''}"
             for u, c, cmd in heavy[:8]]
    return f"{len(heavy)} procesow >20% CPU:\n" + "\n".join(lines)


def scheduler_hints() -> list[str]:
    """Czy to wezel klastra ze schedulerem."""
    hints = []
    for var, label in [
        ("SLURM_JOB_ID", "SLURM job"),
        ("SLURM_CPUS_ON_NODE", "SLURM rdzenie na wezle"),
        ("SLURM_CPUS_PER_TASK", "SLURM rdzenie na zadanie"),
        ("PBS_JOBID", "PBS job"),
        ("OMP_NUM_THREADS", "OMP_NUM_THREADS"),
        ("MKL_NUM_THREADS", "MKL_NUM_THREADS"),
    ]:
        if os.environ.get(var):
            hints.append(f"{label} = {os.environ[var]}")
    return hints


def environment() -> None:
    line("SRODOWISKO")
    print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
    for mod in ("numpy", "pandas", "rdkit", "matplotlib", "mordred", "skfp"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "")
            print(f"  {mod:11s} OK {ver}")
        except ImportError:
            need = "WYMAGANE" if mod in ("rdkit", "mordred", "skfp", "numpy") else "do wykresow"
            print(f"  {mod:11s} BRAK  <- {need}")

    try:
        import skfp

        repo = Path(skfp.__file__).resolve().parent.parent
        run = lambda a: subprocess.run(a, capture_output=True, text=True, cwd=repo).stdout.strip()
        branch, sha = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]), run(["git", "rev-parse", "--short", "HEAD"])
        dirty = "  (NIEZACOMMITOWANE ZMIANY)" if run(["git", "status", "--porcelain"]) else ""
        print(f"\n  skfp repo   {repo}")
        print(f"  branch      {branch}@{sha}{dirty}")
        print("  ^ SPRAWDZ, czy to branch benchmarkowy (opt-17 + kappa + IC)")
    except Exception as e:
        print(f"\n  nie udalo sie odczytac brancha skfp: {e}")


def probe(n: int = 100) -> None:
    """Szybki pomiar wydajnosci jednego rdzenia."""
    line(f"POMIAR WYDAJNOSCI (1 rdzen, {n} czasteczek)")
    try:
        import warnings

        warnings.filterwarnings("ignore")
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        from skfp.datasets.moleculenet import load_bbbp
        from skfp.fingerprints import NewMordredFingerprint
    except ImportError as e:
        print(f"  pominieto: {e}")
        return

    smiles = list(load_bbbp(as_frame=True)["SMILES"])[: n * 2]
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None][:n]

    fp = NewMordredFingerprint(n_jobs=1, verbose=False)
    fp.transform(mols[:3])
    t0 = time.perf_counter()
    fp.transform(mols)
    dt = time.perf_counter() - t0
    rate = len(mols) / dt
    print(f"  nasze   : {rate:7.2f} mol/s  -> 10k czasteczek w {10_000 / rate / 60:6.1f} min")

    try:
        from mordred import Calculator, descriptors

        calc = Calculator(descriptors, ignore_3D=True)
        for _ in calc.map(mols[:3], nproc=1, quiet=True):
            pass
        t0 = time.perf_counter()
        c = sum(1 for _ in calc.map(mols, nproc=1, quiet=True))
        dtm = time.perf_counter() - t0
        rm = c / dtm
        print(f"  mordred : {rm:7.2f} mol/s  -> 10k czasteczek w {10_000 / rm / 60:6.1f} min")
        print(f"\n  speedup : {rate / rm:.2f}x")
        print(f"  odniesienie z wykresu kolegi: mordred = 190 s / 10k "
              f"(tu wychodzi {10_000 / rm:.0f} s)")
    except ImportError:
        print("  mordred : BRAK - nie ma z czym porownac")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", action="store_true", help="dodaj pomiar wydajnosci (~1 min)")
    p.add_argument("--probe-size", type=int, default=100)
    args = p.parse_args()

    line("MASZYNA")
    print(f"  host        {platform.node()}")
    print(f"  system      {platform.system()} {platform.release()}")
    cpu = cpu_info()
    print(f"  procesor    {cpu['model']}")
    if "physical" in cpu:
        print(f"  rdzenie     {cpu['physical']} fizycznych / {cpu['logical']} logicznych")
    else:
        print(f"  rdzenie     {cpu['logical']} logicznych")
    print(f"  DOSTEPNE MI {cpu['affinity']}  <- tej liczby uzywaj jako n_jobs")
    if cpu["affinity"] != cpu["logical"]:
        print(f"              UWAGA: system widzi {cpu['logical']}, "
              f"ale masz przydzielone {cpu['affinity']}")
    lim = cgroup_limit()
    if lim:
        print(f"  limit CPU   {lim}  <- kontener ogranicza!")
    print(f"  pamiec      {memory_info()}")

    hints = scheduler_hints()
    if hints:
        print("\n  scheduler / zmienne srodowiskowe:")
        for h in hints:
            print(f"    {h}")

    line("OBCIAZENIE")
    desc, idle = load_info(cpu["affinity"])
    print(f"  {desc}")
    print(f"  {other_users()}")
    print()
    if idle:
        print("  -> Maszyna wyglada na wolna. Faze B (pomiary bezwzgledne:")
        print("     zestawy 'threads' i 'reference') mozna robic teraz.")
    else:
        print("  -> Maszyna jest zajeta. Faze A (ilorazy) mozna puscic,")
        print("     ale zestawy 'threads' i 'reference' PRZELOZ - pod obciazeniem")
        print("     wyjda zanizone i nieporownywalne z wykresem kolegi.")

    environment()

    if args.probe:
        probe(args.probe_size)

    line("CO DALEJ")
    print("  1. python bench_run.py --suite calib --out calib.csv    <- projekcja budzetu")
    print(f"  2. faza A rownolegle (mozesz puscic ~{max(1, cpu['affinity'] // 2)} procesow naraz)")
    print("  3. faza B na pustej maszynie")
    print("  4. python bench_analyze.py --inputs \"res_*.csv\" --outdir figures")


if __name__ == "__main__":
    main()
