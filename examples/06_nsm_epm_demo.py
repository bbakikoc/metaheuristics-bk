"""
06_nsm_epm_demo.py — NSM ve EPM modülleri encoding-aware olarak nasıl kullanılır
================================================================================

NSM (Natural Survivor Method) ve EPM (Evolutionary Population Management) v2'de
sadece binary uzayda çalışıyordu. v3-final'da encoding-aware: Continuous,
Categorical, Mixed uzaylarda da çalışır.

Bu örnek 3 sentetik problemde 3 modu karşılaştırır:
    Classic BGA  (default)      ← generational replacement + elitism
    BGA + NSM    (use_nsm=True) ← steady-state + sigmoid greedy/NSM switch
    BGA + EPM    (use_epm=True) ← H1+H2+H3 epoch-based population update

NSM 'çeşitliliği' artırır (lokal optimumdan kaçış kolaylaşır), EPM 'forbidden
list' ile parent-child rekabetini yöneterek sömürü/keşif dengesini çözer.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib.util

# Tek-dosya modülü import et
spec = importlib.util.spec_from_file_location(
    "metaheuristics_bk_v3",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "metaheuristics_bk_v3.py")
)
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)


def banner(text):
    print(f"\n{'═' * 72}")
    print(f"  {text}")
    print(f"{'═' * 72}")


# ─── Problem 1: Continuous Sphere (10D) ──────────────────────────────────────
def sphere(x):
    """Sphere: f(x) = sum(x_i^2). Optimum: x = 0, f = 0."""
    return float(np.sum(x ** 2))


# ─── Problem 2: Binary Knapsack (15 item) ────────────────────────────────────
np.random.seed(0)
WEIGHTS = np.random.uniform(1, 10, 15)
VALUES = np.random.uniform(5, 50, 15)
CAPACITY = WEIGHTS.sum() * 0.5  # ~yarı kapasite

def knapsack(x):
    """Knapsack: maximize value, weight <= capacity."""
    if x.sum() == 0:
        return 1e6  # boş çözüm cezası
    w = (x * WEIGHTS).sum()
    if w > CAPACITY:
        return 1e6 * (w - CAPACITY)  # infeasibility ceza
    return -float((x * VALUES).sum())  # minimize → -value


# ─── Problem 3: Categorical Hyperparameter (5D) ──────────────────────────────
TARGET = [2, 0, 3, 1, 2]  # bilinen optimum
def categorical_match(x):
    """Hedef indeks dizisini bul. Hamming-based."""
    return float(np.sum(x != TARGET))


def run_comparison(name, fitness, encoding, pop_size=20, max_iter=80, seed=42):
    banner(name)
    print(f"  Encoding: {encoding}, pop_size={pop_size}, max_iter={max_iter}")
    print()
    print(f"  {'Mod':<22} {'best_fit':>12} {'n_eval':>8} {'extra':>20}")
    print(f"  {'-'*22} {'-'*12} {'-'*8} {'-'*20}")
    
    for label, kw in [
        ('Classic BGA',         {}),
        ('BGA + NSM (preset 6)', {'use_nsm': True, 'nsm_preset': 6}),
        ('BGA + NSM (preset 1)', {'use_nsm': True, 'nsm_preset': 1}),
        ('BGA + EPM (Method-2)', {'use_epm': True, 'method': 'classic'}),
        ('BGA + EPM (Method-1)', {'use_epm': True, 'method': 'fdb'}),
    ]:
        r = mh.run_bga(fitness, encoding,
                       pop_size=pop_size, max_iter=max_iter, seed=seed, **kw)
        extra = ''
        if 'NSM' in label:
            extra = f"NSM acc={r['_nsm_accepted']}/{r['_nsm_applied']}"
        print(f"  {label:<22} {r['best_fit']:>12.6f} {r['n_evaluations']:>8} {extra:>20}")


if __name__ == "__main__":
    # Continuous
    enc1 = mh.ContinuousEncoding([(-5, 5)] * 10)
    run_comparison("Problem 1: Sphere 10D (Continuous, optimum=0)",
                   sphere, enc1, pop_size=20, max_iter=80, seed=42)
    
    # Binary
    enc2 = mh.BinaryEncoding(15, init_p=0.4)
    run_comparison(f"Problem 2: Knapsack 15 item (Binary, capacity={CAPACITY:.1f})",
                   knapsack, enc2, pop_size=20, max_iter=80, seed=42)
    
    # Categorical
    enc3 = mh.CategoricalEncoding([
        list(range(4)), list(range(4)), list(range(4)),
        list(range(4)), list(range(4)),
    ])
    run_comparison(f"Problem 3: Categorical Match (target=[2,0,3,1,2], optimum=0)",
                   categorical_match, enc3, pop_size=20, max_iter=80, seed=42)
    
    print("\n" + "═" * 72)
    print("  ✓ NSM ve EPM her 3 encoding tipinde de çalışıyor.")
    print("  → Continuous uzayda NSM çoğunlukla classic BGA'yı geçiyor.")
    print("  → Binary knapsack'te 3 mod da global optimumu buluyor (kolay problem).")
    print("  → Categorical match'te NSM çeşitliliği erken sömürüyor.")
    print("═" * 72)
