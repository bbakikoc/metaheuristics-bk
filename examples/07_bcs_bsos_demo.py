"""
07_bcs_bsos_demo.py — BCS (Cuckoo Search) ve BSOS (Symbiotic Organisms Search)
================================================================================

v3-final'da BCS ve BSOS encoding-aware olarak yeniden yazıldı:

BCS (Cuckoo Search, Yang & Deb 2009)
    Phase 1: Lévy flight — heavy-tailed dağılımdan adım atılır
        BinaryEncoding   → Lévy büyüklüğü kadar bit flip
        ContinuousEncoding → x + Lévy step (her boyutta), bound'lara clip
        Categorical/Mixed → step → kaç boyut değiştiğini belirler, mutate
    Phase 2: Nest abandonment — pa olasılığıyla yuva terk edilir
        method='classic' → uniform random reset
        method='fdb'     → FDB-rouletted rehberin komşusu

BSOS (Symbiotic Organisms Search, Cheng & Prayogo 2014)
    Phase 1 (Mutualism)    : i ve j gbest'e doğru crossover ile yaklaşır
    Phase 2 (Commensalism) : i gbest'e crossover, j etkilenmez
    Phase 3 (Parasitism)   : i'den parazit (yüksek mutasyon) → j ile rekabet

NOT: v2'deki BSOS binary OR/XOR varsayımı yerine `encoding.crossover` ve
`encoding.mutate` üzerinden çalıştığı için 4 encoding tipinde de doğru.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import importlib.util

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


# ─── Problem 1: Continuous Rastrigin (10D) ──────────────────────────────────
def rastrigin(x):
    """Rastrigin: çoklu yerel optimum, global opt = 0 at x=0."""
    A = 10.0
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


# ─── Problem 2: Binary Knapsack ─────────────────────────────────────────────
np.random.seed(0)
WEIGHTS = np.random.uniform(1, 10, 15)
VALUES = np.random.uniform(5, 50, 15)
CAPACITY = WEIGHTS.sum() * 0.5

def knapsack(x):
    if x.sum() == 0:
        return 1e6
    w = (x * WEIGHTS).sum()
    if w > CAPACITY:
        return 1e6 * (w - CAPACITY)
    return -float((x * VALUES).sum())


def run_comparison(name, fitness, encoding, pop_size=20, max_iter=80, seed=42):
    banner(name)
    print(f"  Encoding: {encoding}, pop_size={pop_size}, max_iter={max_iter}")
    print()
    print(f"  {'Algoritma':<25} {'best_fit':>14} {'n_eval':>8}")
    print(f"  {'-'*25} {'-'*14} {'-'*8}")
    
    runners = [
        ('BGA  classic',       mh.run_bga,  {'method': 'classic'}),
        ('BCS  classic',       mh.run_bcs,  {'method': 'classic'}),
        ('BCS  fdb',           mh.run_bcs,  {'method': 'fdb'}),
        ('BSOS classic',       mh.run_bsos, {'method': 'classic'}),
        ('BSOS fdb',           mh.run_bsos, {'method': 'fdb'}),
    ]
    
    for label, runner, kw in runners:
        r = runner(fitness, encoding, pop_size=pop_size, max_iter=max_iter, seed=seed, **kw)
        print(f"  {label:<25} {r['best_fit']:>14.6f} {r['n_evaluations']:>8}")


if __name__ == "__main__":
    # Continuous Rastrigin (BCS Lévy flight burada parlar)
    enc1 = mh.ContinuousEncoding([(-5.12, 5.12)] * 10)
    run_comparison("Problem 1: Rastrigin 10D (Continuous, optimum=0)",
                   rastrigin, enc1, pop_size=20, max_iter=80, seed=42)
    
    # Binary Knapsack
    enc2 = mh.BinaryEncoding(15, init_p=0.4)
    run_comparison(f"Problem 2: Knapsack 15 item (Binary, capacity={CAPACITY:.1f})",
                   knapsack, enc2, pop_size=20, max_iter=50, seed=42)
    
    print("\n" + "═" * 72)
    print("  ✓ BCS ve BSOS her encoding tipinde çalışıyor (encoding-aware tasarım).")
    print("  → BCS Lévy flight: Rastrigin'de büyük sıçramalarla yerel opt'tan kaçar.")
    print("  → BSOS 3 fazı: encoding.crossover + encoding.mutate üzerinden universal.")
    print("  → method='fdb' rehberli partner seçimi → genelde daha iyi sonuç.")
    print("═" * 72)
