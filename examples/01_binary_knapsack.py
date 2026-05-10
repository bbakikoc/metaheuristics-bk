"""
Example 01: Binary Encoding (Knapsack Problem)
================================================

v3'te Binary encoding ile çalışan klasik knapsack problemi.
v2 ile aynı, ama yeni Encoding API kullanılarak.

Çalıştır:
    python examples/01_binary_knapsack.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from metaheuristics_bk_v3 import (
    run_bga, run_bde, run_bpso,
    BinaryEncoding,
)


def main():
    np.random.seed(42)
    n_items = 20
    weights = np.random.randint(1, 30, size=n_items)
    values = np.random.randint(10, 100, size=n_items)
    capacity = weights.sum() // 2
    
    print("Knapsack Problem (v3 Binary Encoding)")
    print("=" * 55)
    print(f"  Eşya sayısı: {n_items}")
    print(f"  Kapasite:    {capacity}/{weights.sum()}")
    
    def knapsack_fitness(sol):
        sol = np.asarray(sol, dtype=int)
        w = (sol * weights).sum()
        if w > capacity:
            return 1e6
        return -(sol * values).sum()
    
    # v3 yeniliği: BinaryEncoding ile encoding-aware
    encoding = BinaryEncoding(n_features=n_items)
    
    print(f"\n  Encoding: {encoding}")
    print(f"\n  Algoritma karşılaştırması:")
    
    for name, runner in [('BGA', run_bga), ('BDE', run_bde), ('BPSO', run_bpso)]:
        res = runner(
            knapsack_fitness, encoding,
            method='fdb_argmax',
            pop_size=20, max_iter=30, seed=42,
        )
        sol = res['best_sol'].astype(int)
        weight = (sol * weights).sum()
        value = (sol * values).sum()
        print(f"    {name:5s}: değer={value:>4d}, ağırlık={weight:>3d}/{capacity}")


if __name__ == "__main__":
    main()
