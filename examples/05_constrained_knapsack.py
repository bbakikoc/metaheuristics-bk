"""
Example 05: Constrained Optimization (Multi-Constraint Knapsack)
==================================================================

Klasik knapsack + ek kısıtlar:
  1. Toplam ağırlık ≤ kapasite
  2. En az 5 eşya seçilmeli (min count)
  3. Eşya 0 ve eşya 1'den en az biri seçilmeli (logical OR)

3 farklı constraint handling stratejisi karşılaştırılır.

Çalıştır:
    python examples/05_constrained_knapsack.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from metaheuristics_bk_v3 import (
    run_bga, BinaryEncoding,
    make_constrained_fitness, evaluate_constraints,
)


def main():
    # ─── Problem ────────────────────────────────────────────────────────
    np.random.seed(42)
    n_items = 15
    weights = np.random.randint(1, 25, size=n_items)
    values = np.random.randint(10, 80, size=n_items)
    capacity = 80
    min_count = 5

    print("Constrained Knapsack")
    print("=" * 60)
    print(f"  Eşya sayısı: {n_items}")
    print(f"  Ağırlıklar:  {weights.tolist()}")
    print(f"  Değerler:    {values.tolist()}")
    print(f"\n  Kısıtlar:")
    print(f"    1) Toplam ağırlık ≤ {capacity}")
    print(f"    2) En az {min_count} eşya seçilmeli")
    print(f"    3) Eşya[0] veya Eşya[1] mutlaka seçilmeli")

    # ─── Objektif (negatif çünkü min ediyoruz) ─────────────────────────
    def objective(sol):
        sol = np.asarray(sol, dtype=int)
        return -float((sol * values).sum())

    # ─── Kısıtlar (>0 = ihlal) ─────────────────────────────────────────
    def c1_weight(sol):
        sol = np.asarray(sol, dtype=int)
        return (sol * weights).sum() - capacity

    def c2_min_count(sol):
        sol = np.asarray(sol, dtype=int)
        return min_count - sol.sum()

    def c3_logical_or(sol):
        sol = np.asarray(sol, dtype=int)
        # En az biri 1 olmalı; ihlal: ikisi de 0
        return 1.0 if (sol[0] == 0 and sol[1] == 0) else -1.0

    constraints = [c1_weight, c2_min_count, c3_logical_or]

    # ─── 3 strateji karşılaştırması ────────────────────────────────────
    encoding = BinaryEncoding(n_features=n_items)

    print(f"\n  3 constraint handling stratejisi:")
    print("=" * 60)

    for method in ['feasibility_rules', 'death_penalty', 'penalty']:
        wrapped_fit = make_constrained_fitness(
            fitness_fn=objective,
            constraints=constraints,
            method=method,
        )
        result = run_bga(
            fitness_fn=wrapped_fit,
            encoding=encoding,
            method='fdb_argmax',
            pop_size=20, max_iter=40, seed=42,
        )
        sol = result['best_sol'].astype(int)
        info = evaluate_constraints(sol, constraints)
        value = (sol * values).sum()
        weight = (sol * weights).sum()
        n_selected = sol.sum()
        has_01 = sol[0] == 1 or sol[1] == 1

        print(f"\n  Strateji: {method}")
        print(f"    Çözüm: {sol.tolist()}")
        print(f"    Değer = {value}, ağırlık = {weight}/{capacity}, "
              f"eşya sayısı = {n_selected}")
        print(f"    C1 (ağırlık):  {'✓' if weight <= capacity else '✗ İHLAL'}")
        print(f"    C2 (≥{min_count} eşya): {'✓' if n_selected >= min_count else '✗ İHLAL'}")
        print(f"    C3 (0 veya 1): {'✓' if has_01 else '✗ İHLAL'}")
        print(f"    Feasible: {info['feasible']}")


if __name__ == "__main__":
    main()
