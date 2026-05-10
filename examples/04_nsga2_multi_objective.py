"""
Example 04: Multi-Objective Optimization (NSGA-II)
====================================================

İki çelişen objektif: hata vs özellik sayısı.
NSGA-II Pareto-optimal çözüm kümesini bulur.

Problem: Sentetik regression veride feature selection yap, ama hem 
RMSE'yi hem de seçilen özellik sayısını minimize et.

Çalıştır:
    python examples/04_nsga2_multi_objective.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from metaheuristics_bk_v3 import run_nsga2, BinaryEncoding


def main():
    # ─── Sentetik veri ─────────────────────────────────────────────────
    X, y = make_regression(n_samples=200, n_features=30, n_informative=8,
                            noise=15, random_state=42)
    print(f"Multi-Objective Feature Selection (NSGA-II)")
    print("=" * 60)
    print(f"  Veri: {X.shape[0]} örnek × {X.shape[1]} özellik")
    print(f"  Gerçek anlamlı özellik: 8")

    # ─── Inner CV ──────────────────────────────────────────────────────
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(kf.split(X))

    # ─── Multi-objective fitness ──────────────────────────────────────
    def multi_obj_fitness(sol):
        """İki objektif: (RMSE, n_features). İkisi de minimize."""
        sol = np.asarray(sol, dtype=int)
        if sol.sum() == 0:
            return np.array([1e6, 0.0])  # hiç özellik yoksa
        sel = np.where(sol == 1)[0]
        rmses = []
        for tr, te in folds:
            m = Ridge(alpha=1.0)
            m.fit(X[tr][:, sel], y[tr])
            rmses.append(np.sqrt(mean_squared_error(y[te], m.predict(X[te][:, sel]))))
        return np.array([np.mean(rmses), float(sol.sum())])

    # ─── NSGA-II koştur ────────────────────────────────────────────────
    print("\n  NSGA-II çalışıyor (pop=40, max_iter=60)...")
    encoding = BinaryEncoding(n_features=30)
    result = run_nsga2(
        fitness_fn=multi_obj_fitness,
        encoding=encoding,
        n_objectives=2,
        pop_size=40,
        max_iter=60,
        seed=42,
        verbose=True,
    )

    # ─── Pareto front'u göster ─────────────────────────────────────────
    print(f"\n✓ Tamamlandı")
    print(f"  Pareto front size: {len(result['pareto_front'])}")
    print(f"  Toplam değerlendirme: {result['n_evaluations']}")

    # f1 (RMSE) sırasına göre Pareto noktaları
    pf = result['pareto_fitness']
    sorted_idx = np.argsort(pf[:, 0])

    print(f"\n  Pareto-optimal trade-off noktaları:")
    print(f"  {'RMSE':>10s}  {'#features':>10s}  {'Açıklama':<40s}")
    print("  " + "─" * 65)
    for k, idx in enumerate(sorted_idx):
        rmse, nf = pf[idx]
        if k == 0:
            note = "← En düşük RMSE (çok özellik kullanır)"
        elif k == len(sorted_idx) - 1:
            note = "← En az özellik (RMSE yüksek)"
        else:
            note = ""
        print(f"  {rmse:>10.4f}  {int(nf):>10d}  {note}")

    # En iyi balanced trade-off (knee point)
    # Normalize edip min mesafe ideal point'e
    pf_norm = (pf - pf.min(axis=0)) / (pf.max(axis=0) - pf.min(axis=0) + 1e-10)
    distances = np.linalg.norm(pf_norm, axis=1)
    knee = sorted_idx[np.argmin(distances[sorted_idx])]
    print(f"\n  Knee point (en dengeli trade-off):")
    print(f"    RMSE = {pf[knee, 0]:.4f}, #features = {int(pf[knee, 1])}")
    sol = result['pareto_front'][knee].astype(int)
    print(f"    Seçilen özellik indeksleri: {np.where(sol==1)[0].tolist()}")


if __name__ == "__main__":
    main()
