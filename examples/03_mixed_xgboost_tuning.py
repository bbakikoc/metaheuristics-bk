"""
Example 03: Mixed Encoding — XGBoost Hyperparameter Optimization
==================================================================

v3'ün gerçek-dünya senaryosu: XGBoost hyperparameter'larını optimize et.
Continuous (learning_rate), Categorical (max_depth), Binary (use_subsample) 
karma yapıda — sadece v3 ile mümkün.

Çalıştır:
    python examples/03_mixed_xgboost_tuning.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from metaheuristics_bk_v3 import (
    run_bga,
    BinaryEncoding, ContinuousEncoding, CategoricalEncoding, MixedEncoding,
)


def main():
    # ─── Sentetik regression problemi ──────────────────────────────────
    X, y = make_regression(n_samples=300, n_features=20, n_informative=10,
                            noise=10, random_state=42)
    
    print("XGBoost Hyperparameter Tuning (v3 Mixed Encoding)")
    print("=" * 65)
    print(f"  Veri: {X.shape[0]} örnek × {X.shape[1]} özellik")
    
    # ─── v3 Mixed encoding ─────────────────────────────────────────────
    # 4 hyperparameter'ı 3 farklı tipte kodla:
    encoding = MixedEncoding({
        'learning_rate': ContinuousEncoding(
            bounds=[(0.01, 0.3)],
            log_scale=[True],  # log-uniform örnekleme
        ),
        'max_depth': CategoricalEncoding([
            [3, 5, 7, 10, 15],
        ]),
        'subsample': ContinuousEncoding(
            bounds=[(0.5, 1.0)],
        ),
        'use_l2_reg': BinaryEncoding(n_features=1, ensure_one=False),
    })
    
    print(f"\n  Hyperparameter uzayı:")
    print(f"    learning_rate ∈ [0.01, 0.3]   (log-uniform)")
    print(f"    max_depth     ∈ {{3, 5, 7, 10, 15}}")
    print(f"    subsample     ∈ [0.5, 1.0]")
    print(f"    use_l2_reg    ∈ {{0, 1}}")
    
    # ─── Custom fitness: 3-fold CV RMSE ────────────────────────────────
    try:
        import xgboost as xgb
    except ImportError:
        print("\n⚠ XGBoost kurulu değil, sentetik fitness ile devam edilecek.")
        print("  pip install xgboost ile kurabilirsin")
        # Gerçekçi olmasın diye basit fitness:
        def fitness_fn(sol):
            score = 0
            score += abs(np.log(sol['learning_rate'][0] / 0.05)) * 2  # ~0.05 ideal
            score += abs([3,5,7,10,15][sol['max_depth'][0]] - 7) * 0.1
            score += abs(sol['subsample'][0] - 0.8) * 2
            score -= sol['use_l2_reg'][0] * 0.3
            return float(score)
    else:
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        folds = list(kf.split(X))
        
        def fitness_fn(sol):
            depth_choices = [3, 5, 7, 10, 15]
            params = {
                'n_estimators': 50,
                'learning_rate': float(sol['learning_rate'][0]),
                'max_depth':     depth_choices[int(sol['max_depth'][0])],
                'subsample':     float(sol['subsample'][0]),
                'reg_lambda':    1.0 if sol['use_l2_reg'][0] == 1 else 0.0,
                'random_state':  42,
                'verbosity':     0,
                'n_jobs':        -1,
            }
            rmses = []
            for tr, te in folds:
                m = xgb.XGBRegressor(**params)
                m.fit(X[tr], y[tr])
                rmses.append(np.sqrt(mean_squared_error(y[te], m.predict(X[te]))))
            return float(np.mean(rmses))
    
    # ─── BGA ile hyperparameter ara ────────────────────────────────────
    print(f"\n  BGA + FDB-argmax çalışıyor (pop=15, iter=20)...")
    res = run_bga(
        fitness_fn=fitness_fn,
        encoding=encoding,
        method='fdb_argmax',
        pop_size=15,
        max_iter=20,
        seed=42,
        verbose=True,
    )
    
    # ─── Sonuçları decode et ───────────────────────────────────────────
    sol = res['best_sol']
    depth_choices = [3, 5, 7, 10, 15]
    print(f"\n✓ En iyi hyperparameter'lar:")
    print(f"    learning_rate = {sol['learning_rate'][0]:.5f}")
    print(f"    max_depth     = {depth_choices[int(sol['max_depth'][0])]}")
    print(f"    subsample     = {sol['subsample'][0]:.3f}")
    print(f"    use_l2_reg    = {bool(sol['use_l2_reg'][0])}")
    print(f"\n  Best fitness (CV RMSE): {res['best_fit']:.4f}")
    print(f"  Toplam değerlendirme:   {res['n_evaluations']}")


if __name__ == "__main__":
    main()
