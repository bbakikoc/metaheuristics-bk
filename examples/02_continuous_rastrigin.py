"""
Example 02: Continuous Optimization (Rastrigin Function)
==========================================================

v3'te ContinuousEncoding ile sürekli fonksiyon optimizasyonu.
v1 ve v2'de bu mümkün değildi — v3'ün ana yeniliği.

Rastrigin: çok-modlu, çok sayıda lokal minimum içeren klasik 
benchmark fonksiyonu. Global min: f(0) = 0.

Çalıştır:
    python examples/02_continuous_rastrigin.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from metaheuristics_bk_v3 import (
    run_bga, run_bde, run_bpso,
    ContinuousEncoding,
)


def rastrigin(x):
    """Rastrigin function: çok-modlu, global min = 0 at x=0."""
    A = 10
    return float(A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x)))


def sphere(x):
    """Sphere function: tek-modlu, global min = 0 at x=0."""
    return float(np.sum(x**2))


def rosenbrock(x):
    """Rosenbrock function: dar bir vadi, global min = 0 at x=1."""
    return float(np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))


def main():
    n_dim = 10
    bounds = [(-5.12, 5.12)] * n_dim
    encoding = ContinuousEncoding(bounds=bounds)
    
    functions = {
        'Sphere':     (sphere,     "global min = 0 at x=0"),
        'Rastrigin':  (rastrigin,  "global min = 0 at x=0 (multi-modal)"),
        'Rosenbrock': (rosenbrock, "global min = 0 at x=1 (banana)"),
    }
    
    print(f"Continuous Optimization (dim={n_dim}, encoding={encoding})")
    print("=" * 70)
    
    for fn_name, (fn, info) in functions.items():
        print(f"\n  {fn_name}: {info}")
        for alg_name, runner in [('BGA', run_bga), ('BDE', run_bde), ('BPSO', run_bpso)]:
            res = runner(
                fn, encoding,
                method='fdb_argmax',
                pop_size=30, max_iter=100, seed=42,
            )
            print(f"    {alg_name:5s}: best_fit = {res['best_fit']:>10.4f}  "
                  f"(n_eval={res['n_evaluations']})")


if __name__ == "__main__":
    main()
