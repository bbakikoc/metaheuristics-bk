"""
10_logging_autotune_demo.py — Logging ve Auto-Tuning
======================================================

v3-final-rc2 yeni:

1. LOGGING — `log_file='opt.csv'`:
   Her iter (ya da log_every kez) durumu CSV'ye yaz: iter, best_fit,
   mean_fit, std_fit, n_evaluations, elapsed_seconds. 4 modda da çalışır.

2. AUTO-TUNING — `tune_and_run()`:
   Algoritma × method × pop_size kombinasyonlarını küçük bütçeyle dener,
   en iyi konfigürasyonu kalan bütçeyle final koşar.
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


def rastrigin(x):
    A = 10.0
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))


if __name__ == "__main__":
    enc = mh.ContinuousEncoding([(-5.12, 5.12)] * 6)
    
    # ─── LOGGING ───
    print("═══ LOGGING ═══")
    log_path = "/tmp/rastrigin_optim.csv"
    
    r = mh.run_bga(rastrigin, enc, pop_size=20, max_iter=30, seed=42,
                   log_file=log_path, log_every=5, verbose=True)
    print(f"\n  best_fit = {r['best_fit']:.6f}")
    print(f"  CSV log:")
    with open(log_path) as f:
        for line in f:
            print(f"    {line.rstrip()}")
    os.remove(log_path)
    
    # ─── AUTO-TUNING ───
    print("\n═══ AUTO-TUNING ═══")
    print("3 algoritma × 2 method × 2 pop_size = 12 config, budget=3000\n")
    
    result = mh.tune_and_run(
        rastrigin, enc,
        algorithms=('bga', 'bcs', 'bsos'),
        methods=('classic', 'fdb'),
        pop_sizes=(15, 25),
        n_trials_per_config=2,
        budget=3000,
        seed=42,
        verbose=True,
    )
    
    print(f"\n→ En iyi konfigürasyon: {result['best_config']}")
    print(f"→ Final fitness:       {result['best_result']['best_fit']:.6f}")
    print(f"→ Toplam eval kullanımı: {result['total_evaluations']}")
    
    # En iyi 3 trial sonucu
    sorted_trials = sorted(result['trial_results'], key=lambda r: r['mean_fit'])
    print("\nEn iyi 3 trial konfigürasyonu:")
    for t in sorted_trials[:3]:
        print(f"  {t['alg']:5s} {t['method']:7s} ps={t['pop_size']:3d}: "
              f"mean_fit={t['mean_fit']:.6f}")
    
    print("\n═══ Pratik Senaryo ═══")
    print(" Yeni bir problemde hangi algoritmanın iyi olacağı bilinmiyor mu?")
    print(" tune_and_run ile 5-10 dakikalık trial faz, sonra otomatik final.")
    print(" Logging sonrası post-mortem analiz: best_fit eğrisi, std evolution.")
