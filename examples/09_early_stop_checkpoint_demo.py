"""
09_early_stop_checkpoint_demo.py — Early stopping ve checkpointing
=====================================================================

Uzun koşumlarda iki pratik özellik:

1. EARLY STOPPING — `early_stopping_patience=N`:
   Son N iterasyonda iyileşme `early_stopping_tol`'dan az ise koşum
   max_iter'e ulaşmadan durur. Result dict'inde `stopped_early=True` flag'i.
   3 modda da çalışır: classic, NSM, EPM.

2. CHECKPOINTING — `checkpoint_every=N` + `checkpoint_path='ckpt.pkl'`:
   Her N iter'de durum (rng + pop + fit + history + gbest) pickle olarak
   kaydedilir. `resume_from='ckpt.pkl'` ile sonraki çağrı oradan devam eder.
   Sadece classic mod (NSM/EPM v3.1'e bırakıldı).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import importlib.util

spec = importlib.util.spec_from_file_location(
    "metaheuristics_bk_v3",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "metaheuristics_bk_v3.py")
)
mh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mh)


def sphere(x):
    return float(np.sum(x ** 2))


if __name__ == "__main__":
    enc = mh.ContinuousEncoding([(-5, 5)] * 8)
    
    print("═══ Early Stopping ═══\n")
    
    print("A. early_stopping_patience yok (max_iter=200 sonuna kadar koş):")
    r = mh.run_bga(sphere, enc, pop_size=15, max_iter=200, seed=42)
    print(f"   iter koştu = {len(r['history'])-1}, stopped_early = {r['stopped_early']}")
    print(f"   best_fit   = {r['best_fit']:.8f}")
    
    print("\nB. early_stopping_patience=20 (yakınsama → erken dur):")
    for label, kw in [('Classic', {}), ('NSM', {'use_nsm': True}), ('EPM', {'use_epm': True})]:
        r = mh.run_bga(sphere, enc, pop_size=15, max_iter=200, seed=42,
                       early_stopping_patience=20, early_stopping_tol=1e-7, **kw)
        print(f"   {label:8s}: iter={len(r['history'])-1:>3}/200, "
              f"stopped={r['stopped_early']}, best_fit={r['best_fit']:.8f}")
    
    print("\n═══ Checkpointing ═══\n")
    
    ckpt = "/tmp/bga_demo_checkpoint.pkl"
    
    print(f"A. İlk koşum: 30 iter, her 10 iter'de checkpoint kaydet → {ckpt}")
    t0 = time.time()
    r1 = mh.run_bga(sphere, enc, pop_size=15, max_iter=30, seed=42,
                    checkpoint_every=10, checkpoint_path=ckpt, verbose=True)
    print(f"   süre = {time.time()-t0:.2f}s, best_fit = {r1['best_fit']:.8f}")
    
    print(f"\nB. Resume: aynı checkpoint'ten 50 iter daha (toplamda 80'e kadar)")
    t0 = time.time()
    r2 = mh.run_bga(sphere, enc, pop_size=15, max_iter=80, seed=42,
                    resume_from=ckpt, verbose=True)
    print(f"   süre = {time.time()-t0:.2f}s, best_fit = {r2['best_fit']:.8f}")
    print(f"   r2 history length = {len(r2['history'])} (resume'dan sonra devam etti)")
    
    # Temizle
    if os.path.exists(ckpt):
        os.remove(ckpt)
    
    print("\n═══ Pratik Senaryolar ═══\n")
    print(" 1. Pahalı fitness (XGBoost CV gibi):")
    print("    n_jobs=4 + early_stopping_patience=10 → 4x hızlı + erken dur")
    print()
    print(" 2. Uzun arama (cluster gece koşumu):")
    print("    checkpoint_every=50, max_iter=10000")
    print("    → Server crash olursa resume_from ile devam et")
    print()
    print(" 3. Hyperparameter tuning sweep'i:")
    print("    Aynı checkpoint'i farklı seed'lerle resume etmek mümkün değil")
    print("    (RNG state save edildi). Yeni seed için sıfırdan koş.")
