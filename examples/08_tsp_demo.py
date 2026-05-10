"""
08_tsp_demo.py — PermutationEncoding ile TSP optimizasyonu
============================================================

⏱️ Çalışma süresi: ~10-15 saniye (9 algoritma kıyaslaması).

PermutationEncoding (v3-final) permütasyon-koruyucu operatörlerle çalışır:
    init     → np.arange(n) shuffle
    crossover→ PMX (Partially Matched Crossover)
    mutate   → swap mutation
    distance → Kendall tau distance (sıralı ikili sayısı)

Bu örnek 20-şehirli rastgele bir TSP problemi üzerinde 9 algoritmayı
karşılaştırır. Permütasyon uzayında BGA, BCS, BSOS, BABC çalışır
(BPSO ve BGWO continuous-yerel olduğu için Permutasyon desteklemiyor).

Çıktı olarak optimal tur'un şehir sırasını ve uzunluğunu verir.
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


# ─── 20-şehirli rastgele TSP ─────────────────────────────────────────────
N_CITIES = 20
np.random.seed(123)
CITIES = np.random.uniform(0, 100, (N_CITIES, 2))


def tour_length(perm):
    """Kapalı tur uzunluğu (son şehirden ilkine geri dönüş dahil)."""
    total = 0.0
    for i in range(len(perm)):
        a = CITIES[perm[i]]
        b = CITIES[perm[(i + 1) % len(perm)]]
        total += np.linalg.norm(a - b)
    return float(total)


def random_tour_baseline(seed=42, n_trials=1000):
    """Rastgele turların ortalamasını ve en iyisini bul (baseline)."""
    rng = np.random.RandomState(seed)
    best = float('inf')
    sum_len = 0.0
    for _ in range(n_trials):
        p = rng.permutation(N_CITIES)
        L = tour_length(p)
        sum_len += L
        if L < best:
            best = L
    return sum_len / n_trials, best


def main():
    print(f"═══ TSP {N_CITIES} şehir — PermutationEncoding demo ═══\n")
    
    # Baseline: rastgele turlar
    avg_random, best_random = random_tour_baseline(seed=42, n_trials=1000)
    print(f"Baseline (1000 rastgele tur):")
    print(f"  Ortalama uzunluk: {avg_random:.2f}")
    print(f"  En iyi rastgele:  {best_random:.2f}")
    print()
    
    # PermutationEncoding ile algoritma karşılaştırması
    enc = mh.PermutationEncoding(n_items=N_CITIES)
    
    print(f"Algoritmalar (pop_size=15, max_iter=50):")
    print(f"  {'Algoritma':<25} {'best_fit':>10} {'iyileşme %':>11}")
    print(f"  {'-'*25} {'-'*10} {'-'*11}")
    
    runners = [
        ('BGA classic',         mh.run_bga,  {'method': 'classic'}),
        ('BGA fdb',             mh.run_bga,  {'method': 'fdb'}),
        ('BGA + NSM (preset 6)', mh.run_bga, {'use_nsm': True, 'nsm_preset': 6}),
        ('BGA + EPM (Method-2)', mh.run_bga, {'use_epm': True, 'method': 'classic'}),
        ('BCS classic',         mh.run_bcs,  {'method': 'classic'}),
        ('BCS fdb',             mh.run_bcs,  {'method': 'fdb'}),
        ('BSOS classic',        mh.run_bsos, {'method': 'classic'}),
        ('BSOS fdb',            mh.run_bsos, {'method': 'fdb'}),
        ('BABC fdb',            mh.run_babc, {'method': 'fdb'}),
    ]
    
    best_overall = float('inf')
    best_alg = ''
    best_perm = None
    for label, runner, kw in runners:
        r = runner(tour_length, enc, pop_size=15, max_iter=50, seed=42, **kw)
        improvement = (avg_random - r['best_fit']) / avg_random * 100.0
        print(f"  {label:<25} {r['best_fit']:>10.2f} {improvement:>10.1f}%")
        if r['best_fit'] < best_overall:
            best_overall = r['best_fit']
            best_alg = label
            best_perm = r['best_sol']
    
    print()
    print(f"En iyi sonuç: {best_alg} → {best_overall:.2f}")
    print(f"  Tur sırası: {best_perm.tolist()}")
    print(f"  İyileşme (rasgele ortalamaya göre): "
          f"{(avg_random - best_overall) / avg_random * 100:.1f}%")
    print(f"  İyileşme (rasgele en iyiye göre):   "
          f"{(best_random - best_overall) / best_random * 100:.1f}%")


if __name__ == "__main__":
    main()
