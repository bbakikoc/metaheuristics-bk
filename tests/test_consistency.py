"""
v3-final Consistency Test
==========================

5 algoritma × 4 encoding + NSGA-II + 3 constraint method
+ NSM/EPM × 4 encoding + reproducibility.

Çalıştır:
    python tests/test_consistency.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import metaheuristics_bk_v3 as mh


# Top-level pickle-able fitness fonksiyonu (parallel test için zorunlu)
def _parallel_test_fitness(x):
    """Sphere fitness — top-level def, multiprocessing.Pool'a pickle-able."""
    return float(np.sum(x ** 2))


def test_all_algorithm_encoding_combos():
    """7 algoritma × 4 encoding = 28 kombinasyon (4 NotImplementedError beklenir).
    
    BPSO ve BGWO Categorical/Mixed desteklemiyor (continuous-yerel mantık).
    """
    print('═══ 1. 7 Algoritma × 4 Encoding ═══')
    
    # Test verileri
    bin_enc = mh.BinaryEncoding(n_features=10)
    cont_enc = mh.ContinuousEncoding(bounds=[(-3, 3)] * 5)
    cat_enc = mh.CategoricalEncoding([['a','b','c'], [1, 2, 3]])
    mix_enc = mh.MixedEncoding({
        'lr': mh.ContinuousEncoding([(0.01, 0.3)]),
        'depth': mh.CategoricalEncoding([[3, 5, 7]]),
    })
    
    fns = {
        'Binary': (bin_enc, lambda s: -float(np.asarray(s, dtype=int).sum())),
        'Continuous': (cont_enc, lambda x: float(np.sum(x**2))),
        'Categorical': (cat_enc, lambda s: float(np.sum(s))),
        'Mixed': (mix_enc, lambda s: abs(s['lr'][0] - 0.1) + abs(s['depth'][0] - 1)),
    }
    
    algos = {
        'BGA': mh.run_bga, 'BDE': mh.run_bde, 'BPSO': mh.run_bpso,
        'BABC': mh.run_babc, 'BGWO': mh.run_bgwo,
        'BCS': mh.run_bcs, 'BSOS': mh.run_bsos,
    }
    
    ok, skipped = 0, 0
    for alg_name, runner in algos.items():
        for enc_name, (enc, fn) in fns.items():
            try:
                # BCS ve BSOS 'fdb_argmax' desteklemiyor, sadece 'classic'/'fdb'
                m = 'fdb' if alg_name in ('BCS', 'BSOS') else 'fdb_argmax'
                res = runner(fn, enc, method=m,
                            pop_size=10, max_iter=8, seed=42)
                assert 'best_sol' in res
                print(f"  ✓ {alg_name:5s} + {enc_name:12s}: fit={res['best_fit']:.3f}")
                ok += 1
            except NotImplementedError:
                print(f"  - {alg_name:5s} + {enc_name:12s}: skipped (NotImplemented)")
                skipped += 1
            except Exception as e:
                print(f"  ✗ {alg_name:5s} + {enc_name:12s}: {type(e).__name__}: {e}")
    
    return ok, skipped


def test_nsga2():
    """NSGA-II ile ZDT1 multi-objective."""
    print('\n═══ 2. NSGA-II Multi-Objective ═══')
    
    def zdt1(x):
        f1 = float(x[0])
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - np.sqrt(f1 / g) if g > 0 else 0
        return np.array([f1, float(g * h)])
    
    enc = mh.ContinuousEncoding(bounds=[(0, 1)] * 5)
    res = mh.run_nsga2(zdt1, enc, n_objectives=2,
                       pop_size=20, max_iter=30, seed=42)
    
    assert 'pareto_front' in res
    assert 'pareto_fitness' in res
    n_pareto = len(res['pareto_front'])
    print(f"  ✓ NSGA-II: Pareto front size = {n_pareto}")
    return 1


def test_constraints():
    """3 constraint handling stratejisi."""
    print('\n═══ 3. Constraint Handling ═══')
    
    np.random.seed(42)
    weights = np.array([5, 8, 12, 15, 20, 10])
    capacity = 30
    
    def obj(sol):
        return -float((sol * np.arange(1, 7)).sum())
    
    def c_weight(sol):
        return (sol * weights).sum() - capacity
    
    enc = mh.BinaryEncoding(n_features=6)
    
    ok = 0
    for method in ['feasibility_rules', 'death_penalty', 'penalty']:
        wrapped = mh.make_constrained_fitness(obj, [c_weight], method=method)
        res = mh.run_bga(wrapped, enc, method='fdb_argmax',
                       pop_size=10, max_iter=20, seed=42)
        sol = res['best_sol'].astype(int)
        info = mh.evaluate_constraints(sol, [c_weight])
        is_feas = info['feasible']
        print(f"  {'✓' if is_feas else '✗'} {method:22s}: feasible={is_feas}")
        if is_feas:
            ok += 1
    
    return ok


def test_reproducibility():
    """Aynı seed → aynı sonuç."""
    print('\n═══ 4. Reproducibility ═══')
    
    enc = mh.ContinuousEncoding(bounds=[(-5, 5)] * 5)
    fn = lambda x: float(np.sum(x**2))
    
    r1 = mh.run_bga(fn, enc, pop_size=10, max_iter=10, seed=42)
    r2 = mh.run_bga(fn, enc, pop_size=10, max_iter=10, seed=42)
    
    same = abs(r1['best_fit'] - r2['best_fit']) < 1e-10
    print(f"  {'✓' if same else '✗'} Aynı seed → aynı fitness: {same}")
    return 1 if same else 0


def test_population_center():
    """Her encoding tipinde population_center doğru tipte ve boyutta dönüyor mu?"""
    print('\n═══ 5. population_center (NSM altyapısı) ═══')
    rng = np.random.RandomState(0)
    
    cases = [
        ('Binary',      mh.BinaryEncoding(8), np.ndarray, 8),
        ('Continuous',  mh.ContinuousEncoding([(-3, 3)] * 5), np.ndarray, 5),
        ('Categorical', mh.CategoricalEncoding([['a','b','c'], [1,2,3,4]]), np.ndarray, 2),
    ]
    
    ok = 0
    for name, enc, expected_type, expected_dim in cases:
        pop = enc.init_population(10, rng)
        c = enc.population_center(pop)
        is_ok = isinstance(c, expected_type) and len(c) == expected_dim
        print(f"  {'✓' if is_ok else '✗'} {name:12s}: type={type(c).__name__}, dim={len(c)}")
        if is_ok:
            ok += 1
    
    # Mixed dict döner
    mix = mh.MixedEncoding({
        'b': mh.BinaryEncoding(3),
        'c': mh.ContinuousEncoding([(0, 1)] * 2),
    })
    pop = mix.init_population(10, rng)
    c = mix.population_center(pop)
    is_ok = isinstance(c, dict) and 'b' in c and 'c' in c
    print(f"  {'✓' if is_ok else '✗'} {'Mixed':12s}: dict keys={list(c.keys())}")
    if is_ok:
        ok += 1
    
    return ok


def test_nsm_epm_combos():
    """NSM ve EPM × 4 encoding kombinasyonu çalışıyor + makul sonuç veriyor mu?"""
    print('\n═══ 6. NSM/EPM × 4 Encoding ═══')
    
    bin_enc = mh.BinaryEncoding(n_features=8)
    cont_enc = mh.ContinuousEncoding(bounds=[(-3, 3)] * 5)
    cat_enc = mh.CategoricalEncoding([['a','b','c'], [1, 2, 3]])
    mix_enc = mh.MixedEncoding({
        'lr': mh.ContinuousEncoding([(0.01, 0.3)]),
        'depth': mh.CategoricalEncoding([[3, 5, 7]]),
    })
    
    fns = {
        'Binary':      (bin_enc,  lambda s: -float(np.asarray(s, dtype=int).sum())),
        'Continuous':  (cont_enc, lambda x: float(np.sum(x**2))),
        'Categorical': (cat_enc,  lambda s: float(np.sum(s))),
        'Mixed':       (mix_enc,  lambda s: abs(s['lr'][0] - 0.1) + abs(s['depth'][0] - 1)),
    }
    
    ok = 0
    for mode_label, kw in [('NSM', {'use_nsm': True}), ('EPM', {'use_epm': True})]:
        for enc_name, (enc, fn) in fns.items():
            try:
                res = mh.run_bga(fn, enc, method='classic',
                                 pop_size=10, max_iter=10, seed=42, **kw)
                assert 'best_sol' in res and 'best_fit' in res
                # NSM modunda debug istatistik kontrolü
                if mode_label == 'NSM':
                    assert '_nsm_applied' in res
                print(f"  ✓ BGA+{mode_label} + {enc_name:12s}: fit={res['best_fit']:.4f}")
                ok += 1
            except Exception as e:
                print(f"  ✗ BGA+{mode_label} + {enc_name:12s}: {type(e).__name__}: {e}")
    
    return ok  # 8 beklenen


def test_nsm_epm_reproducibility():
    """NSM ve EPM modlarında reproducibility (aynı seed → aynı sonuç)."""
    print('\n═══ 7. NSM/EPM Reproducibility ═══')
    
    enc = mh.ContinuousEncoding(bounds=[(-5, 5)] * 5)
    fn = lambda x: float(np.sum(x ** 2))
    
    ok = 0
    for label, kw in [('NSM', {'use_nsm': True}), ('EPM', {'use_epm': True})]:
        r1 = mh.run_bga(fn, enc, pop_size=10, max_iter=15, seed=7, **kw)
        r2 = mh.run_bga(fn, enc, pop_size=10, max_iter=15, seed=7, **kw)
        same = abs(r1['best_fit'] - r2['best_fit']) < 1e-10
        print(f"  {'✓' if same else '✗'} {label}: aynı seed → aynı fitness: {same}")
        if same:
            ok += 1
    
    return ok  # 2 beklenen


def test_nsm_epm_edge_cases():
    """Edge case'ler: invalid preset, NSM+EPM kombinasyonu (yasak)."""
    print('\n═══ 8. NSM/EPM Edge Cases ═══')
    
    enc = mh.BinaryEncoding(5)
    fn = lambda s: -float(s.sum())
    
    ok = 0
    
    # 1. Invalid preset → ValueError
    try:
        mh.run_bga(fn, enc, use_nsm=True, nsm_preset=99,
                   pop_size=8, max_iter=5, seed=42)
        print("  ✗ Invalid nsm_preset: hata bekleniyordu, atılmadı")
    except ValueError:
        print("  ✓ Invalid nsm_preset: ValueError doğru atıldı")
        ok += 1
    
    # 2. NSM + EPM kombine artık çalışıyor (v3-final-rc2'de eklendi)
    try:
        r = mh.run_bga(fn, enc, use_nsm=True, use_epm=True,
                       pop_size=8, max_iter=5, seed=42)
        works = 'best_fit' in r and '_nsm_filtered_epochs' in r
        print(f"  {'✓' if works else '✗'} NSM+EPM kombine mod: çalıştı, fit={r['best_fit']:.4f}")
        if works:
            ok += 1
    except Exception as e:
        print(f"  ✗ NSM+EPM kombine mod: {type(e).__name__}: {e}")
    
    # 3. NSM gbest'i koruyor mu? (initial_best > final_best — fitness küçülmeli ya da sabit kalmalı)
    enc_c = mh.ContinuousEncoding([(-5, 5)] * 5)
    fn_c = lambda x: float(np.sum(x ** 2))
    r = mh.run_bga(fn_c, enc_c, use_nsm=True, pop_size=10, max_iter=20, seed=42)
    # history monoton azalmalı (gbest hiç bozulmaz)
    monotone = all(r['history'][i] >= r['history'][i+1] for i in range(len(r['history'])-1))
    print(f"  {'✓' if monotone else '✗'} NSM history monoton azalan (gbest korunur): {monotone}")
    if monotone:
        ok += 1
    
    return ok  # 3 beklenen


def test_permutation_encoding():
    """PermutationEncoding: sınıf invariantları + gerçek TSP koşumu."""
    print('\n═══ 9. PermutationEncoding (TSP) ═══')
    
    enc = mh.PermutationEncoding(n_items=10)
    rng = np.random.RandomState(42)
    ok = 0
    
    # 1. init permütasyon mu?
    p = enc.init_individual(rng)
    is_perm = sorted(p.tolist()) == list(range(10))
    print(f"  {'✓' if is_perm else '✗'} init: gerçek permütasyon = {is_perm}")
    if is_perm:
        ok += 1
    
    # 2. PMX yavruları permütasyon mu?
    p1 = enc.init_individual(rng)
    p2 = enc.init_individual(rng)
    c1, c2 = enc.crossover(rng, p1, p2)
    pmx_ok = (sorted(c1.tolist()) == list(range(10)) and 
              sorted(c2.tolist()) == list(range(10)))
    print(f"  {'✓' if pmx_ok else '✗'} PMX crossover: yavrular permütasyon = {pmx_ok}")
    if pmx_ok:
        ok += 1
    
    # 3. swap mutation permütasyon koruyor mu?
    m = enc.mutate(rng, p1, rate=0.3)
    mut_ok = sorted(m.tolist()) == list(range(10))
    print(f"  {'✓' if mut_ok else '✗'} swap mutation: permütasyon korunuyor = {mut_ok}")
    if mut_ok:
        ok += 1
    
    # 4. Kendall tau distance: self=0, reversed=1
    d_self = enc.distance(p1, p1)
    d_rev = enc.distance(p1, p1[::-1])
    dist_ok = (d_self == 0.0 and d_rev == 1.0)
    print(f"  {'✓' if dist_ok else '✗'} Kendall tau: self={d_self}, reversed={d_rev}")
    if dist_ok:
        ok += 1
    
    # 5. Gerçek TSP koşumu — algoritma fitness'i azaltabiliyor mu?
    np.random.seed(0)
    cities = np.random.uniform(0, 100, (10, 2))
    def tsp(perm):
        d = 0.0
        for i in range(len(perm)):
            d += np.linalg.norm(cities[perm[i]] - cities[perm[(i+1) % len(perm)]])
        return float(d)
    
    r = mh.run_bga(tsp, enc, pop_size=20, max_iter=30, seed=42)
    initial = r['history'][0]
    final = r['history'][-1]
    improved = final < initial
    print(f"  {'✓' if improved else '✗'} BGA + TSP: tour length {initial:.1f} → {final:.1f}")
    if improved:
        ok += 1
    
    # 6. Birden fazla algoritma permütasyon ile çalışmalı
    multi_ok = 0
    for runner in [mh.run_bga, mh.run_bcs, mh.run_bsos, mh.run_babc]:
        try:
            r = runner(tsp, enc, pop_size=15, max_iter=10, seed=42)
            assert sorted(r['best_sol'].tolist()) == list(range(10))
            multi_ok += 1
        except Exception:
            pass
    multi_pass = multi_ok == 4
    print(f"  {'✓' if multi_pass else '✗'} 4 algoritma TSP koşum: {multi_ok}/4")
    if multi_pass:
        ok += 1
    
    return ok  # 6 beklenen


def test_parallel_evaluation():
    """n_jobs > 1: classic ve EPM modlarında paralel eval reproducibility korur mu?"""
    print('\n═══ 10. Parallel Evaluation (n_jobs) ═══')
    
    enc = mh.ContinuousEncoding([(-5, 5)] * 6)
    ok = 0
    
    # 1. Classic BGA: n_jobs=1 ve n_jobs=2 aynı sonuç (reproducibility)
    r1 = mh.run_bga(_parallel_test_fitness, enc, pop_size=10, max_iter=8,
                    seed=42, n_jobs=1)
    r2 = mh.run_bga(_parallel_test_fitness, enc, pop_size=10, max_iter=8,
                    seed=42, n_jobs=2)
    same = abs(r1['best_fit'] - r2['best_fit']) < 1e-10
    print(f"  {'✓' if same else '✗'} Classic BGA: n_jobs=1 == n_jobs=2 ({r1['best_fit']:.6f}): {same}")
    if same:
        ok += 1
    
    # 2. EPM mode: paralel batch eval reproducibility
    r1 = mh.run_bga(_parallel_test_fitness, enc, use_epm=True,
                    pop_size=10, max_iter=8, seed=42, n_jobs=1)
    r2 = mh.run_bga(_parallel_test_fitness, enc, use_epm=True,
                    pop_size=10, max_iter=8, seed=42, n_jobs=2)
    same = abs(r1['best_fit'] - r2['best_fit']) < 1e-10
    print(f"  {'✓' if same else '✗'} EPM mode: n_jobs=1 == n_jobs=2 ({r1['best_fit']:.6f}): {same}")
    if same:
        ok += 1
    
    # 3. NSM mode: n_jobs > 1 warning + fallback to seri (sonuç değişmemeli)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = mh.run_bga(_parallel_test_fitness, enc, use_nsm=True,
                       pop_size=10, max_iter=8, seed=42, n_jobs=4)
        warned = any('NSM' in str(warning.message) for warning in w)
    print(f"  {'✓' if warned else '✗'} NSM + n_jobs>1: warning emitted = {warned}")
    if warned:
        ok += 1
    
    # 4. Diğer algoritmalar n_jobs reproducibility (BDE, BABC, BCS, BSOS, BGWO)
    multi_alg_ok = 0
    for alg_name, runner in [('BDE', mh.run_bde), ('BABC', mh.run_babc),
                              ('BCS', mh.run_bcs), ('BSOS', mh.run_bsos),
                              ('BGWO', mh.run_bgwo)]:
        try:
            r1 = runner(_parallel_test_fitness, enc, pop_size=10, max_iter=5,
                        seed=42, n_jobs=1)
            r2 = runner(_parallel_test_fitness, enc, pop_size=10, max_iter=5,
                        seed=42, n_jobs=2)
            if abs(r1['best_fit'] - r2['best_fit']) < 1e-10:
                multi_alg_ok += 1
        except Exception:
            pass
    multi_pass = multi_alg_ok == 5
    print(f"  {'✓' if multi_pass else '✗'} Diğer alg n_jobs reproducibility: {multi_alg_ok}/5")
    if multi_pass:
        ok += 1
    
    return ok  # 4 beklenen


def test_early_stopping_and_checkpoint():
    """Early stopping ve checkpointing — classic BGA, NSM, EPM."""
    print('\n═══ 11. Early Stopping & Checkpointing ═══')
    
    enc = mh.ContinuousEncoding([(-5, 5)] * 5)
    fn = lambda x: float(np.sum(x ** 2))
    ok = 0
    
    # 1. Classic BGA early stop devreye giriyor mu?
    r = mh.run_bga(fn, enc, pop_size=10, max_iter=300, seed=42,
                   early_stopping_patience=15)
    iters_run = len(r['history']) - 1
    early = r['stopped_early'] and iters_run < 300
    print(f"  {'✓' if early else '✗'} Classic early stop: {iters_run}/300 iter, stopped={r['stopped_early']}")
    if early:
        ok += 1
    
    # 2. NSM mode early stop
    r = mh.run_bga(fn, enc, use_nsm=True, pop_size=10, max_iter=300, seed=42,
                   early_stopping_patience=15)
    iters_run = len(r['history']) - 1
    early = r['stopped_early'] and iters_run < 300
    print(f"  {'✓' if early else '✗'} NSM early stop:     {iters_run}/300 iter, stopped={r['stopped_early']}")
    if early:
        ok += 1
    
    # 3. EPM mode early stop
    r = mh.run_bga(fn, enc, use_epm=True, pop_size=10, max_iter=300, seed=42,
                   early_stopping_patience=15)
    iters_run = len(r['history']) - 1
    early = r['stopped_early'] and iters_run < 300
    print(f"  {'✓' if early else '✗'} EPM early stop:     {iters_run}/300 iter, stopped={r['stopped_early']}")
    if early:
        ok += 1
    
    # 4. Checkpointing: save → resume → aynı sonucu üretmeli (Classic)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        ckpt_path = tmp.name
    
    try:
        r1 = mh.run_bga(fn, enc, pop_size=10, max_iter=20, seed=42,
                        checkpoint_every=10, checkpoint_path=ckpt_path)
        r2 = mh.run_bga(fn, enc, pop_size=10, max_iter=20, seed=42,
                        resume_from=ckpt_path)
        valid = r2['best_fit'] <= r1['best_fit'] + 1e-10
        print(f"  {'✓' if valid else '✗'} Classic checkpoint resume: r1={r1['best_fit']:.6f}, r2={r2['best_fit']:.6f}")
        if valid:
            ok += 1
    finally:
        import os
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
    
    # 5. NSM checkpoint resume
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        ckpt_path = tmp.name
    try:
        r1 = mh.run_bga(fn, enc, use_nsm=True, pop_size=10, max_iter=20, seed=42,
                        checkpoint_every=10, checkpoint_path=ckpt_path)
        r2 = mh.run_bga(fn, enc, use_nsm=True, pop_size=10, max_iter=20, seed=42,
                        resume_from=ckpt_path)
        valid = r2['best_fit'] <= r1['best_fit'] + 1e-10
        print(f"  {'✓' if valid else '✗'} NSM checkpoint resume:     r1={r1['best_fit']:.6f}, r2={r2['best_fit']:.6f}")
        if valid:
            ok += 1
    finally:
        import os
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
    
    # 6. EPM checkpoint resume
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        ckpt_path = tmp.name
    try:
        r1 = mh.run_bga(fn, enc, use_epm=True, pop_size=10, max_iter=20, seed=42,
                        checkpoint_every=10, checkpoint_path=ckpt_path)
        r2 = mh.run_bga(fn, enc, use_epm=True, pop_size=10, max_iter=20, seed=42,
                        resume_from=ckpt_path)
        valid = r2['best_fit'] <= r1['best_fit'] + 1e-10
        print(f"  {'✓' if valid else '✗'} EPM checkpoint resume:     r1={r1['best_fit']:.6f}, r2={r2['best_fit']:.6f}")
        if valid:
            ok += 1
    finally:
        import os
        if os.path.exists(ckpt_path):
            os.unlink(ckpt_path)
    
    return ok  # 6 beklenen


def test_epm_nsm_combine():
    """EPM-NSM kombine mod: B3 hibrit fazı (v3-final-rc2 yeni)."""
    print('\n═══ 12. EPM-NSM Kombine Mod (B3) ═══')
    
    enc = mh.ContinuousEncoding([(-5, 5)] * 6)
    fn = lambda x: float(np.sum(x ** 2))
    ok = 0
    
    # 1. Çalışıyor + result dict standart anahtarları içeriyor
    r = mh.run_bga(fn, enc, use_nsm=True, use_epm=True,
                   pop_size=10, max_iter=30, seed=42)
    keys_ok = all(k in r for k in ['best_sol', 'best_fit', 'history',
                                    'n_evaluations', 'stopped_early',
                                    '_nsm_filtered_epochs'])
    print(f"  {'✓' if keys_ok else '✗'} EPM-NSM kombine: keys ok, fit={r['best_fit']:.6f}")
    if keys_ok:
        ok += 1
    
    # 2. Reproducibility
    r1 = mh.run_bga(fn, enc, use_nsm=True, use_epm=True,
                    pop_size=10, max_iter=20, seed=7)
    r2 = mh.run_bga(fn, enc, use_nsm=True, use_epm=True,
                    pop_size=10, max_iter=20, seed=7)
    same = abs(r1['best_fit'] - r2['best_fit']) < 1e-10
    print(f"  {'✓' if same else '✗'} EPM-NSM reproducibility: {r1['best_fit']:.6f} == {r2['best_fit']:.6f}")
    if same:
        ok += 1
    
    # 3. switch_prob=0 (max_iter=1 → epoch+1=1, 1/1=1, sigmoid(15*0.5)≈1) — 
    #    Aslında erken epoch için switch_prob düşük; epoch=1 zaten ilerlemenin sonu
    #    Bunun yerine sigmoid_switch matematiksel testi
    sw_early = mh.sigmoid_switch(1, 100)  # erken
    sw_late = mh.sigmoid_switch(99, 100)  # geç
    print(f"  {'✓' if sw_early < 0.1 and sw_late > 0.9 else '✗'} sigmoid_switch erken→geç: {sw_early:.3f} → {sw_late:.3f}")
    if sw_early < 0.1 and sw_late > 0.9:
        ok += 1
    
    # 4. EPM-NSM 4 encoding'de çalışmalı
    cases = [
        ('Binary',      mh.BinaryEncoding(8), lambda s: -float(s.sum())),
        ('Categorical', mh.CategoricalEncoding([['a','b','c']]*4), lambda s: float(np.sum(s))),
        ('Mixed',       mh.MixedEncoding({'b': mh.BinaryEncoding(3),
                                          'c': mh.ContinuousEncoding([(0,1)]*2)}),
                        lambda s: -float(s['b'].sum()) + float(np.sum(s['c']**2))),
    ]
    enc_ok = 0
    for name, e, f in cases:
        try:
            r = mh.run_bga(f, e, use_nsm=True, use_epm=True,
                           pop_size=10, max_iter=8, seed=42)
            assert 'best_fit' in r
            enc_ok += 1
        except Exception as ex:
            print(f"    ✗ EPM-NSM + {name}: {type(ex).__name__}: {ex}")
    enc_pass = enc_ok == 3
    print(f"  {'✓' if enc_pass else '✗'} EPM-NSM × 3 encoding: {enc_ok}/3")
    if enc_pass:
        ok += 1
    
    return ok  # 4 beklenen


def test_logging():
    """Logging: 4 modda da CSV log dosyası doğru üretiliyor mu?"""
    print('\n═══ 13. Logging (CSV) ═══')
    
    enc = mh.ContinuousEncoding([(-5, 5)] * 4)
    fn = lambda x: float(np.sum(x ** 2))
    ok = 0
    
    import tempfile
    import os
    
    for label, kw in [('Classic', {}),
                      ('NSM',     {'use_nsm': True}),
                      ('EPM',     {'use_epm': True}),
                      ('EPM-NSM', {'use_nsm': True, 'use_epm': True})]:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as tmp:
            log_path = tmp.name
        try:
            r = mh.run_bga(fn, enc, pop_size=10, max_iter=8, seed=42,
                           log_file=log_path, log_every=2, **kw)
            with open(log_path) as f:
                lines = f.readlines()
            # Header + en az 2 veri satırı (8 iter, log_every=2 → 4 log)
            header_ok = lines[0].strip() == 'iter,best_fit,mean_fit,std_fit,n_evaluations,elapsed_seconds'
            data_ok = len(lines) >= 3  # header + 2+ data rows
            valid = header_ok and data_ok
            print(f"  {'✓' if valid else '✗'} {label:8s} log: header_ok={header_ok}, lines={len(lines)}")
            if valid:
                ok += 1
        finally:
            if os.path.exists(log_path):
                os.unlink(log_path)
    
    return ok  # 4 beklenen


def test_auto_tuning():
    """tune_and_run: trial faz + final koşum, en iyi config seçimi."""
    print('\n═══ 14. Auto-Tuning Meta-Loop ═══')
    
    enc = mh.ContinuousEncoding([(-5, 5)] * 4)
    fn = lambda x: float(np.sum(x ** 2))
    ok = 0
    
    # 1. Default çalışıyor + 4 anahtar dönüyor
    r = mh.tune_and_run(fn, enc,
                       algorithms=('bga', 'bde'),
                       methods=('classic',),
                       pop_sizes=(10,),
                       n_trials_per_config=2,
                       budget=500, seed=42)
    keys_ok = all(k in r for k in ['best_config', 'best_result',
                                    'trial_results', 'total_evaluations'])
    print(f"  {'✓' if keys_ok else '✗'} tune_and_run keys ok, best={r['best_config']}")
    if keys_ok:
        ok += 1
    
    # 2. Trial result sayısı doğru (n_alg × n_method × n_ps)
    r = mh.tune_and_run(fn, enc,
                       algorithms=('bga', 'bde'),
                       methods=('classic', 'fdb'),
                       pop_sizes=(10, 15),
                       n_trials_per_config=1,
                       budget=400, seed=42)
    expected_n = 2 * 2 * 2
    n_ok = len(r['trial_results']) == expected_n
    print(f"  {'✓' if n_ok else '✗'} Trial sayısı: {len(r['trial_results'])}/{expected_n}")
    if n_ok:
        ok += 1
    
    # 3. Reproducibility
    r1 = mh.tune_and_run(fn, enc, algorithms=('bga',), methods=('classic',),
                        pop_sizes=(10,), n_trials_per_config=1, budget=300, seed=7)
    r2 = mh.tune_and_run(fn, enc, algorithms=('bga',), methods=('classic',),
                        pop_sizes=(10,), n_trials_per_config=1, budget=300, seed=7)
    same = abs(r1['best_result']['best_fit'] - r2['best_result']['best_fit']) < 1e-10
    print(f"  {'✓' if same else '✗'} Reproducibility: {r1['best_result']['best_fit']:.6f}")
    if same:
        ok += 1
    
    return ok  # 3 beklenen


if __name__ == "__main__":
    print('═' * 65)
    print('  metaheuristics_bk v3-FINAL — CONSISTENCY TEST')
    print('═' * 65)
    
    ok1, skipped = test_all_algorithm_encoding_combos()
    ok2 = test_nsga2()
    ok3 = test_constraints()
    ok4 = test_reproducibility()
    ok5 = test_population_center()
    ok6 = test_nsm_epm_combos()
    ok7 = test_nsm_epm_reproducibility()
    ok8 = test_nsm_epm_edge_cases()
    ok9 = test_permutation_encoding()
    ok10 = test_parallel_evaluation()
    ok11 = test_early_stopping_and_checkpoint()
    ok12 = test_epm_nsm_combine()
    ok13 = test_logging()
    ok14 = test_auto_tuning()
    
    total_ok = ok1 + ok2 + ok3 + ok4 + ok5 + ok6 + ok7 + ok8 + ok9 + ok10 + ok11 + ok12 + ok13 + ok14
    expected = 24 + 1 + 3 + 1 + 4 + 8 + 2 + 3 + 6 + 4 + 6 + 4 + 4 + 3  # 73
    
    print('\n' + '═' * 65)
    print(f"  Algoritma × Encoding:      {ok1}/24 başarılı, {skipped}/4 skipped")
    print(f"  NSGA-II:                   {ok2}/1")
    print(f"  Constraints:               {ok3}/3")
    print(f"  Reproducibility (BGA):     {ok4}/1")
    print(f"  population_center:         {ok5}/4")
    print(f"  NSM/EPM × encoding:        {ok6}/8")
    print(f"  NSM/EPM Reproducibility:   {ok7}/2")
    print(f"  NSM/EPM Edge cases:        {ok8}/3")
    print(f"  PermutationEncoding:       {ok9}/6")
    print(f"  Parallel Evaluation:       {ok10}/4")
    print(f"  Early Stop & Checkpoint:   {ok11}/6")
    print(f"  EPM-NSM Kombine (B3):      {ok12}/4")
    print(f"  Logging (CSV):             {ok13}/4")
    print(f"  Auto-Tuning:               {ok14}/3")
    print(f"  TOPLAM:                    {total_ok}/{expected}")
    print('═' * 65)
    
    if total_ok == expected:
        print('\n✓ Tüm testler başarılı!')
    else:
        sys.exit(1)
