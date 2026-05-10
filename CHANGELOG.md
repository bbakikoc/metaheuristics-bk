# Changelog

## [0.3.0-rc2] - 2026-05-10 ⭐⭐⭐ v3-FINAL

İlk başta v3.1'e bırakılan tüm geliştirmeler bu sürümde tamamlandı.
**73/73 consistency test** geçiyor.

### Added

#### EPM-NSM Kombine Mod (B3)
- `epm_nsm_population_update()` — EPM Algoritma 2'nin NSM-süzgeçli versiyonu
- `run_bga(use_nsm=True, use_epm=True)` — artık çalışıyor (önceden NotImplementedError)
- Sigmoid switch ile fit-half kararına NSM süzgeci eklenir
- 4 encoding'de (Binary, Continuous, Categorical, Mixed) çalışır

#### NSM ve EPM için Checkpointing
- Önceden yalnızca classic mod'da olan checkpoint, NSM ve EPM modlarına da
- NSM checkpoint state: rng + pop + fit + n_eval + gbest + history + iter +
  4 NSM istatistiği (nsm_applied, nsm_accepted, greedy_applied, greedy_accepted)
- EPM checkpoint state: rng + P + F + n_eval + gbest + history + epoch
- Resume tüm modlarda transparan: `resume_from='ckpt.pkl'`

#### Diğer Algoritmaların Parallel Desteği
- `run_bde, run_bpso, run_babc, run_bgwo, run_bcs, run_bsos` artık `n_jobs` alıyor
- BDE: full paralel (trial generation sıralı, eval batch paralel)
- BPSO/BABC/BGWO/BCS/BSOS: init eval paralel (iter içi sıralı, iç state bağımlılıkları)

#### Logging (CSV)
- `_IterLogger` sınıfı: 4 modda da CSV log dosyası
- Sütunlar: `iter, best_fit, mean_fit, std_fit, n_evaluations, elapsed_seconds`
- `log_file='opt.csv', log_every=5` parametreleri

#### Auto-Tuning Meta-Loop
- `tune_and_run(fitness_fn, encoding, algorithms, methods, pop_sizes, ...)` 
- Trial faz (~%30 budget): tüm kombinasyonları küçük denemelerle test eder
- Final faz (~%70 budget): en iyi konfigürasyonu tam koşar
- Result: `best_config`, `best_result`, `trial_results`, `total_evaluations`

### Changed
- Versiyon: `0.3.0-rc1` → `0.3.0-rc2`
- `__all__`'a 2 yeni public symbol: `epm_nsm_population_update`, `tune_and_run`
- Test sayısı: 59 → **73** (14 yeni test)

### Fixed
- (rc1'de) EPM modunda tek-sayı pop_size durumu (otomatik +1 + uyarı)

## [0.3.0-rc1] - 2026-05-10

## [0.3.0-rc1] - 2026-05-10

Tüm v2 işlevleri encoding-aware olarak v3'e taşındı + 4 yeni özellik.
**59/59 consistency test** geçiyor.

### Added

#### NSM ve EPM Modülleri (Madde 1)
- `NSM_PRESETS` — 10 preset (MATLAB NSM.m birebir)
- `nsm_decide()` — encoding-aware Xold↔Xnew karar fonksiyonu
- `sigmoid_switch()`, `dynamic_weights()` — yardımcılar
- `epm_population_update()` — H2 (n/2 fitness + n/2 FDB seçim)
- `epm_guide_select()` — H3 fit-half/fdb-half eşleşmesi
- `Encoding.population_center()` — 4 encoding için (Binary→mod, Continuous→mean,
  Categorical→en-sık-kategori, Mixed→bileşene delegate, Permutation→medoid)
- `run_bga(use_nsm=True)` ve `run_bga(use_epm=True)` opsiyonel modlar
- `examples/06_nsm_epm_demo.py`

#### BCS ve BSOS Algoritmaları (Madde 2)
- `run_bcs` — Cuckoo Search (Lévy flight) — 4 encoding'te encoding-aware:
  Binary→bit-flip, Continuous→step, Categorical→reset, Mixed→delegate
- `run_bsos` — Symbiotic Organisms Search 3 fazlı — `encoding.crossover` ve
  `encoding.mutate` üzerinden universal (binary OR/XOR varsayımı yok)
- `_levy_flight_step`, `_levy_perturb` yardımcıları
- `examples/07_bcs_bsos_demo.py`

#### PermutationEncoding (Madde 3)
- `PermutationEncoding(n_items)` sınıfı:
  - init: shuffle permütasyon
  - crossover: PMX (Partially Matched Crossover, Goldberg & Lingle 1985)
  - mutate: swap mutation (en az 1 swap garantisi)
  - distance: normalized Kendall tau
  - population_center: medoid (popülasyon-içi en merkezi birey)
- TSP, job-shop scheduling, sequence optimization için
- BGA, BCS, BSOS, BABC, BDE permütasyon ile çalışır (BPSO, BGWO çalışmaz)
- NSM ve EPM bile permütasyon uzayında çalışıyor (encoding-aware!)
- `examples/08_tsp_demo.py` — 20-şehirli TSP, BABC+fdb %58 iyileşme

#### Parallel Fitness Evaluation (Madde 4)
- `_eval_pop(fitness_fn, pop, n_jobs)` helper — `multiprocessing.Pool`
- `run_bga(n_jobs=4)` — 4-5x hızlanma pahalı fitness'larda
- Reproducibility korunur: aynı seed → aynı sonuç
- Classic mode + EPM mode için aktif (NSM steady-state ignore eder)
- ⚠️ `fitness_fn` pickle-able olmalı (top-level def, lambda DEĞİL)

#### Early Stopping (Madde 5)
- `early_stopping_patience=N` — N iter iyileşme yoksa otomatik dur
- `early_stopping_tol=1e-8` — iyileşme eşiği
- 3 modda da çalışır: classic, NSM, EPM
- Result dict'inde `stopped_early` flag'i

#### Checkpointing (Madde 6)
- `checkpoint_every=N` + `checkpoint_path='ckpt.pkl'` → her N iter'de save
- `resume_from='ckpt.pkl'` → kaldığı yerden devam
- Pickle'da kaydedilen state: `rng, pop, fit, gbest, history, iter, _meta`
- Sadece classic mod (NSM/EPM v3.1'e bırakıldı)
- `examples/09_early_stop_checkpoint_demo.py`

### Changed
- Versiyon: `0.3.0-beta` → `0.3.0-rc1`
- `__all__` listesine 8 yeni public symbol eklendi
- Test sayısı: 21 → **59** (38 yeni test)
- BÖLÜM yapısı: 5 → 7 (NSM/EPM Bölüm 6, Constraints Bölüm 7)

### Fixed
- EPM modunda tek-sayı pop_size durumu (otomatik +1 ile düzeltilir + uyarı)

## [0.3.0-beta] - 2026-05-10

### Added
- **2 yeni algoritma** (encoding-agnostic):
  - `run_babc` — Artificial Bee Colony (3 fazı: Employed/Onlooker/Scout)
  - `run_bgwo` — Grey Wolf Optimizer (V-shape variant + Elitism)
- **NSGA-II multi-objective** (Deb et al. 2002):
  - `run_nsga2` — Pareto front döner
  - Fast non-dominated sort O(MN²)
  - Crowding distance ile diversity
  - Constrained version (Deb 2000)
- **Native constraint handling**:
  - `make_constrained_fitness()` wrapper
  - 3 strateji: `feasibility_rules`, `death_penalty`, `penalty`
  - `evaluate_constraints()` çözüm raporlaması
- 2 yeni örnek:
  - `04_nsga2_multi_objective.py` — Pareto front, knee point
  - `05_constrained_knapsack.py` — 3 kısıtlı knapsack, 3 strateji
- 21/21 consistency test geçiyor (önceki: 11/11)

### Changed
- Versiyon: `0.3.0-alpha` → `0.3.0-beta`
- README ve docs güncellendi

## [0.3.0-alpha] - 2026-05-10
- İlk universal sürüm: 4 encoding + 3 algoritma (BGA, BDE, BPSO)
- 11/11 test
