# Changelog

## v0.3.5 — 2026-09-25

### Düzeltme: `run_bde` continuous-dışı mutasyon oranı (KUSUR)
- Binary/Categorical/Mixed/Permutation dallarında `encoding.mutate(rng, c1, F)`
  çağrısı DE ölçek katsayısını (F=0.5) bit-çevirme olasılığı olarak geçiriyordu
  → ikili BDE rastgele aramaya denk (OneMax n=60, 1000 FE: hata 14–20 vs
  rastgele 18–20; MEALPY-DE 11.5; F=1/n → 0). Bulgu: mathandai projesi NB01
  ikili sağlaması (validation/binary_sanity.py + mealpy_crosscheck.py).
- Yeni parametre `noncont_mutation_rate: Optional[float] = None` → encoding
  varsayılanı (Binary 1/n). `F`/`CR` yalnız ContinuousEncoding'de.
- Davranış: continuous RNG akışı özdeş; binary BDE sonuçları v0.3.4 ile
  KARŞILAŞTIRILAMAZ (yeniden koşum gerekir). Diğer 6 algoritma değişmedi.
- Etki alanı notu: YL tezi (v8) Stage 08 BDE sonuçları v0.3.4 ile üretildi;
  tez metninde beyan gerekir.

## v0.3.4 — 2026-07-10

### EPM resmi kod doğrulaması (FEX #181754, Üstünsoy et al. 2025)
- `epm_population_update()` seçim indeksleri, resmi `pop_productions.m`
  (Method-1) ve `pop_productions_2.m` (Method-2) transkriptleriyle 10.000
  rastgele popülasyonda BİREBİR özdeş doğrulandı
  (`validation/epm_matlab_crosscheck.py`, T1–T3).
- Makale pseudo-kodundaki iki belirsizlik resmi kodla çözüldü; kütüphane
  zaten doğru taraftaymış: (a) U = [P; Pchild] (ebeveynler önce — makale
  Eq. 6'nın tersi), (b) akraba ofseti ±n (Algorithm-2'deki "n/2" yazımında
  n = |U|). Ayrıntı: kod içi EPM UYARLAMA BEYANI bloğu.
- Beyanlı kalan sapmalar: eşit-fitness'ta randperm (resmi) vs deterministik
  (kütüphane); doldurma prosedürü; H3 rehber eşleşmesinin resmi kodda
  algoritma/Case-özgü olması; resmi EPM-AGDE'nin epoch-içi açgözlü
  değiştirmeyi koruması (makale Algorithm-4 ile çelişir — makale-kod farkı).
- EPM continuous doğrulama durumu güncellendi: H2 çekirdeği "resmi kodla
  doğrulanmış"; algoritma-bütünü için "EPM-uyarlaması" çerçevesi sürer.

### Validation eklemeleri
- `validation/epm_matlab_crosscheck.py` (yukarıdaki T1–T4).
- `validation/fdb_sos_pilot3_reference.py` + `pilot3_reference_results.json`:
  fdb_reference yön testi (CEC2017 D=10; kazanma sayıları fdb-referans 4/7,
  classic 2/7, fdb-uyarlaması 1/7 — PİLOT, kanonik değil).

### Davranış
- **Davranış değişikliği YOK.** Yalnız dokümantasyon/yorum blokları ve sürüm
  damgaları (README/RELEASE_NOTES/pyproject/`__version__`). RNG akışı
  v0.3.3 ile özdeş — A/B probu 4 method kombinasyonunda bit-düzeyinde aynı
  `best_fit` üretti.

## v0.3.3 — 2026-07-10

### Yeni: mesafe metriği parametreleştirildi (resmi FDB koduyla hiza)
- `ContinuousEncoding(bounds, distance_metric=...)`: `'euclidean'` (varsayılan,
  aralık-normalize — v3 davranışı, geriye uyumlu) | `'manhattan'` (HAM |a−b|
  toplamı — resmi FDB-SOS MATLAB kodu FEX #72311 `fitnessDistanceBalance.m`
  ile birebir) | `callable(a, b) -> float` (özel/uyarlanmış metrik).
  Bilinmeyen değer `ValueError` (fail-fast). Gerekçe: KBS 2020 Algorithm 2
  metriği kullanıcıya bırakır ("Select a distance metric"); kütüphane genel
  amaçlı olduğundan metrik projeye göre seçilebilir olmalı.

### Yeni: `run_bsos` FDB davranış parametreleri
- `fdb_phases` (varsayılan: üç faz — tarihsel davranış), `fdb_refresh`
  (`'iter'` varsayılan | `'selection'` = resmi kod), `fdb_exclude_self`
  (varsayılan True). Bilinmeyen değerler `ValueError`.
- `fdb_reference=True` preset'i: yalnız-mutualism + seçim-başına skor +
  self dahil; ContinuousEncoding'de `distance_metric='manhattan'` ŞART
  (değilse raise — sessiz metrik değişimi yok). Bu preset resmi kodla
  yapısal hizadır; raporlarda "FDB-SOS (Kahraman 2020, resmi kod hizası)"
  olarak etiketlenebilir. Preset dışı kombinasyonlar "FDB-uyarlaması"dır.

### Davranış notları (pilot-etkili, kanonik sonuç yok)
- `method='fdb'` partner seçimi: argmax kendisiyse artık uniform'a düşmez,
  self hariç en yüksek skorlu birey seçilir (deterministik ve tutarlı).
- BSOS'ta method≠classic RNG tüketim sırası değişti → v0.3.2 ile aynı seed
  farklı akış verebilir. KANONİK KART'ta kanonik sayı bulunmadığından
  (tümü pilot) yayın etkisi yoktur; yeni koşumlar v0.3.3 ile yapılır.

### Doğrulama
- `validation/fdb_matlab_crosscheck.py`: T1 makale örnek olayı (Fig. 1b)
  birebir; T3 Manhattan metrikle resmi kod transkriptiyle 10.000 rastgele
  popülasyonda 0.0 fark; T2 metrik farkının etkisi (%83.3 argmax uyuşması);
  T4 preset duman testi. Testler: 18 → 28 (tests/test_fdb_reference.py).
- Bulgu (SAVUNMA_DOSYASI adayı): KBS 2020 Tablo 3 kazanan varyantı SFDB2
  (çarpımsal) olarak tanımlar; yayımlanan resmi kod ise toplamsal skoru
  (normF + normD) uygular. Kütüphane resmi kodla aynı (toplamsal) taraftadır.


## v0.3.1 — 2026-07-09

### ⚠ KIRICI DEĞİŞİKLİK: method adları kalıcılaştı
- `'fdb'` artık **deterministik klasik FDB** (Kahraman et al. 2020) — v3.0'daki `'fdb_argmax'`.
- `'rfdb'` artık **rulet-FDB** (Aras et al. 2021) — v3.0'daki `'fdb'`.
- `'fdb_argmax'` geriye-uyum alias'ı olarak kabul edilir → `'fdb'`ye eşlenir (tek seferlik uyarıyla).
- **v3.0 scriptinde `method='fdb'` kullanan kod v3.1'de FARKLI (deterministik) davranır** — `'rfdb'`ye güncelle.
- Her method ilk kullanımda tek satırlık hatırlatma basar; `SHOW_METHOD_HINTS = False` ile kapatılır.

### Düzeltmeler
- **BGWO (kritik):** V-shape transfer pozisyona değil ADIMA uygulanır + binary'de
  birey-bazlı greedy seçim. v3.0'da OneMax(40)'ta ~13'te takılıyordu (BGA=0);
  düzeltme sonrası ~2.6 (5 seed ort.). Ölü kod (kullanılmayan rand çekimi) silindi.
  Regresyon testi: `test_bgwo_convergence`.
- **Fail-fast:** bilinmeyen `method` değeri artık anında `ValueError` (v3.0'da
  yazım hatası sessizce rulet davranışına düşüyordu).
- **BPSO:** `'fdb'` ve `'rfdb'` artık farklı davranır (v3.0'da ikisi de argmax'a
  düşüyordu). Regresyon testi: `test_bpso_methods_differ`.
- `tune_and_run` artık `n_jobs` parametresini koşumlara geçirir.
- BABC `limit` docstring'i kodla eşitlendi; BCS/BSOS docstring'leri 3 method'u belgeler.

### Yenilikler
- `select_features(X, y, ...)`: herhangi bir tablosal veri setinde (DataFrame/ndarray)
  tek çağrıda meta-sezgisel özellik seçimi — task auto, NaN medyan-doldurma,
  sütun adı korunumu, sklearn model enjeksiyonu.
- `load_csv_dataset(path, target, drop=...)`: CSV → (X, y).
- `examples/11_csv_feature_selection.py`.
- NSM modülüne **uyarlama beyanı**: MPS/PS yaklaşıklamaları, erken-çıkış kuralları
  ve sigmoid switch yönünün makaleden farkları kod içinde açıkça belgelendi;
  NSM_PRESETS provenance'ı (MATLAB NSM.m) not edildi.

### Test
- 73 → **81 test** (fail-fast/alias 4, BGWO yakınsama 1, BPSO ayrımı 1, select_features 2).


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

## v0.3.2 (2026-07-09)
- **BSOS düzeltmesi (continuous):** `ContinuousEncoding` için orijinal
  Cheng & Prayogo (2014) aritmetik fazları eklendi (Mutual Vector + BF,
  rand(-1,1) commensalism, boyut-altkümesi parasitism). Önceki
  crossover-tabanlı davranış diğer encoding'lerde UYARLAMA olarak sürüyor.
- Doğrulama: MEALPY OriginalSOS ile aynı bütçede Sphere/Rastrigin/Ackley'de
  aynı mertebe (Rastrigin=0, Ackley=4.44e-16 birebir). Binary regresyon temiz.
- **Yeni örnek:** `examples/12_method_comparison.py` — kendi probleminde
  classic/fdb/rfdb karşılaştırma şablonu (doğrulama oturumu bulgularıyla).
- `validation/` klasörü eklendi: MEALPY çapraz kontrol, SOS yeniden doğrulama,
  EPM teşhis, FDB-SOS CEC2017 pilot scriptleri + sonuçlar + notlar.
