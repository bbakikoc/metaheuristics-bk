# v3-beta'dan v3-final'a Yol Haritası

## v3-beta (Mevcut) — Geniş Kapsam

✅ **Tamamlanan özellikler:**
- 4 encoding: Binary, Continuous, Categorical, Mixed
- 5 algoritma: BGA, BDE, BPSO, BABC, BGWO (encoding-agnostic)
- 3 rehberlik: classic, fdb, fdb_argmax (encoding-aware distance metric ile)
- NSGA-II multi-objective (Pareto front, crowding distance)
- 3 constraint handling stratejisi (feasibility_rules, death_penalty, penalty)
- 5 örnek (knapsack, Rastrigin, mixed XGBoost, NSGA-II, constrained)
- 21/21 test geçiyor
- Reproducibility (aynı seed → aynı sonuç)

## v3-final — Sonraki Adımlar (~10-15 saat)

### 1. v2'deki kalan algoritmaları taşı (~4 saat)

Şu an v3-beta'da 5/7 algoritma var. Eksik:
- BCS (Binary Cuckoo Search) — Lévy flight ile
- BSOS (Binary Symbiotic Organisms Search) — 3 fazlı

Strateji: encoding-aware operatörlerle taşı, Lévy flight için ContinuousEncoding 
özel davranış (binary'de sigmoid mapping).

### 2. NSM ve EPM modüllerini v3'e port (~3 saat)

v2'deki bu iki modül binary varsayıyor. v3 için:
- NSM'in OFS/MPS/PS skorları encoding-bağımsız (sadece distance metric'e bağımlı)
- EPM popülasyon yönetimi de zaten encoding-agnostic
- API: `run_bga(..., use_nsm=True)` veya `run_bga(..., use_epm=True)`

### 3. Parallel fitness evaluation (~2 saat)

```python
from multiprocessing import Pool

result = run_bga(
    fitness_fn=expensive_fit,
    encoding=encoding,
    n_jobs=4,  # 4 paralel worker
)
```

Pop_size=20 ile fitness 5x hızlanabilir (yarış koşulu yok, embarrassingly parallel).

### 4. Early stopping & checkpointing (~2 saat)

```python
result = run_bga(
    fitness_fn=fit,
    encoding=encoding,
    max_iter=1000,
    early_stopping_patience=20,  # 20 iter improvement yoksa dur
    log_file='optimization.log',
    checkpoint_every=50,         # her 50 iter'de save
)
```

### 5. PermutationEncoding ekle (~3 saat)

Travelling Salesman Problem (TSP), job-shop scheduling, sequence optimization 
için. Operatörler:
- PMX crossover (Partially Matched Crossover)
- Swap mutation, inversion mutation, scramble mutation
- Sequence-based distance (Kendall tau veya Levenshtein)

### 6. Sphinx documentation + Read the Docs (~2 saat)

Tam API referans, tutorial sayfaları, examples gallery.

## v4 Vizyonu

- Hyperband / BOHB tipi multi-fidelity desteği
- Surrogate-assisted optimization (Gaussian Process)
- Parallel coordinates plot (multi-objective görselleştirme)
- Tam scikit-learn uyumlu interface (`MetaHeuristicCV`)

## Tasarım Notları (v3-beta sırasında alınan kararlar)

### Neden NSGA-II'de constraint dominance?

Multi-objective + constraints yaygın bir kombinasyon. Standart NSGA-II 
constraint domination ile genişletilebilir (Deb 2000):

> p constraint-dominates q ↔ p feasible & q infeasible, OR
>                            both infeasible & p has less violation, OR
>                            both feasible & p Pareto-dominates q

Bu sayede aynı algoritma hem multi-obj hem constrained problem çözer.

### Neden 3 constraint stratejisi?

- **feasibility_rules** (default): Deb 2000, parametre yok, en sağlam
- **death_penalty**: feasible bölge geniş ise iyi, dar ise çıkmaz
- **penalty**: penalty_factor ayarı zor ama bazen daha iyi exploration

Kullanıcı problemine göre seçer; varsayılan feasibility_rules.

### NSGA-II population size önerileri

- 2 objektif: pop_size = 20-50
- 3 objektif: pop_size = 50-100
- 4+ objektif: NSGA-II yetmez, NSGA-III gerekir (v4 planı)

## Tahmini Toplam Efor (v3-final)

| Görev | Tahmini |
|---|---|
| BCS + BSOS port | 4 saat |
| NSM + EPM port | 3 saat |
| Parallel evaluation | 2 saat |
| Early stop + logging | 2 saat |
| PermutationEncoding | 3 saat |
| Sphinx docs | 2 saat |
| **TOPLAM** | **~16 saat** |

Bu konuşma kapsamına sığmaz — gelecek konuşmalarda parça parça eklenir.
