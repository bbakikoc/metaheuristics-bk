# metaheuristics_bk v3-final

> **Universal Meta-Heuristic Algorithms Library — Encoding-Aware**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0--rc2-green.svg)]()
[![Tests](https://img.shields.io/badge/tests-73%2F73-brightgreen.svg)]()

Tek dosyalık (`metaheuristics_bk_v3.py`, ~3000 satır), encoding-agnostic
metasezgisel optimizasyon kütüphanesi. Binary, Continuous, Categorical,
Mixed ve Permutation solution uzaylarında 7 algoritma + 2 rehberlik modülü
(NSM, EPM) + multi-objective (NSGA-II) + constraint handling + paralel
fitness eval + erken durdurma + checkpointing.

> 📘 **Kendi veri setinde optimizasyon yapacaksan, [USER_GUIDE.md](USER_GUIDE.md)'a bak.**
> Adım adım rehber: veri seti yükleme → problem tanımlama → algoritma seçimi
> → sonuçları kaydetme. 6 tipik senaryo (feature selection, hyperparameter
> tuning, TSP, function optimization, multi-objective, constrained) için
> çalışan kod örnekleri içeriyor.

---

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [5-Dakikalık Başlangıç](#5-dakikalık-başlangıç)
- [Kapsamlı API](#kapsamlı-api)
- [Yeni Bir Projede Entegrasyon](#yeni-bir-projede-entegrasyon)
- [Mimari Tasarım](#mimari-tasarım)
- [Test Etme](#test-etme)
- [Sürüm Geçmişi](#sürüm-geçmişi)

---

## Özellikler

### 5 Encoding Tipi (Solution Uzayı)
- `BinaryEncoding` — `{0,1}^n` (knapsack, feature selection, set cover)
- `ContinuousEncoding` — `[low, high]^n` (function optimization, hyperparameter tuning)
- `CategoricalEncoding` — kategori indeksleri (kernel/optimizer/activation seçimi)
- `MixedEncoding` — yukarıdakileri dict ile birleştirir (XGBoost-tipi karma)
- `PermutationEncoding` — `{0,1,...,n-1}` permütasyonu (TSP, scheduling)

### 7 Single-Objective Algoritma (encoding-agnostic)

| Alg | Tam Adı | Kategori×Mixed | Permutation |
|---|---|---|---|
| `run_bga` | Binary Genetic Algorithm | ✅ | ✅ |
| `run_bde` | Binary Differential Evolution | ✅ | ✅ |
| `run_bpso` | Binary Particle Swarm Optimization | ❌ | ❌ |
| `run_babc` | Binary Artificial Bee Colony | ✅ | ✅ |
| `run_bgwo` | Binary Grey Wolf Optimizer | ❌ | ❌ |
| `run_bcs` | Binary Cuckoo Search (Lévy flight) | ✅ | ✅ |
| `run_bsos` | Binary Symbiotic Organisms Search | ✅ | ✅ |

(BPSO ve BGWO continuous-yerel mekanikler kullandığı için kategorik/karma
uzayda mantıklı değil — `NotImplementedError` fırlatır.)

### 3 Rehberlik Modu (her algoritmada)
- `method='classic'` — tournament/uniform random selection
- `method='fdb'` — RFDB rouletted (Aras 2021)
- `method='fdb_argmax'` — Klasik FDB argmax (Kahraman 2020)

### NSM (Natural Survivor Method) ve EPM (Evolutionary Population Management)
- `run_bga(use_nsm=True)` — sigmoid switch ile greedy ↔ NSM survivor selection
- `run_bga(use_epm=True)` — H1+H2+H3 epoch-based popülasyon yönetimi
- **Encoding-aware**: 4 encoding'te de çalışır (v2'de yalnızca binary'di)

### Multi-Objective: `run_nsga2`
- Fast non-dominated sort + crowding distance (Deb et al. 2002)
- Constrained NSGA-II desteği (Deb 2000)

### Native Constraint Handling: `make_constrained_fitness`
- 3 strateji: `feasibility_rules` (önerilen), `death_penalty`, `penalty`

### Parallel Fitness Evaluation
- `n_jobs > 1` ile multiprocessing.Pool
- Reproducibility korunur (aynı seed → aynı sonuç)

### Early Stopping & Checkpointing
- `early_stopping_patience=N` → N iter iyileşme yoksa erken dur
- `checkpoint_every=N + checkpoint_path='ckpt.pkl'` → periyodik save (Classic, NSM, EPM)
- `resume_from='ckpt.pkl'` → kaldığı yerden devam

### Logging (CSV) — v3-final-rc2 yeni
- `log_file='opt.csv', log_every=5` → her N iter durumu CSV'ye yaz
- Sütunlar: `iter, best_fit, mean_fit, std_fit, n_evaluations, elapsed_seconds`

### Auto-Tuning Meta-Loop — v3-final-rc2 yeni
- `tune_and_run(...)` — algoritma × method × pop_size kombinasyonlarını
  küçük bütçeyle dener, en iyi konfigürasyonu kalan bütçeyle final koşar.
- Hangi algoritmanın iyi olacağını bilmediğin yeni problemler için ideal.

### EPM-NSM Kombine Mod (B3) — v3-final-rc2 yeni
- `run_bga(use_nsm=True, use_epm=True)` artık çalışıyor.
- B3 hibrit fazı: EPM tabanı + fit-half'a sigmoid_switch ile NSM süzgeci.

---

## Kurulum

### Tek dosya (önerilen — bağımlılık yok, kolay entegrasyon)

```bash
# Sadece ana dosyayı projenize kopyalayın:
cp metaheuristics_bk_v3.py /yourproject/utils/

# Tek bağımlılık: numpy
pip install numpy
```

Kullanım:
```python
import sys; sys.path.append('/yourproject/utils')
import metaheuristics_bk_v3 as mh
```

### pip-style (lokal install)

```bash
git clone <your-private-repo-url> metaheuristics-bk
cd metaheuristics-bk
pip install -e .
```

---

## 5-Dakikalık Başlangıç

### A. Function Optimization (Continuous)

```python
import numpy as np
import metaheuristics_bk_v3 as mh

def rastrigin(x):
    """Multimodal function, global optimum f(0) = 0."""
    return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x)))

enc = mh.ContinuousEncoding(bounds=[(-5.12, 5.12)] * 10)
result = mh.run_bga(rastrigin, enc, pop_size=30, max_iter=200, seed=42)

print(f"Best fit: {result['best_fit']:.6f}")
print(f"Best sol: {result['best_sol']}")
```

### B. Feature Selection (Binary)

```python
def fs_fitness(mask):
    if mask.sum() == 0:
        return 1e6  # boş seçim cezası
    score = your_cv_function(X[:, mask.astype(bool)], y)
    sparsity_penalty = 0.01 * mask.sum() / len(mask)
    return -score + sparsity_penalty  # minimize

enc = mh.BinaryEncoding(n_features=X.shape[1])
result = mh.run_bga(fs_fitness, enc, method='fdb_argmax',
                    pop_size=20, max_iter=100, seed=42)
selected_features = result['best_sol'].astype(bool)
```

### C. TSP (Permutation)

```python
cities = np.random.uniform(0, 100, (20, 2))

def tour_length(perm):
    return sum(np.linalg.norm(cities[perm[i]] - cities[perm[(i+1) % len(perm)]])
               for i in range(len(perm)))

enc = mh.PermutationEncoding(n_items=20)
result = mh.run_babc(tour_length, enc, method='fdb',
                     pop_size=30, max_iter=100, seed=42)
print(f"En iyi tur: {result['best_sol'].tolist()}")
print(f"Uzunluk: {result['best_fit']:.2f}")
```

### D. Hyperparameter Tuning (Mixed)

```python
enc = mh.MixedEncoding({
    'lr':     mh.ContinuousEncoding([(0.001, 0.5)], log_scale=[True]),
    'depth':  mh.CategoricalEncoding([[3, 5, 7, 9, 11]]),
    'use_bn': mh.BinaryEncoding(n_features=1),
})

def hp_fitness(params):
    lr = params['lr'][0]
    depth = enc.components['depth'].decode(params['depth'])[0]
    use_bn = bool(params['use_bn'][0])
    return -train_and_evaluate(lr, depth, use_bn)  # minimize

result = mh.run_bga(hp_fitness, enc, pop_size=15, max_iter=50, seed=42)
```

---

## Kapsamlı API

### 1. Encoding Sınıfları

Her encoding aşağıdaki interface'i sağlar:

```python
class Encoding(ABC):
    @property
    def dim(self) -> int: ...
    
    def init_individual(self, rng) -> Any: ...
    def init_population(self, pop_size, rng) -> list: ...
    def crossover(self, rng, p1, p2) -> tuple: ...
    def mutate(self, rng, x, rate=None) -> Any: ...
    def distance(self, a, b) -> float: ...
    def population_center(self, population) -> Any:  # NSM için
        ...
```

#### Kendi Encoding'inizi Yazma

```python
class TimeSeriesEncoding(mh.Encoding):
    """Örnek: özel domain encoding."""
    def __init__(self, n_lags, max_lag=20):
        self.n_lags = n_lags
        self.max_lag = max_lag
    
    @property
    def dim(self):
        return self.n_lags
    
    def init_individual(self, rng):
        return np.sort(rng.choice(self.max_lag, self.n_lags, replace=False))
    
    def crossover(self, rng, p1, p2):
        # uniform crossover + sort + dedup mantığınızı yazın
        ...
    
    def mutate(self, rng, x, rate=None):
        # random reset of one lag
        ...
    
    def distance(self, a, b):
        return float(np.sum(a != b))
```

### 2. Single-Objective Algoritmalar

Tümü aynı API'ye sahip:

```python
result = mh.run_<alg>(
    fitness_fn,           # callable(solution) -> float (minimize)
    encoding,             # Encoding instance
    method='classic',     # 'classic' | 'fdb' | 'fdb_argmax'
    pop_size=20,
    max_iter=50,
    seed=42,
    verbose=False,
    # opsiyonel:
    n_jobs=1,                          # paralel eval (sadece run_bga)
    early_stopping_patience=None,      # erken dur (sadece run_bga)
    early_stopping_tol=1e-8,
    checkpoint_every=None,             # sadece classic mod
    checkpoint_path=None,
    resume_from=None,
)

# Result:
result['best_sol']        # En iyi çözüm (encoding tipinde)
result['best_fit']        # En iyi fitness
result['history']         # Her iter'in best_fit'i
result['n_evaluations']   # Toplam fitness eval sayısı
result['stopped_early']   # Early stop devreye girdi mi? (run_bga'da)
```

### 3. NSM ve EPM Modülleri

```python
# NSM — sigmoid switch ile greedy ↔ NSM survivor selection
result = mh.run_bga(
    fitness_fn, encoding,
    use_nsm=True,
    nsm_preset=6,           # 1..10 arası NSM_PRESETS preset'i
    nsm_merkez_sayisi=None, # None → max(2, pop_size//2)
    pop_size=20, max_iter=100,
)

# Sonuca özel NSM istatistikleri:
result['_nsm_applied']    # NSM uygulanan toplam denemesi
result['_nsm_accepted']   # NSM ile kabul edilen
result['_greedy_applied'] # Greedy karşılaştırma sayısı
result['_greedy_accepted']# Greedy ile kabul edilen

# EPM — H1+H2+H3 epoch-based popülasyon yönetimi
result = mh.run_bga(
    fitness_fn, encoding,
    use_epm=True,
    method='classic',  # Method-2 (önerilen) | 'fdb' = Method-1
    pop_size=20,       # ⚠️ çift sayı olmalı (yoksa otomatik +1)
    max_iter=100,
)
```

### 4. Multi-Objective (NSGA-II)

```python
def multi_obj(x):
    return np.array([rmse(x), n_features(x), -coverage(x)])

result = mh.run_nsga2(
    multi_obj, encoding,
    n_objectives=3,
    pop_size=50, max_iter=100,
    seed=42,
)

result['pareto_front']    # list of solutions
result['pareto_fitness']  # 2-D array (n, n_objectives)
```

### 5. Constraint Handling

```python
def base_fitness(x):
    return -float((x * values).sum())

def c_weight(x):
    return (x * weights).sum() - capacity  # <= 0 ise feasible

constrained_fit = mh.make_constrained_fitness(
    base_fitness,
    constraints=[c_weight],
    method='feasibility_rules',  # 'death_penalty' | 'penalty'
)

result = mh.run_bga(constrained_fit, enc, pop_size=20, max_iter=100, seed=42)
info = mh.evaluate_constraints(result['best_sol'], [c_weight])
print(info['feasible'], info['violations'])
```

### 6. Parallel Evaluation

```python
# fitness_fn pickle-able olmalı (top-level def, lambda DEĞİL)
def expensive_fit(x):
    # pahalı işlem (XGBoost CV vb.)
    return ...

result = mh.run_bga(
    expensive_fit, encoding,
    pop_size=30, max_iter=100,
    n_jobs=4,    # 4 worker; n_jobs=-1 → tüm CPU
)
# 4-5x hızlanma tipik (overhead'a bağlı)
# ⚠️ NSM modunda n_jobs ignore edilir (steady-state)
```

### 7. Early Stopping ve Checkpointing

```python
# Early stopping
result = mh.run_bga(
    fitness_fn, encoding,
    max_iter=10000,             # büyük sınır koy
    early_stopping_patience=20, # 20 iter iyileşme yoksa dur
    early_stopping_tol=1e-7,
)

# Checkpointing — uzun koşumlar (cluster, gece koşumu)
result = mh.run_bga(
    fitness_fn, encoding,
    max_iter=5000,
    checkpoint_every=100,
    checkpoint_path='/tmp/opt_state.pkl',
    verbose=True,
)

# Resume — server crash sonrası, ya da iter sayısını uzatmak için
result = mh.run_bga(
    fitness_fn, encoding,
    max_iter=10000,                       # daha büyük yeni hedef
    resume_from='/tmp/opt_state.pkl',     # önceki state'den devam
)
```

---

## Yeni Bir Projede Entegrasyon

### Pattern 1: Tek-dosya import (en basit)

```python
# my_project/optimizer.py
import sys
sys.path.insert(0, '/path/to/metaheuristics_bk_v3')
import metaheuristics_bk_v3 as mh

class HyperparamOptimizer:
    def __init__(self, model_class, X_train, y_train):
        self.model_class = model_class
        self.X = X_train
        self.y = y_train
        self.encoding = mh.MixedEncoding({
            'lr': mh.ContinuousEncoding([(1e-4, 1e-1)], log_scale=[True]),
            'l2': mh.ContinuousEncoding([(1e-6, 1e-1)], log_scale=[True]),
            'hidden': mh.CategoricalEncoding([[16, 32, 64, 128]]),
        })
    
    def fit_score(self, params):
        model = self.model_class(
            lr=params['lr'][0],
            l2=params['l2'][0],
            hidden=self.encoding.components['hidden'].decode(params['hidden'])[0],
        )
        model.fit(self.X, self.y)
        return -model.score(self.X, self.y)
    
    def optimize(self, budget=50):
        return mh.run_bga(
            self.fit_score, self.encoding,
            pop_size=10, max_iter=budget // 10,
            method='fdb_argmax', seed=42,
        )
```

### Pattern 2: Configurable wrapper

```python
def make_optimizer(problem_type, dim, **kwargs):
    """Factory: problem tipine göre encoding seç, BGA döndür."""
    encs = {
        'binary': mh.BinaryEncoding(dim),
        'continuous': mh.ContinuousEncoding([(-5, 5)] * dim),
        'permutation': mh.PermutationEncoding(dim),
    }
    enc = encs[problem_type]
    
    def runner(fitness_fn, **runner_kwargs):
        return mh.run_bga(fitness_fn, enc, **kwargs, **runner_kwargs)
    
    return runner

# Kullanım
opt = make_optimizer('continuous', dim=10, pop_size=20, max_iter=100,
                     use_nsm=True, n_jobs=4)
result = opt(my_fitness, seed=42, early_stopping_patience=15)
```

### Pattern 3: Sklearn-uyumlu BaseEstimator

```python
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

class MetaHeuristicSelector(BaseEstimator):
    def __init__(self, base_estimator, n_iter=50, pop_size=20,
                 alpha=0.85, method='fdb_argmax', seed=42):
        self.base_estimator = base_estimator
        self.n_iter = n_iter
        self.pop_size = pop_size
        self.alpha = alpha
        self.method = method
        self.seed = seed
    
    def fit(self, X, y):
        encoding = mh.BinaryEncoding(n_features=X.shape[1])
        
        def fitness(mask):
            if mask.sum() == 0:
                return 1e6
            sub = X[:, mask.astype(bool)]
            cv = cross_val_score(self.base_estimator, sub, y, cv=5).mean()
            return -self.alpha * cv + (1 - self.alpha) * mask.sum() / len(mask)
        
        result = mh.run_bga(
            fitness, encoding,
            method=self.method,
            pop_size=self.pop_size, max_iter=self.n_iter,
            seed=self.seed,
        )
        self.support_ = result['best_sol'].astype(bool)
        self.history_ = result['history']
        return self
    
    def transform(self, X):
        return X[:, self.support_]
```

---

## Mimari Tasarım

### Encoding-Agnostic Tasarım

Algoritmalar **solution tipini hiç bilmez**; tüm operatörleri encoding üzerinden çağırır:

```
[ Algoritma ]  ──crossover/mutate/distance──>  [ Encoding ]
     │                                                │
     │ (encoding-agnostic döngü mantığı)              │ (tip-spesifik operatörler)
     │                                                │
     v                                                v
   genel iter, parent select,              binary: bit-flip
   elitism, history tracking               continuous: SBX/polynomial
                                            categorical: random reset
                                            permutation: PMX/swap
                                            mixed: bileşene delegate
```

Yeni bir encoding eklediğinizde **tüm 7 algoritma + NSM + EPM otomatik destekler.**

### Modüler Bileşen Tasarımı

```
┌─────────────────────────────────────────────────┐
│ 1. ENCODING SINIFLARI (BÖLÜM 1)                 │
│    - Solution uzayını tanımlar                  │
│    - init/crossover/mutate/distance/center      │
└─────────────────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────┐
│ 2. FDB SKOR (BÖLÜM 2)                           │
│    - Encoding-aware mesafe + normalizasyon      │
└─────────────────────────────────────────────────┘
                       │
                       v
┌─────────────────────────────────────────────────┐
│ 3. ALGORITMALAR (BÖLÜM 3 + 4)                   │
│    BGA, BDE, BPSO, BABC, BGWO, BCS, BSOS        │
└─────────────────────────────────────────────────┘
              │                          │
              v                          v
┌────────────────────┐    ┌──────────────────────────┐
│ 5. NSGA-II         │    │ 6. NSM/EPM (BÖLÜM 6)     │
│    Multi-objective │    │    Survivor strategies   │
└────────────────────┘    └──────────────────────────┘
              │                          │
              └────────────┬─────────────┘
                           v
┌─────────────────────────────────────────────────┐
│ 7. CONSTRAINTS + parallel + early/checkpoint    │
│    (orthogonal — tüm algoritmaları sarmalar)    │
└─────────────────────────────────────────────────┘
```

---

## Test Etme

```bash
python tests/test_consistency.py
```

Beklenen çıktı: `TOPLAM: 73/73 ✓ Tüm testler başarılı!`

Test kapsamı:
- 7 algoritma × 4 encoding (24 başarılı + 4 beklenen NotImplementedError)
- NSGA-II ZDT1 Pareto front
- 3 constraint stratejisi
- Reproducibility (BGA, NSM, EPM, paralel modlar)
- 4 encoding'in `population_center` davranışı
- NSM/EPM × 4 encoding kombinasyonu
- NSM/EPM edge cases (invalid preset, kombinasyon yasağı, gbest koruma)
- PermutationEncoding (PMX, swap, Kendall tau, TSP koşum)
- Parallel evaluation (n_jobs=2 reproducibility)
- Early stopping (3 mod) + checkpoint resume

---

## Sürüm Geçmişi

| Sürüm | Tarih | İçerik |
|---|---|---|
| **v0.3.0-rc2** | 2026-05-10 | + EPM-NSM kombine mod (B3) + NSM/EPM checkpointing + Diğer alg n_jobs + Logging (CSV) + Auto-tuning (73/73 test) |
| v0.3.0-rc1 | 2026-05-10 | NSM/EPM port + BCS/BSOS + PermutationEncoding + Parallel + Early stop + Checkpointing (59/59 test) |
| v0.3.0-beta | 2026-05-10 | + 2 algoritma (BABC, BGWO) + NSGA-II + Constraints (21/21 test) |
| v0.3.0-alpha | 2026-05-10 | İlk universal sürüm: 4 encoding + 3 algoritma (11/11 test) |
| v0.2.0 | (önceki) | Binary + custom fitness |
| v0.1.0 | (önceki) | YKS feature selection (özelleşmiş) |

Detaylı liste: `CHANGELOG.md`

---

## Atıf

```bibtex
@misc{koc2026metaheuristics_v3final,
  author       = {Ko{\c{c}}, Baki},
  title        = {{metaheuristics\_bk v3-final}: Universal Encoding-Aware
                  Meta-Heuristic Library},
  year         = {2026},
  howpublished = {Bursa Teknik Üniversitesi, Bilgisayar Mühendisliği YL},
  note         = {7 algoritma × 5 encoding + NSM/EPM + NSGA-II}
}
```

## Lisans

MIT — özgürce kullan.

## Yazar

**Baki KOÇ** — Bursa Teknik Üniversitesi, Bilgisayar Mühendisliği YL, 2026
