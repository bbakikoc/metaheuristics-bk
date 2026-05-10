# 📘 Kullanıcı Rehberi — Veri Setinden Optimizasyon Sonucuna

> **Bu rehber:** Elinde bir CSV/Excel veri seti var ve metasezgisel bir
> algoritma ile optimizasyon yapmak istiyorsun. Bu doküman, "veri setini
> oku → problem tanımla → algoritma çalıştır → sonuç al" adımlarını
> kapsayan eksiksiz bir başvuru kaynağıdır.

---

## 📋 İçindekiler

- [Temel Mantık](#temel-mantık)
- [Karar Ağacı: Hangi Encoding?](#karar-ağacı-hangi-encoding)
- [Adım Adım: Yeni Bir Veri Seti İçin Optimizasyon](#adım-adım-yeni-bir-veri-seti-için-optimizasyon)
- [Tipik Senaryo 1: Feature Selection (CSV → Binary)](#senaryo-1-feature-selection-csv--binary)
- [Tipik Senaryo 2: Hyperparameter Tuning (Mixed)](#senaryo-2-hyperparameter-tuning-mixed)
- [Tipik Senaryo 3: TSP/Scheduling (Permutation)](#senaryo-3-tsp--scheduling-permutation)
- [Tipik Senaryo 4: Function Optimization (Continuous)](#senaryo-4-function-optimization-continuous)
- [Tipik Senaryo 5: Multi-Objective (Pareto Front)](#senaryo-5-multi-objective-pareto-front)
- [Tipik Senaryo 6: Constrained Optimization](#senaryo-6-constrained-optimization)
- [Sonuçları Kaydetme & Görselleştirme](#sonuçları-kaydetme--görselleştirme)
- [Algoritma & Method Seçim Tablosu](#algoritma--method-seçim-tablosu)
- [Sık Sorulan Sorular](#sık-sorulan-sorular)

---

## Temel Mantık

Kütüphane **veri setini değil, problemi** optimize eder. Her optimizasyon
problemi 3 şeyin tanımlanmasını ister:

```
┌──────────────────────────────────────────────────┐
│  1. ENCODING       → Solution nasıl kodlanır?    │
│                      (Binary? Continuous? vs.)    │
│  2. FITNESS_FN     → Bir solution ne kadar iyi?   │
│                      (callable: solution → float) │
│  3. ALGORITMA      → Hangi metasezgisel?          │
│                      (run_bga, run_bcs, vs.)      │
└──────────────────────────────────────────────────┘
```

Veri setini `fitness_fn` içinde okur veya kullanırsın — algoritma veri setini
direkt görmez. Bu yüzden tek bir kütüphaneyle TSP, feature selection, XGBoost
tuning, mühendislik tasarımı, hepsini optimize edebilirsin.

---

## Karar Ağacı: Hangi Encoding?

```
Solution ne tip bir şey?
│
├── Sabit uzunlukta 0/1 vektörü
│   (örn: feature seçimi, knapsack)
│   └─→ BinaryEncoding(n_features=N)
│
├── Sabit uzunlukta gerçek sayı vektörü
│   (örn: function optimization, ağırlıklar)
│   └─→ ContinuousEncoding(bounds=[(lo,hi),...])
│
├── Kategorik kutulardan seçim (her boyutta sonlu seçenek)
│   (örn: kernel='rbf'/'linear', activation='relu'/'tanh')
│   └─→ CategoricalEncoding(categories=[['a','b'],['x','y','z'],...])
│
├── Karışık tip (continuous + categorical + binary)
│   (örn: XGBoost: lr=float, depth=int, use_bn=bool)
│   └─→ MixedEncoding({'lr': ContinuousEncoding(...), 'depth': Categorical(...)})
│
└── Permütasyon: {0,1,...,n-1}'in bir sıralaması
    (örn: TSP, job scheduling, gezgin satıcı)
    └─→ PermutationEncoding(n_items=N)
```

---

## Adım Adım: Yeni Bir Veri Seti İçin Optimizasyon

### Şablon (her senaryoda tekrarlayacağın kalıp)

```python
import sys
sys.path.insert(0, '/path/to/metaheuristics-bk-v3-final')
import metaheuristics_bk_v3 as mh
import numpy as np
import pandas as pd

# ─── 1. Veri setini yükle ───────────────────────────────────
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# ─── 2. Encoding tanımla ────────────────────────────────────
# Senaryona göre seç (aşağıdaki bölümlere bak)
enc = mh.BinaryEncoding(n_features=X.shape[1])

# ─── 3. Fitness fonksiyonunu yaz ────────────────────────────
def fitness(solution):
    """solution: encoding'in döndürdüğü tip (örn: numpy array of 0/1)
       return: float (minimize edilecek değer)
    """
    # solution'u kullan, X ve y üzerinde değerlendir
    score = compute_something(solution, X, y)
    return -score  # küçük=iyi olduğu için negatif

# ─── 4. Algoritmayı çalıştır ────────────────────────────────
result = mh.run_bga(
    fitness, enc,
    pop_size=30, max_iter=100,
    method='fdb_argmax',
    seed=42, verbose=True,
)

# ─── 5. Sonucu kaydet/raporla ───────────────────────────────
print(f"Best fitness: {result['best_fit']}")
print(f"Best solution: {result['best_sol']}")
np.save('best_solution.npy', result['best_sol'])
np.save('history.npy', np.array(result['history']))
```

---

## Senaryo 1: Feature Selection (CSV → Binary)

**Problem:** Bir sınıflandırma/regresyon datasetinde "hangi özellikleri
seçmeliyim?" sorusu. CSV'de N feature varsa, çözüm uzayı `{0,1}^N`.

```python
import metaheuristics_bk_v3 as mh
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Veri seti
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# Encoding
enc = mh.BinaryEncoding(n_features=X.shape[1])

# Fitness: 5-fold CV accuracy + sparsity penalty
ALPHA = 0.85  # accuracy ağırlığı; (1-α) sparsity ağırlığı

def fitness(mask):
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return 1e6  # boş seçim cezası
    X_sub = X[:, mask_bool]
    cv_score = cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X_sub, y, cv=5, n_jobs=1,
    ).mean()
    sparsity = mask.sum() / len(mask)
    # Minimize: -accuracy + sparsity_penalty
    return -ALPHA * cv_score + (1 - ALPHA) * sparsity

# Çalıştır
result = mh.run_bga(
    fitness, enc,
    method='fdb_argmax',  # FDB rehberlik feature selection'da iyi
    pop_size=20, max_iter=50,
    seed=42, verbose=True,
    early_stopping_patience=10,  # 10 iter iyileşme yoksa dur
)

# Seçilen featurelar
selected_idx = np.where(result['best_sol'] == 1)[0]
selected_names = df.drop('target', axis=1).columns[selected_idx].tolist()
print(f"\nSeçilen {len(selected_names)} feature:")
print(selected_names)
print(f"CV accuracy: {-result['best_fit'] / ALPHA:.4f}")

# Kaydet
pd.DataFrame({
    'feature': df.drop('target', axis=1).columns,
    'selected': result['best_sol'].astype(int),
}).to_csv('feature_selection_result.csv', index=False)
```

> **Veri setini değiştirmek için:** Sadece `pd.read_csv` satırını değiştir.
> Her şey otomatik adapte olur. `n_features=X.shape[1]` boyutu otomatik alır.

---

## Senaryo 2: Hyperparameter Tuning (Mixed)

**Problem:** Bir ML modeli (XGBoost, RandomForest, NN) için en iyi
hiperparametreleri bul. Tipler karışık: `lr=float`, `max_depth=int`,
`use_bn=bool`.

```python
import metaheuristics_bk_v3 as mh
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Veri seti
X, y = ...  # senin veri setin

# MIXED encoding — her parametre kendi tipinde
enc = mh.MixedEncoding({
    'n_estimators': mh.CategoricalEncoding([[50, 100, 200, 500]]),
    'max_depth':    mh.CategoricalEncoding([[3, 5, 7, 10, None]]),
    'min_split':    mh.ContinuousEncoding([(2, 20)]),  # int olarak round'lanacak
    'bootstrap':    mh.BinaryEncoding(n_features=1),
})

def fitness(params):
    n_est = enc.components['n_estimators'].decode(params['n_estimators'])[0]
    depth = enc.components['max_depth'].decode(params['max_depth'])[0]
    min_split = int(round(params['min_split'][0]))
    bootstrap = bool(params['bootstrap'][0])
    
    model = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=depth,
        min_samples_split=min_split,
        bootstrap=bootstrap,
        random_state=42,
    )
    cv = cross_val_score(model, X, y, cv=5).mean()
    return -cv  # minimize

# Auto-tune ile hangi algoritmanın iyi olacağını otomatik bul
result = mh.tune_and_run(
    fitness, enc,
    algorithms=('bga', 'bcs', 'bsos'),
    methods=('classic', 'fdb'),
    pop_sizes=(10, 20),
    n_trials_per_config=2,
    budget=500,
    seed=42, verbose=True,
)

print(f"\nEn iyi config: {result['best_config']}")
print(f"En iyi CV accuracy: {-result['best_result']['best_fit']:.4f}")
print(f"En iyi parametreler: {result['best_result']['best_sol']}")
```

---

## Senaryo 3: TSP / Scheduling (Permutation)

**Problem:** N şehirli gezgin satıcı (TSP) — her tur şehirlerin bir
permütasyonu. CSV'de şehir koordinatları var.

```python
import metaheuristics_bk_v3 as mh
import numpy as np
import pandas as pd

# Veri seti: cities.csv → x, y koordinatları
df = pd.read_csv('cities.csv')  # kolonlar: city_id, x, y
COORDS = df[['x', 'y']].values
N = len(COORDS)

# Permütasyon encoding
enc = mh.PermutationEncoding(n_items=N)

# Fitness: tur uzunluğu (closed tour)
def tour_length(perm):
    total = 0.0
    for i in range(N):
        a = COORDS[perm[i]]
        b = COORDS[perm[(i + 1) % N]]
        total += np.linalg.norm(a - b)
    return float(total)

# BABC permütasyon uzayında iyi sonuç verir
result = mh.run_babc(
    tour_length, enc,
    method='fdb',
    pop_size=30, max_iter=200,
    seed=42, verbose=True,
    early_stopping_patience=20,
)

# Sonuç
best_perm = result['best_sol']
best_length = result['best_fit']
print(f"En kısa tur uzunluğu: {best_length:.2f}")
print(f"Şehir sırası: {best_perm.tolist()}")

# Tur'u CSV'ye kaydet (sıralı şekilde)
ordered_cities = df.iloc[best_perm].reset_index(drop=True)
ordered_cities['stop_order'] = np.arange(1, N + 1)
ordered_cities.to_csv('best_tour.csv', index=False)
```

---

## Senaryo 4: Function Optimization (Continuous)

**Problem:** Mühendislik/finans alanında bir matematiksel fonksiyonun
minimumunu bulma. Veri seti olmayabilir; fitness doğrudan formülden gelir.

```python
import metaheuristics_bk_v3 as mh
import numpy as np

# Rastrigin function (multimodal benchmark)
def rastrigin(x):
    A = 10.0
    n = len(x)
    return float(A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))

# 30D continuous, [-5.12, 5.12]
enc = mh.ContinuousEncoding(bounds=[(-5.12, 5.12)] * 30)

# BCS (Lévy flight) multimodal'da iyi
result = mh.run_bcs(
    rastrigin, enc,
    method='fdb',
    pop_size=40, max_iter=500,
    seed=42, verbose=True,
    early_stopping_patience=30,
)

print(f"Global minimum yaklaşımı: {result['best_fit']:.6f}")
print(f"x*: {result['best_sol']}")
```

---

## Senaryo 5: Multi-Objective (Pareto Front)

**Problem:** Birden fazla amaç var, biri için diğerini feda etmen gerek.
Örnek: feature selection'da "doğruluk artsın + feature sayısı azalsın".

```python
import metaheuristics_bk_v3 as mh
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Veri seti
X, y = ...

enc = mh.BinaryEncoding(n_features=X.shape[1])

def multi_obj(mask):
    """İki amaç döner: [hata oranı, feature sayısı oranı]"""
    if mask.sum() == 0:
        return np.array([1e6, 1e6])
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=30, random_state=42),
        X[:, mask.astype(bool)], y, cv=5,
    ).mean()
    error_rate = 1 - cv
    feature_ratio = mask.sum() / len(mask)
    return np.array([error_rate, feature_ratio])

result = mh.run_nsga2(
    multi_obj, enc,
    n_objectives=2,
    pop_size=50, max_iter=100,
    seed=42, verbose=True,
)

# Pareto front'u görselleştir
import matplotlib.pyplot as plt
front = np.array(result['pareto_fitness'])
plt.scatter(front[:, 0], front[:, 1])
plt.xlabel('Hata oranı')
plt.ylabel('Feature oranı')
plt.title(f'Pareto Front ({len(front)} solutions)')
plt.grid(True)
plt.savefig('pareto_front.png')

# Pareto solutions'ları kaydet
import pandas as pd
pareto_df = pd.DataFrame(result['pareto_fitness'],
                         columns=['error_rate', 'feature_ratio'])
pareto_df['n_features'] = [int(s.sum()) for s in result['pareto_front']]
pareto_df.to_csv('pareto_solutions.csv', index=False)
```

---

## Senaryo 6: Constrained Optimization

**Problem:** Knapsack tipinde — değer maksimize, ama kısıtlar var (ağırlık,
hacim, kategori limiti).

```python
import metaheuristics_bk_v3 as mh
import numpy as np
import pandas as pd

# Veri: items.csv → weight, value, category
df = pd.read_csv('items.csv')
weights = df['weight'].values
values = df['value'].values
categories = df['category'].values

CAPACITY_W = 100
MAX_PER_CATEGORY = {'electronics': 5, 'clothing': 8, 'food': 10}

enc = mh.BinaryEncoding(n_features=len(df))

def base_fitness(mask):
    return -float((mask * values).sum())  # negate ile minimize

# Kısıt 1: Ağırlık <= kapasite
def c_weight(mask):
    return (mask * weights).sum() - CAPACITY_W  # <= 0 ise feasible

# Kısıt 2: Her kategoriden max sayı
def c_category(mask):
    violations = []
    for cat, limit in MAX_PER_CATEGORY.items():
        cat_mask = (categories == cat)
        count = (mask * cat_mask).sum()
        violations.append(count - limit)
    return max(violations)  # En kötü ihlal

# Wrapper
constrained_fit = mh.make_constrained_fitness(
    base_fitness,
    constraints=[c_weight, c_category],
    method='feasibility_rules',  # Önerilen: feasible > infeasible her zaman
)

result = mh.run_bga(
    constrained_fit, enc,
    method='classic',
    pop_size=30, max_iter=100,
    seed=42, verbose=True,
)

# Çözümün feasibility raporu
info = mh.evaluate_constraints(result['best_sol'], [c_weight, c_category])
print(f"Feasible: {info['feasible']}")
print(f"Constraint violations: {info['violations']}")
print(f"Toplam değer: {-result['best_fit']:.2f}")

# Seçilen item'ları kaydet
selected = df[result['best_sol'].astype(bool)].copy()
selected.to_csv('selected_items.csv', index=False)
```

---

## Sonuçları Kaydetme & Görselleştirme

### A. Convergence eğrisi (her senaryoda işe yarar)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(result['history'])
plt.xlabel('Iterasyon')
plt.ylabel('En iyi fitness')
plt.title('Convergence Curve')
plt.grid(True)
plt.savefig('convergence.png', dpi=120)

# Veya CSV
import pandas as pd
pd.DataFrame({'iter': range(len(result['history'])),
              'best_fit': result['history']}).to_csv('history.csv', index=False)
```

### B. CSV log dosyası (`log_file` parametresi ile)

```python
result = mh.run_bga(
    fitness, enc,
    pop_size=30, max_iter=200, seed=42,
    log_file='optimization_log.csv',
    log_every=5,  # her 5 iter'de log yaz
)
# CSV içeriği: iter, best_fit, mean_fit, std_fit, n_evaluations, elapsed_seconds
```

### C. Birden fazla seed ile robust analiz

```python
results = []
for seed in range(10):
    r = mh.run_bga(fitness, enc, pop_size=20, max_iter=100, seed=seed)
    results.append(r['best_fit'])

print(f"Mean ± Std: {np.mean(results):.4f} ± {np.std(results):.4f}")
print(f"Min: {np.min(results):.4f}, Max: {np.max(results):.4f}")
```

### D. Algoritma karşılaştırması (boxplot)

```python
import matplotlib.pyplot as plt

algorithms = {
    'BGA': mh.run_bga,
    'BCS': mh.run_bcs,
    'BSOS': mh.run_bsos,
    'BABC': mh.run_babc,
}

results_dict = {name: [] for name in algorithms}
for seed in range(15):
    for name, runner in algorithms.items():
        r = runner(fitness, enc, pop_size=20, max_iter=80, seed=seed)
        results_dict[name].append(r['best_fit'])

plt.figure(figsize=(8, 5))
plt.boxplot(results_dict.values(), labels=results_dict.keys())
plt.ylabel('En iyi fitness (15 seed)')
plt.title('Algoritma Karşılaştırması')
plt.grid(True, alpha=0.3)
plt.savefig('algorithm_comparison.png', dpi=120)
```

---

## Algoritma & Method Seçim Tablosu

| Senaryo | Encoding | Önerilen Algoritma | Önerilen Method |
|---|---|---|---|
| Feature selection (kategorik veri) | Binary | `run_bga` veya `run_babc` | `fdb_argmax` |
| Knapsack / set cover | Binary | `run_bga` | `classic` |
| Function optimization (smooth) | Continuous | `run_bde` | `fdb` |
| Function optimization (multimodal) | Continuous | `run_bcs` | `fdb` |
| Hyperparameter tuning | Mixed | `run_bga` | `fdb_argmax` |
| TSP / scheduling | Permutation | `run_babc` veya `run_bsos` | `fdb` |
| Çok yerel optimum, global gerek | Continuous | `run_bcs` (Lévy) | `classic` |
| Sömürü/keşif dengesi gerek | Any | `run_bga(use_nsm=True)` | `fdb_argmax` |
| Bilinmiyor → otomatik bul | Any | `tune_and_run()` | otomatik |

**Method açıklaması:**
- `classic`: Tournament/uniform random (en hızlı, en basit)
- `fdb`: RFDB rouletted (Aras 2021) — orta zorlu problemler
- `fdb_argmax`: Klasik FDB argmax (Kahraman 2020) — sömürü ağırlıklı

**Genel kural:** Eğer hangi algoritmanın iyi olacağı belirsizse,
`tune_and_run()` ile birkaç tanesini otomatik kıyasla, en iyiyi seçtir.

---

## Sık Sorulan Sorular

### S: Pop_size ve max_iter ne olmalı?

Genel kural:
- **Hızlı prototip:** `pop_size=20, max_iter=50` (~1-5 sn)
- **Standart koşum:** `pop_size=30, max_iter=100` (~30 sn - 1 dk)
- **Final/yayın:** `pop_size=50, max_iter=300` + `early_stopping_patience=20`

Pop_size **dim'in 1.5-3 katı** olması iyi bir başlangıç.

### S: Reproducibility nasıl sağlanır?

`seed` parametresini sabit ver. Aynı seed → aynı sonuç. Birden fazla
bağımsız koşum için `seed=0,1,2,...` döngüsü kur.

### S: Çok pahalı bir fitness fonksiyonum var, ne yapmalıyım?

```python
result = mh.run_bga(
    expensive_fitness, enc,
    pop_size=20, max_iter=50,
    n_jobs=4,                       # paralel eval (4-5x hızlanma)
    early_stopping_patience=10,     # erken dur
    checkpoint_every=10,            # crash olursa devam et
    checkpoint_path='/tmp/opt.pkl',
    seed=42,
)
```

**Önemli:** `n_jobs > 1` için `expensive_fitness` pickle-able olmalı —
yani `def` ile top-level tanımlanmalı, `lambda` çalışmaz.

### S: Sürekli koşumda iter sayısını uzatabilir miyim?

```python
# İlk 50 iter
r1 = mh.run_bga(fn, enc, max_iter=50, seed=42,
                checkpoint_every=10, checkpoint_path='ckpt.pkl')

# Bakıyorum, daha iyileşebilir → 50 iter daha
r2 = mh.run_bga(fn, enc, max_iter=100, seed=42,
                resume_from='ckpt.pkl')  # 50'den 100'e devam
```

### S: NSM/EPM ne işe yarar, ne zaman kullanmalıyım?

- **NSM** (`use_nsm=True`): Yerel optimuma takılan koşumlarda ek çeşitlilik
  sağlar. Multimodal fonksiyonlar için iyi.
- **EPM** (`use_epm=True`): Forbidden-list mekaniği ile parent-child
  rekabetini düzenler. Binary uzayda özellikle etkili.
- **EPM-NSM kombine** (`use_nsm=True, use_epm=True`): Her ikisinin avantajını
  birleştirir, B3 hibrit fazı.

Belirsizsen klasik (`use_nsm=False, use_epm=False`) ile başla, yetersizse
NSM ekle.

### S: Sonuçlarımı bir pandas DataFrame'e nasıl koyarım?

```python
import pandas as pd

# Tek koşum
result = mh.run_bga(fn, enc, ...)
df_history = pd.DataFrame({
    'iter': range(len(result['history'])),
    'best_fit': result['history'],
})

# Çoklu seed × çoklu algoritma karşılaştırması
records = []
for alg_name, runner in [('BGA', mh.run_bga), ('BCS', mh.run_bcs)]:
    for seed in range(10):
        r = runner(fn, enc, pop_size=20, max_iter=50, seed=seed)
        records.append({
            'algorithm': alg_name,
            'seed': seed,
            'best_fit': r['best_fit'],
            'n_eval': r['n_evaluations'],
        })
df_compare = pd.DataFrame(records)
print(df_compare.groupby('algorithm')['best_fit'].agg(['mean', 'std', 'min']))
```

### S: Multi-class veya regresyon için fitness nasıl yazılır?

Aynı pattern. Sadece `cross_val_score`'un `scoring` parametresini değiştir:

```python
# Sınıflandırma
cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
fitness = -cv_score  # minimize için negatif

# Regresyon
cv_rmse = -cross_val_score(model, X, y, cv=5,
                            scoring='neg_root_mean_squared_error').mean()
fitness = cv_rmse  # zaten pozitif RMSE, minimize edilecek

# Multi-class
cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1_macro').mean()
fitness = -cv_f1
```

### S: Veri setim çok büyük (1M+ satır), ne yapmalıyım?

CV yerine validation split kullan, fitness hızlansın:

```python
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def fitness(mask):
    if mask.sum() == 0: return 1e6
    model = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42)
    model.fit(X_tr[:, mask.astype(bool)], y_tr)
    return -model.score(X_val[:, mask.astype(bool)], y_val)
```

---

## 🎯 Hızlı Başlangıç: Bir Fonksiyonun Tüm Adımları

```python
"""Tek bir templated örnek — kopyala, kendi datasetinle değiştir."""
import sys
sys.path.insert(0, '/path/to/metaheuristics-bk-v3-final')
import metaheuristics_bk_v3 as mh
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 1. VERİ
df = pd.read_csv('YOUR_DATA.csv')
X = df.drop('target', axis=1).values
y = df['target'].values

# 2. ENCODING
enc = mh.BinaryEncoding(n_features=X.shape[1])

# 3. FITNESS
def fitness(mask):
    if mask.sum() == 0:
        return 1e6
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X[:, mask.astype(bool)], y, cv=5,
    ).mean()
    return -0.85 * cv + 0.15 * (mask.sum() / len(mask))

# 4. ALGORITMA
result = mh.run_bga(
    fitness, enc,
    method='fdb_argmax',
    pop_size=30, max_iter=100,
    seed=42, verbose=True,
    early_stopping_patience=15,
    log_file='optim_log.csv',
)

# 5. KAYDET
selected = df.drop('target', axis=1).columns[result['best_sol'].astype(bool)].tolist()
print(f"Seçilen featurelar: {selected}")
print(f"CV accuracy: {-result['best_fit'] / 0.85:.4f}")

pd.DataFrame({
    'iter': range(len(result['history'])),
    'best_fit': result['history'],
}).to_csv('history.csv', index=False)
```

---

**Daha fazla örnek için:** `examples/` klasörüne bak — 10 farklı senaryo
çalışan halde mevcut.

**API referansı için:** `README.md` dosyasını incele.

**Kütüphane mimarisi için:** `RELEASE_NOTES.md`'i oku.
