"""
metaheuristics_bk v3.1 — Universal Meta-Heuristic Algorithms Library
=====================================================================

Encoding-agnostic metasezgisel optimizasyon kütüphanesi: 7 algoritma
(BGA, BDE, BPSO, BABC, BGWO, BCS, BSOS) + NSM/EPM rehberlik modülleri +
NSGA-II + constraint handling + veri-odaklı hızlı başlangıç
(`select_features`, `load_csv_dataset`).

Sürüm özeti
-----------
v1: Sadece feature selection (regression)         — özelleşmiş
v2: Binary uzayda her problem (custom fitness)    — kısmi genel
v3.0: 5 encoding × 7 algoritma + NSM/EPM + NSGA-II + constraints
v3.1: - Method adları netleşti: 'fdb' (deterministik, Kahraman 2020) ve
        'rfdb' (rulet, Aras 2021). Eski 'fdb_argmax' → 'fdb' olarak
        otomatik eşlenir; ESKİ 'fdb' (rulet) artık 'rfdb' adını taşır.
      - BGWO binary güncellemesi düzeltildi: V-shape transfer POZİSYONA
        değil ADIMA uygulanır + birey-bazlı greedy seçim (yakınsama hatası
        giderildi).
      - Fail-fast: bilinmeyen method değeri anında ValueError.
      - select_features / load_csv_dataset: herhangi bir CSV/DataFrame ile
        tek çağrıda özellik seçimi (sağlık verisi, oyun telemetrisi, vb.).
v0.3.2: BSOS continuous düzeltmesi (orijinal Cheng & Prayogo aritmetiği);
        MEALPY 3.0.3 ile çapraz doğrulandı.
v0.3.3: - ContinuousEncoding(distance_metric=...): 'euclidean' (varsayılan,
          geriye uyumlu) | 'manhattan' (resmi FDB-SOS kodu FEX #72311 ile
          birebir) | callable (özel metrik). Bilinmeyen değer raise.
        - run_bsos FDB davranış parametreleri: fdb_phases / fdb_refresh /
          fdb_exclude_self + fdb_reference preset'i (resmi kod hizası:
          yalnız-mutualism + seçim-başına skor + self dahil + Manhattan
          şartı). Doğrulama: validation/fdb_matlab_crosscheck.py.

Mimari
------
1. Encoding sınıfları (Binary, Continuous, Categorical, Mixed, Permutation):
   - Solution uzayını tanımlar
   - init / crossover / mutation / distance metodlarını sağlar

2. Üç rehberlik modu (VALID_METHODS):
   - 'classic' → Rehbersiz baz (turnuva / uniform-random seçim)
   - 'fdb'     → Klasik FDB (Kahraman et al. 2020): normFitness + normUzaklık
                 skoru EN YÜKSEK birey DETERMİNİSTİK rehber olur
   - 'rfdb'    → Rulet-FDB (Aras et al. 2021): FDB skorları rulet ağırlığı
                 olur, rehber OLASILIKSAL seçilir

3. Universal run_* fonksiyonları:
   - run_bga / run_bde / run_bpso / run_babc / run_bgwo / run_bcs / run_bsos

Kullanım
--------
>>> from metaheuristics_bk_v3 import run_bga, BinaryEncoding
>>>
>>> def my_fitness(solution):
...     return -solution.sum()  # max ones
>>>
>>> result = run_bga(
...     fitness_fn=my_fitness,
...     encoding=BinaryEncoding(n_features=20),
...     method='fdb',
...     pop_size=20, max_iter=50, seed=42,
... )

Veri-odaklı hızlı başlangıç (CSV → özellik seçimi):
>>> from metaheuristics_bk_v3 import load_csv_dataset, select_features
>>> X, y = load_csv_dataset("hasta_verisi.csv", target="teshis")
>>> out = select_features(X, y, algorithm='bga', method='fdb')
>>> out['selected']  # seçilen sütun adları

Yazar:  Baki KOÇ (Bursa Teknik Üniversitesi, 2026)
Lisans: MIT
Versiyon: 0.3.1
GitHub: https://github.com/bbakikoc/metaheuristics-bk
"""

__version__ = "0.3.5"
__author__ = "Baki Koç"
__license__ = "MIT"

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import Callable, Union, List, Dict, Optional, Any


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 1: ENCODING SINIFLARI
# ═══════════════════════════════════════════════════════════════════════════
# Bir Encoding, solution uzayının nasıl olduğunu ve nasıl manipüle
# edileceğini bilir. Algoritmalar Encoding'e "rastgele birey üret",
# "iki ebeveyni çaprazla", "mutasyon yap", "iki birey arası mesafe ne?"
# diye sorar — encoding cevap verir.
# ═══════════════════════════════════════════════════════════════════════════


class Encoding(ABC):
    """Solution uzayını tanımlayan ve manipüle eden soyut taban sınıf.
    
    Her encoding kendi `init`, `crossover`, `mutate`, `distance` metodlarını
    sağlar. Algoritmalar encoding-agnostic kalır.
    
    Subclass'lar
    ------------
    BinaryEncoding      : {0, 1}^n
    ContinuousEncoding  : [low, high]^n
    CategoricalEncoding : {a, b, c, ...}^n
    MixedEncoding       : Yukarıdakilerin kombinasyonu (dict)
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Solution uzayının boyutu."""
        pass
    
    @abstractmethod
    def init_individual(self, rng: np.random.RandomState) -> Any:
        """Rastgele bir birey üretir."""
        pass
    
    @abstractmethod
    def crossover(self, rng: np.random.RandomState, p1: Any, p2: Any):
        """İki ebeveynden iki yavru üretir. (c1, c2) döner."""
        pass
    
    @abstractmethod
    def mutate(self, rng: np.random.RandomState, x: Any, rate: float = None) -> Any:
        """Bir bireye mutasyon uygular (yeni copy döner)."""
        pass
    
    @abstractmethod
    def distance(self, a: Any, b: Any) -> float:
        """İki birey arası mesafe (FDB için lazım)."""
        pass
    
    def init_population(self, pop_size: int, rng: np.random.RandomState) -> list:
        """Popülasyon üretir (default: tek tek init_individual çağırır)."""
        return [self.init_individual(rng) for _ in range(pop_size)]
    
    def population_center(self, population: list) -> Any:
        """Bir popülasyonun "merkezini" hesaplar (NSM modülü tarafından kullanılır).
        
        Tip-spesifik anlam:
            Binary       → elementwise mod (en sık değer; bağ varsa 1)
            Continuous   → elementwise aritmetik ortalama (centroid)
            Categorical  → her boyutta en sık görülen kategori indeksi
            Mixed        → her bileşenin kendi population_center'ı
        
        Subclass'lar bu metodu override etmeli (default: KeyError fırlatır).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} population_center metodunu implement etmeli "
            f"(NSM modülünü kullanmak için)."
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


class BinaryEncoding(Encoding):
    """İkili (binary) encoding: {0, 1}^n.
    
    Klasik feature selection, knapsack, set cover gibi problemler için.
    
    Parameters
    ----------
    n_features : int
        Bit-uzayı boyutu.
    init_p : float, default=0.4
        Başlangıç popülasyonunda her bitin 1 olma olasılığı.
    ensure_one : bool, default=True
        Tüm-sıfır bireylere rastgele bir bit aç (degenerate çözümleri önlemek için).
    """
    
    def __init__(self, n_features: int, init_p: float = 0.4, ensure_one: bool = True):
        self.n_features = n_features
        self.init_p = init_p
        self.ensure_one = ensure_one
    
    @property
    def dim(self):
        return self.n_features
    
    def init_individual(self, rng):
        s = rng.choice([0, 1], size=self.n_features,
                       p=[1 - self.init_p, self.init_p]).astype(int)
        if self.ensure_one and s.sum() == 0:
            s[rng.randint(0, self.n_features)] = 1
        return s
    
    def crossover(self, rng, p1, p2):
        """Uniform crossover (p_swap=0.5)."""
        mask = rng.rand(self.n_features) < 0.5
        c1 = np.where(mask, p2, p1).astype(int)
        c2 = np.where(mask, p1, p2).astype(int)
        return c1, c2
    
    def mutate(self, rng, x, rate=None):
        """Bit-flip mutation; default rate = 1/n."""
        if rate is None:
            rate = 1.0 / self.n_features
        mask = rng.rand(self.n_features) < rate
        new_x = x.copy()
        new_x[mask] ^= 1
        if self.ensure_one and new_x.sum() == 0:
            new_x[rng.randint(0, self.n_features)] = 1
        return new_x
    
    def distance(self, a, b):
        """Hamming distance."""
        return float(np.sum(a != b))
    
    def population_center(self, population):
        """Elementwise mod (>= 0.5 olanlar 1, diğerleri 0). Tie → 1.
        
        v2 NSM'deki `_pop_merkez` ile uyumlu (ortalama ≥ 0.5 → 1).
        """
        arr = np.asarray(population, dtype=float)  # (pop_size, n_features)
        return (arr.mean(axis=0) >= 0.5).astype(int)


class ContinuousEncoding(Encoding):
    """Sürekli encoding: [low, high]^n.
    
    Hyperparameter tuning (sürekli parametreler), function optimization
    (Rastrigin, Ackley, Sphere) gibi problemler için.
    
    Parameters
    ----------
    bounds : list of (float, float)
        Her boyut için (low, high) aralığı. Boyut sayısı bunun uzunluğudur.
    log_scale : list of bool, optional
        Hangi boyutların log-uniform örneklenmesi gerektiği (örn: learning rate).
    distance_metric : {'euclidean', 'manhattan'} veya callable, default='euclidean'
        FDB/NSM gibi mesafe kullanan katmanların metriği (v0.3.3).
        - 'euclidean': aralık-normalize Öklid (v3 varsayılan davranışı — geriye
          uyumlu). KBS 2020 makale metnindeki örnek olayla (Fig. 1b) eşleşir.
        - 'manhattan': HAM |a-b| toplamı — resmi FDB-SOS MATLAB kodu
          (FEX #72311, fitnessDistanceBalance.m) ile BİREBİR.
        - callable(a, b) -> float: özel/uyarlanmış metrik (projeye göre).
        Bilinmeyen değer ValueError üretir (fail-fast, faz3 §3.5).
    """
    
    _VALID_METRICS = ('euclidean', 'manhattan')
    
    def __init__(self, bounds, log_scale=None, distance_metric='euclidean'):
        self.bounds = np.array(bounds, dtype=float)
        self.n_dim = len(bounds)
        if callable(distance_metric):
            self.distance_metric = 'custom'
            self._custom_distance = distance_metric
        elif distance_metric in self._VALID_METRICS:
            self.distance_metric = distance_metric
            self._custom_distance = None
        else:
            raise ValueError(
                f"Bilinmeyen distance_metric: {distance_metric!r}. "
                f"Geçerli: {self._VALID_METRICS} veya callable(a, b) -> float."
            )
        if log_scale is None:
            log_scale = [False] * self.n_dim
        self.log_scale = np.array(log_scale, dtype=bool)
        # Sanity check: log-scale boyutlarda alt sınır > 0 olmalı
        for i, is_log in enumerate(self.log_scale):
            if is_log and self.bounds[i, 0] <= 0:
                raise ValueError(f"Boyut {i} log-scale ama lower bound {self.bounds[i, 0]} <= 0")
    
    @property
    def dim(self):
        return self.n_dim
    
    def init_individual(self, rng):
        x = np.empty(self.n_dim)
        for i in range(self.n_dim):
            lo, hi = self.bounds[i]
            if self.log_scale[i]:
                x[i] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
            else:
                x[i] = rng.uniform(lo, hi)
        return x
    
    def crossover(self, rng, p1, p2):
        """SBX (Simulated Binary Crossover) — sürekli uzay için standart.
        
        Deb & Agrawal (1995). η_c = 20 (literatür default).
        """
        eta = 20.0
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        for i in range(self.n_dim):
            if rng.rand() < 0.5 and abs(p1[i] - p2[i]) > 1e-14:
                u = rng.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (eta + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
            else:
                c1[i], c2[i] = p1[i], p2[i]
        # Sınırla
        c1 = np.clip(c1, self.bounds[:, 0], self.bounds[:, 1])
        c2 = np.clip(c2, self.bounds[:, 0], self.bounds[:, 1])
        return c1, c2
    
    def mutate(self, rng, x, rate=None):
        """Polynomial mutation (η_m = 20).
        
        Deb (2001). Gauss yerine polynomial: bound'lara saygılı.
        """
        if rate is None:
            rate = 1.0 / self.n_dim
        eta_m = 20.0
        new_x = x.copy()
        for i in range(self.n_dim):
            if rng.rand() < rate:
                lo, hi = self.bounds[i]
                u = rng.rand()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    new_x[i] = x[i] + delta * (x[i] - lo)
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    new_x[i] = x[i] + delta * (hi - x[i])
                new_x[i] = np.clip(new_x[i], lo, hi)
        return new_x
    
    def distance(self, a, b):
        """Metrik-farkındalı mesafe (bkz. distance_metric parametresi)."""
        if self._custom_distance is not None:
            return float(self._custom_distance(a, b))
        if self.distance_metric == 'manhattan':
            # HAM Manhattan — resmi FDB-SOS koduyla (FEX #72311) birebir.
            return float(np.abs(np.asarray(a) - np.asarray(b)).sum())
        # 'euclidean' → aralık-normalize Öklid (v3 varsayılanı, geriye uyumlu)
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        ranges = np.where(ranges == 0, 1.0, ranges)  # 0'a bölmeyi önle
        normalized_diff = (a - b) / ranges
        return float(np.sqrt((normalized_diff ** 2).sum()))
    
    def population_center(self, population):
        """Aritmetik ortalama (centroid). Bound'lara göre kırpılır."""
        arr = np.asarray(population, dtype=float)
        center = arr.mean(axis=0)
        return np.clip(center, self.bounds[:, 0], self.bounds[:, 1])


class CategoricalEncoding(Encoding):
    """Kategorik encoding: her boyut belirli kategorilerden birini alır.
    
    Hyperparameter tuning (kernel='rbf'/'poly', activation='relu'/'tanh'),
    yapısal kararlar için.
    
    Parameters
    ----------
    categories : list of list
        Her boyut için kategoriler listesi. 
        Örnek: [['relu', 'tanh', 'sigmoid'], [16, 32, 64, 128]]
    """
    
    def __init__(self, categories):
        self.categories = [list(c) for c in categories]
        self.n_dim = len(categories)
        self.n_cats = [len(c) for c in self.categories]
    
    @property
    def dim(self):
        return self.n_dim
    
    def init_individual(self, rng):
        # int array: her boyut için kategori indeksi
        return np.array([rng.randint(0, n) for n in self.n_cats], dtype=int)
    
    def crossover(self, rng, p1, p2):
        """Uniform crossover (kategori indeksleri üzerinde)."""
        mask = rng.rand(self.n_dim) < 0.5
        c1 = np.where(mask, p2, p1).astype(int)
        c2 = np.where(mask, p1, p2).astype(int)
        return c1, c2
    
    def mutate(self, rng, x, rate=None):
        """Random reset mutation: rate olasılıkla kategori değiştir."""
        if rate is None:
            rate = 1.0 / self.n_dim
        new_x = x.copy()
        for i in range(self.n_dim):
            if rng.rand() < rate and self.n_cats[i] > 1:
                # Mevcut kategoriden farklı birini seç
                avail = list(range(self.n_cats[i]))
                avail.remove(int(x[i]))
                new_x[i] = avail[rng.randint(0, len(avail))]
        return new_x
    
    def distance(self, a, b):
        """Normalized Hamming distance (kaç boyut farklı / toplam)."""
        return float(np.sum(a != b)) / self.n_dim
    
    def population_center(self, population):
        """Her boyutta en sık görülen kategori indeksi (mod). Tie → en küçük indeks."""
        arr = np.asarray(population, dtype=int)  # (pop_size, n_dim)
        center = np.empty(self.n_dim, dtype=int)
        for d in range(self.n_dim):
            counts = np.bincount(arr[:, d], minlength=self.n_cats[d])
            center[d] = int(np.argmax(counts))
        return center
    
    def decode(self, x):
        """Indeks dizisinden gerçek kategori değerlerine çevir."""
        return [self.categories[i][int(x[i])] for i in range(self.n_dim)]


class MixedEncoding(Encoding):
    """Karma encoding: Binary + Continuous + Categorical alt-uzayları kombine eder.
    
    Hyperparameter tuning gibi karma tip problemlerde kullanılır.
    Solution bir dict olarak temsil edilir; her alt-encoding ile çağrı yapılır.
    
    Parameters
    ----------
    components : dict[str, Encoding]
        Her bir alt-uzay için isim → encoding eşlemesi.
        Örnek: {'lr': ContinuousEncoding([(0.001, 0.1)]),
                'depth': CategoricalEncoding([[3, 5, 7]]),
                'use_bn': BinaryEncoding(n_features=1)}
    """
    
    def __init__(self, components: Dict[str, Encoding]):
        self.components = components
        self.names = list(components.keys())
    
    @property
    def dim(self):
        return sum(e.dim for e in self.components.values())
    
    def init_individual(self, rng):
        return {name: enc.init_individual(rng) for name, enc in self.components.items()}
    
    def crossover(self, rng, p1, p2):
        c1, c2 = {}, {}
        for name, enc in self.components.items():
            c1[name], c2[name] = enc.crossover(rng, p1[name], p2[name])
        return c1, c2
    
    def mutate(self, rng, x, rate=None):
        new_x = {}
        for name, enc in self.components.items():
            new_x[name] = enc.mutate(rng, x[name], rate)
        return new_x
    
    def distance(self, a, b):
        """Tüm alt-encoding'lerin distance toplamı (basit kombinasyon)."""
        return sum(enc.distance(a[name], b[name]) for name, enc in self.components.items())
    
    def population_center(self, population):
        """Her bileşen için ayrı ayrı population_center hesapla."""
        return {
            name: enc.population_center([ind[name] for ind in population])
            for name, enc in self.components.items()
        }
    
    def __repr__(self):
        return f"MixedEncoding({list(self.components.keys())})"


class PermutationEncoding(Encoding):
    """Permütasyon encoding: {0, 1, ..., n-1}'in bir sıralaması.
    
    TSP (Travelling Salesman Problem), job-shop scheduling, sequence
    optimization gibi her bireyin bir permütasyon olduğu problemler için.
    
    Operatörler permütasyon-koruyucu:
        crossover: PMX (Partially Matched Crossover, Goldberg & Lingle 1985)
        mutate   : swap mutation (rate olasılıkla her pozisyon başka biriyle takas)
        distance : Kendall tau (sıralı ikili sayısı / toplam ikili)
    
    Parameters
    ----------
    n_items : int
        Permütasyon uzunluğu (örn. TSP'de şehir sayısı).
    
    Examples
    --------
    >>> enc = PermutationEncoding(n_items=10)
    >>> rng = np.random.RandomState(0)
    >>> p = enc.init_individual(rng)
    >>> sorted(p.tolist()) == list(range(10))  # gerçek bir permütasyon
    True
    """
    
    def __init__(self, n_items: int):
        if n_items < 2:
            raise ValueError(f"PermutationEncoding n_items >= 2 olmalı (verildi: {n_items})")
        self.n_items = n_items
    
    @property
    def dim(self):
        return self.n_items
    
    def init_individual(self, rng):
        p = np.arange(self.n_items, dtype=int)
        rng.shuffle(p)
        return p
    
    def crossover(self, rng, p1, p2):
        """PMX (Partially Matched Crossover).
        
        Goldberg & Lingle (1985). İki kesim noktası arası bölge p2'den
        kopyalanır, dışarısı p1'den ama çakışmayı önleyen mapping ile.
        Yavrular yine geçerli permütasyondur.
        """
        n = self.n_items
        c1 = np.full(n, -1, dtype=int)
        c2 = np.full(n, -1, dtype=int)
        
        # 2 kesim noktası
        a, b = sorted(rng.choice(n, 2, replace=False))
        if a == b:
            b = min(n - 1, a + 1)
            if a == b:  # n=2 edge case
                a = 0
        
        # Orta segment: p2'den c1'e, p1'den c2'ye
        c1[a:b+1] = p2[a:b+1]
        c2[a:b+1] = p1[a:b+1]
        
        # Mapping: c1'in orta segmentindeki her item için p1'deki karşılığı bul
        def fill_outside(child, donor, mid_segment_other):
            """Orta segment dışını donor'dan doldur, çakışma varsa mapping uygula."""
            for i in list(range(0, a)) + list(range(b+1, n)):
                val = donor[i]
                # Çakışma: val zaten orta segmentte mi?
                while val in mid_segment_other:
                    # Map: mid_segment_other'da val'in indeksini bul, oradan donor'da karşılığı al
                    idx_in_mid = np.where(mid_segment_other == val)[0][0] + a
                    val = donor[idx_in_mid]
                child[i] = val
        
        fill_outside(c1, p1, c1[a:b+1])
        fill_outside(c2, p2, c2[a:b+1])
        
        return c1, c2
    
    def mutate(self, rng, x, rate=None):
        """Swap mutation: rate olasılıkla her pozisyonu başka biriyle takas et.
        
        Permütasyon korunur. Eğer rate çok düşükse hiç swap olmayabilir;
        en az 1 swap garanti edilir (rate > 0 ise).
        """
        if rate is None:
            rate = 1.0 / self.n_items
        new_x = x.copy()
        any_swap = False
        for i in range(self.n_items):
            if rng.rand() < rate:
                j = rng.randint(0, self.n_items)
                new_x[i], new_x[j] = new_x[j], new_x[i]
                any_swap = True
        # Garanti: rate > 0 ise en az 1 swap
        if not any_swap and rate > 0:
            i, j = rng.choice(self.n_items, 2, replace=False)
            new_x[i], new_x[j] = new_x[j], new_x[i]
        return new_x
    
    def distance(self, a, b):
        """Normalized Kendall tau distance.
        
        İki permütasyon arasındaki "discordant pair" sayısının max'a bölümü.
        Aynıysa 0, tam ters çevrilmişse 1.
        
        O(n^2) — büyük n için pahalı; n>100'da bootstrap düşünülebilir
        ama şu an basit ve doğru.
        """
        n = self.n_items
        # a ve b'deki her elementin pozisyonu
        pos_a = np.argsort(a)
        pos_b = np.argsort(b)
        # Discordant pairs sayısı
        disc = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Aynı çift (i,j) için a'da ve b'de göreli sırası farklı mı?
                if (pos_a[i] < pos_a[j]) != (pos_b[i] < pos_b[j]):
                    disc += 1
        max_disc = n * (n - 1) / 2
        return float(disc) / max_disc if max_disc > 0 else 0.0
    
    def population_center(self, population):
        """Permütasyon merkezi: medoid (popülasyon içinden mesafe-ortalaması en küçük).
        
        Permütasyon uzayında "merkez" gerçek bir permütasyon olmalı; mean
        anlamsız. Bu yüzden medoid (population içinden seçilen merkez) kullanılır.
        """
        n_pop = len(population)
        if n_pop == 0:
            return None
        if n_pop == 1:
            return population[0].copy()
        # Tüm çiftler arası mesafe matrisi
        dist_sums = np.zeros(n_pop)
        for i in range(n_pop):
            for j in range(n_pop):
                if i != j:
                    dist_sums[i] += self.distance(population[i], population[j])
        return population[int(np.argmin(dist_sums))].copy()
    
    def __repr__(self):
        return f"PermutationEncoding(n_items={self.n_items})"


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 2: REHBERLİK METHOD'LARI ve FDB SKOR
# ═══════════════════════════════════════════════════════════════════════════
# v3.1 adlandırma sözleşmesi (KALICI — bir daha değişmeyecek):
#   'classic' → rehbersiz baz
#   'fdb'     → Klasik FDB (Kahraman et al. 2020, KBS 190:105169):
#               skor = normFitness + normUzaklık(best'e); EN YÜKSEK skorlu
#               birey DETERMİNİSTİK olarak rehber seçilir.
#   'rfdb'    → Rulet-FDB (Aras et al. 2021): aynı skor vektörü rulet
#               ağırlığı olur; rehber OLASILIKSAL seçilir.
# Eski adlar: v3.0'da 'fdb_argmax' = bugünkü 'fdb', v3.0'da 'fdb' = bugünkü
# 'rfdb'. 'fdb_argmax' geriye-uyum için otomatik eşlenir; ESKİ 'fdb'
# kullanan v3.0 scriptleri v3.1'de FARKLI (deterministik) davranır —
# CHANGELOG'daki kırıcı değişiklik notuna bak.
# ═══════════════════════════════════════════════════════════════════════════

VALID_METHODS = ('classic', 'fdb', 'rfdb')

_METHOD_ALIASES = {
    'fdb_argmax': 'fdb',   # v3.0 eski adı → v3.1 kanonik adı
}

# Kullanım sırasında bir kez basılan hatırlatma metinleri
_METHOD_HINTS = {
    'classic': ("[hatırlatma] method='classic' → rehbersiz baz: seçim "
                "turnuva/uniform-random, FDB skoru devrede değil."),
    'fdb':     ("[hatırlatma] method='fdb' → Klasik FDB (Kahraman 2020): "
                "normFitness + best'e-normUzaklık toplamı EN YÜKSEK olan birey "
                "DETERMİNİSTİK rehber olur. (v3.0'daki adı: 'fdb_argmax')"),
    'rfdb':    ("[hatırlatma] method='rfdb' → Rulet-FDB (Aras 2021): FDB "
                "skorları rulet ağırlığıdır, rehber OLASILIKSAL seçilir. "
                "(v3.0'daki adı: 'fdb')"),
}

#: Hatırlatma mesajlarını tamamen kapatmak için False yap.
SHOW_METHOD_HINTS = True

_shown_hints: set = set()
_shown_alias_warnings: set = set()


def _normalize_method(method: str) -> str:
    """Method adını doğrular, eski adları eşler, bilinmeyende raise eder.

    Fail-fast sözleşmesi: bilinmeyen bir method değeri ASLA sessizce bir
    varsayılana düşmez — anında ValueError. (Yazım hatası 'fbd' gibi
    değerlerin sessizce yanlış mekanizma koşturması v3.0'da mümkündü.)
    """
    if not isinstance(method, str):
        raise ValueError(f"method bir string olmalı, verilen: {type(method).__name__}")

    if method in _METHOD_ALIASES:
        canon = _METHOD_ALIASES[method]
        if method not in _shown_alias_warnings:
            print(f"[uyarı] method='{method}' eski (v3.0) ad — yeni adı "
                  f"'{canon}'. Otomatik eşlendi; scriptini güncellemen önerilir.")
            _shown_alias_warnings.add(method)
        method = canon

    if method not in VALID_METHODS:
        raise ValueError(
            f"Bilinmeyen method: '{method}'. Geçerli değerler: {VALID_METHODS}. "
            f"Not: v3.0'daki 'fdb' (rulet) artık 'rfdb'; v3.0'daki "
            f"'fdb_argmax' (deterministik) artık 'fdb'."
        )

    if SHOW_METHOD_HINTS and method not in _shown_hints:
        print(_METHOD_HINTS[method])
        _shown_hints.add(method)

    return method


def fdb_scores(population, fitness_vals, encoding: Encoding, method='rfdb'):
    """Encoding-aware FDB rehberlik skoru.

    Mesafe metriği encoding'in distance() metodundan gelir; nf+nd formülü
    (Kahraman et al. 2020'nin eşit-ağırlıklı FDB skoru ile orantılı) aynı
    kalır ama Hamming yerine encoding-spesifik mesafe kullanılır.

    Parameters
    ----------
    population : list of individuals
    fitness_vals : 1-D array of fitness values
    encoding : Encoding
    method : {'classic', 'fdb', 'rfdb'}
        'classic' → tüm skorlar 1 (rehberlik yok)
        'fdb'     → one-hot: en yüksek FDB skorlu birey 1, diğerleri 0
                    (deterministik rehber; Kahraman 2020)
        'rfdb'    → ham FDB skor vektörü (rulet ağırlığı; Aras 2021)

    Returns
    -------
    1-D array of scores

    Raises
    ------
    ValueError : bilinmeyen method değerinde (fail-fast).
    """
    method = _normalize_method(method)
    ps = len(population)
    if method == 'classic':
        return np.ones(ps)

    fmin, fmax = float(fitness_vals.min()), float(fitness_vals.max())
    nf = np.ones(ps) if fmax == fmin else 1 - (fitness_vals - fmin) / (fmax - fmin)

    best_idx = int(np.argmin(fitness_vals))
    best = population[best_idx]
    dist = np.array([encoding.distance(s, best) for s in population], dtype=float)
    dmin, dmax = dist.min(), dist.max()
    nd = np.zeros(ps) if dmax == dmin else (dist - dmin) / (dmax - dmin)

    raw = nf + nd
    if method == 'fdb':
        out = np.zeros(ps)
        out[int(np.argmax(raw))] = 1.0
        return out
    return raw  # method == 'rfdb'


def _selection_probs(scores):
    s = np.maximum(scores, 0.0)
    total = s.sum()
    if total <= 1e-12:
        return np.ones(len(s)) / len(s)
    return s / total


def _tournament(rng, fit, k=3):
    cands = rng.choice(len(fit), k, replace=False)
    return int(cands[np.argmin(fit[cands])])


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 3: UNIVERSAL ALGORITMALAR (encoding-agnostic)
# ═══════════════════════════════════════════════════════════════════════════


def run_bga(fitness_fn: Callable, encoding: Encoding,
            method: str = 'classic',
            pop_size: int = 20, max_iter: int = 50,
            tournament_k: int = 3, p_crossover: float = 0.8,
            mutation_rate: Optional[float] = None,
            seed: int = 42, verbose: bool = False,
            # ─── v3-final: NSM/EPM opsiyonel parametreler ─────────────────
            use_nsm: bool = False,
            use_epm: bool = False,
            nsm_preset: int = 6,
            nsm_merkez_sayisi: Optional[int] = None,
            # ─── v3-final: Parallel evaluation ─────────────────────────────
            n_jobs: int = 1,
            # ─── v3-final: Early stopping ─────────────────────────────────
            early_stopping_patience: Optional[int] = None,
            early_stopping_tol: float = 1e-8,
            # ─── v3-final: Checkpointing ──────────────────────────────────
            checkpoint_every: Optional[int] = None,
            checkpoint_path: Optional[str] = None,
            resume_from: Optional[str] = None,
            # ─── v3-final-rc2: Logging ────────────────────────────────────
            log_file: Optional[str] = None,
            log_every: int = 1):
    """Universal Genetic Algorithm — herhangi bir encoding üzerinde çalışır.
    
    3 mod halinde çalışır (parametre kombinasyonuna göre):
        use_nsm=False, use_epm=False (default) → klasik BGA (geriye uyumlu)
        use_nsm=True,  use_epm=False           → BGA+NSM (steady-state, sigmoid switch)
        use_nsm=False, use_epm=True            → BGA+EPM (epoch-based, H1+H2+H3)
    
    `use_nsm=True` ve `use_epm=True` aynı anda DESTEKLENMİYOR (kombinasyon EPM-NSM
    olarak ayrı bir ileri adım — v3.1+).
    
    Parameters
    ----------
    fitness_fn : callable(solution) -> float
        Minimize edilecek fitness fonksiyonu.
        ⚠️ n_jobs > 1 ise pickle-able olmalı (top-level `def`, lambda değil).
    encoding : Encoding
        Solution uzayını ve operatörleri tanımlar.
    method : {'classic', 'fdb', 'rfdb'}
        Rehberlik modu:
            'classic' → turnuva seçimi (FDB yok)
            'fdb'     → deterministik FDB rehber (Kahraman 2020)
            'rfdb'    → rulet-FDB rehber (Aras 2021)
        NSM modunda parent seçiminde, EPM modunda Method-1/Method-2
        ayrımında kullanılır (classic → Method-2, fdb/rfdb → Method-1).
    pop_size : int, default=20
    max_iter : int, default=50
    tournament_k : int, default=3
        Sadece classic mode'da; NSM/EPM kendi mekaniklerini kullanır.
    p_crossover : float, default=0.8
    mutation_rate : float or None
        None ise encoding'in default'u kullanılır (genelde 1/dim).
    seed : int, default=42
    verbose : bool, default=False
    use_nsm : bool, default=False
        NSM survivor selection'ı aktive eder (sigmoid switch ile karışık).
    use_epm : bool, default=False
        EPM popülasyon yönetimini aktive eder (H1+H2+H3).
    nsm_preset : int in 1..10, default=6
        NSM ağırlık preset'i (NSM_PRESETS[nsm_preset]). 6 = Kahraman et al. TLABC
        Case-1 default'u.
    nsm_merkez_sayisi : int or None
        NSM PS bileşenindeki top-K. None → max(2, pop_size // 2).
    n_jobs : int, default=1
        Paralel fitness eval worker sayısı. 1=seri (default), -1=tüm CPU.
        Sadece `classic` mod (use_nsm=False, use_epm=False) için etkili —
        steady-state NSM ve EPM iç döngüsü paralelleşemez.
        ⚠️ fitness_fn pickle-able olmalı (Windows'ta zorunlu).
    early_stopping_patience : int or None
        None → erken durdurma yok (max_iter sonuna kadar koş).
        int  → bu kadar iter boyunca improvement (>tol) yoksa dur.
    early_stopping_tol : float, default=1e-8
        Iyileşme eşiği. fit[t] < fit[t-1] - tol ise iyileşme sayılır.
    checkpoint_every : int or None
        None → checkpoint yok. int → her N iter'de durumu pickle'a kaydet.
        Sadece classic mod ve EPM mod için (NSM steady-state karmaşık).
    checkpoint_path : str or None
        Pickle dosyasının tam yolu. checkpoint_every ile birlikte verilmeli.
    resume_from : str or None
        Daha önce kaydedilmiş checkpoint'ten devam et. None → sıfırdan başla.
        Sadece classic mod için.
    
    Returns
    -------
    dict with keys: best_sol, best_fit, history, n_evaluations, stopped_early
        NSM modunda ek: _nsm_applied, _greedy_applied, _nsm_accepted, _greedy_accepted
    """
    method = _normalize_method(method)
    if use_nsm and use_epm:
        # v3-final-rc2: EPM-NSM kombine mod
        if checkpoint_every is not None or resume_from is not None:
            import warnings
            warnings.warn("Checkpoint EPM-NSM kombinasyonunda v3.1'e bırakıldı. Ignore.")
        return _run_bga_epm_nsm(
            fitness_fn, encoding, method, pop_size, max_iter,
            p_crossover, mutation_rate, seed, verbose, n_jobs,
            nsm_preset, nsm_merkez_sayisi,
            early_stopping_patience, early_stopping_tol,
            log_file, log_every,
        )
    
    if use_nsm:
        if n_jobs != 1:
            import warnings
            warnings.warn("NSM steady-state — n_jobs ignore ediliyor (her child sıralı eval).")
        return _run_bga_nsm(
            fitness_fn, encoding, method, pop_size, max_iter,
            tournament_k, p_crossover, mutation_rate, seed, verbose,
            nsm_preset, nsm_merkez_sayisi,
            early_stopping_patience, early_stopping_tol,
            checkpoint_every, checkpoint_path, resume_from,
            log_file, log_every,
        )
    if use_epm:
        return _run_bga_epm(
            fitness_fn, encoding, method, pop_size, max_iter,
            p_crossover, mutation_rate, seed, verbose, n_jobs,
            early_stopping_patience, early_stopping_tol,
            checkpoint_every, checkpoint_path, resume_from,
            log_file, log_every,
        )
    return _run_bga_classic(
        fitness_fn, encoding, method, pop_size, max_iter,
        tournament_k, p_crossover, mutation_rate, seed, verbose, n_jobs,
        early_stopping_patience, early_stopping_tol,
        checkpoint_every, checkpoint_path, resume_from,
        log_file, log_every,
    )


def _run_bga_classic(fitness_fn, encoding, method, pop_size, max_iter,
                     tournament_k, p_crossover, mutation_rate, seed, verbose,
                     n_jobs=1, early_stopping_patience=None, early_stopping_tol=1e-8,
                     checkpoint_every=None, checkpoint_path=None, resume_from=None,
                     log_file=None, log_every=1):
    """Klasik BGA — generational replacement + elitism. v3-beta davranışı."""
    import pickle
    
    # ─── Resume from checkpoint ───
    if resume_from is not None:
        with open(resume_from, 'rb') as f:
            state = pickle.load(f)
        rng = state['rng']
        pop = state['pop']
        fit = state['fit']
        n_eval = state['n_eval']
        gbest = state['gbest']
        gbest_fit = state['gbest_fit']
        history = state['history']
        start_iter = state['iter'] + 1
        if verbose:
            print(f"  Resumed from {resume_from} @ iter {start_iter} "
                  f"(best_fit={gbest_fit:.6f})")
    else:
        rng = np.random.RandomState(seed)
        pop = encoding.init_population(pop_size, rng)
        fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
        n_eval = pop_size
        bi = int(np.argmin(fit))
        gbest = _copy_solution(pop[bi])
        gbest_fit = float(fit[bi])
        history = [gbest_fit]
        start_iter = 0
    
    stopped_early = False
    logger = _IterLogger(log_file, log_every) if log_file else None
    
    for it in range(start_iter, max_iter):
        scores = fdb_scores(pop, fit, encoding, method=method)
        probs = _selection_probs(scores)
        
        new_pop = [_copy_solution(pop[int(np.argmin(fit))])]  # elitism
        
        while len(new_pop) < pop_size:
            if method == 'classic':
                p1_idx = _tournament(rng, fit, k=tournament_k)
                p2_idx = _tournament(rng, fit, k=tournament_k)
            else:
                p1_idx = int(rng.choice(pop_size, p=probs))
                p2_idx = int(rng.choice(pop_size, p=probs))
                tries = 0
                while p2_idx == p1_idx and tries < 5:
                    p2_idx = int(rng.choice(pop_size, p=probs))
                    tries += 1
            
            p1, p2 = pop[p1_idx], pop[p2_idx]
            
            if rng.rand() < p_crossover:
                c1, c2 = encoding.crossover(rng, p1, p2)
            else:
                c1, c2 = _copy_solution(p1), _copy_solution(p2)
            
            c1 = encoding.mutate(rng, c1, mutation_rate)
            c2 = encoding.mutate(rng, c2, mutation_rate)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        pop = new_pop
        fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
        n_eval += pop_size
        
        cur_best = int(np.argmin(fit))
        if fit[cur_best] < gbest_fit:
            gbest = _copy_solution(pop[cur_best])
            gbest_fit = float(fit[cur_best])
        history.append(gbest_fit)
        if logger:
            logger.log(it + 1, gbest_fit, fit, n_eval)
        
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
        
        # Early stopping kontrolü
        if _check_early_stop(history, early_stopping_patience, early_stopping_tol):
            stopped_early = True
            if verbose:
                print(f"  Early stop @ iter {it+1}: son {early_stopping_patience} "
                      f"iter'de iyileşme < {early_stopping_tol}")
            break
        
        # Checkpointing
        if (checkpoint_every is not None and checkpoint_path is not None
                and (it + 1) % checkpoint_every == 0):
            state = {
                'rng': rng, 'pop': pop, 'fit': fit, 'n_eval': n_eval,
                'gbest': gbest, 'gbest_fit': gbest_fit,
                'history': history, 'iter': it,
                # Meta — kullanıcı doğru parametrelerle resume edebilsin
                '_meta': {'pop_size': pop_size, 'method': method, 'seed': seed,
                          'encoding_repr': repr(encoding)},
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            if verbose:
                print(f"  Checkpoint saved @ iter {it+1} → {checkpoint_path}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
        'stopped_early': stopped_early,
    }


def _run_bga_nsm(fitness_fn, encoding, method, pop_size, max_iter,
                 tournament_k, p_crossover, mutation_rate, seed, verbose,
                 nsm_preset, nsm_merkez_sayisi,
                 early_stopping_patience=None, early_stopping_tol=1e-8,
                 checkpoint_every=None, checkpoint_path=None, resume_from=None,
                 log_file=None, log_every=1):
    """BGA + NSM (B1 fazı): steady-state + sigmoid switch (greedy ↔ NSM).
    
    Standart BGA ile farklar:
        - Generational replacement YOK; tek-tek child üretip steady-state
          replacement yapılır.
        - Sigmoid switch: erken iter ~%0 NSM (greedy hakim), geç iter ~%100 NSM.
        - Elitism: gbest'i hiç replace etme (target başkasına döner).
    """
    import pickle
    
    if nsm_merkez_sayisi is None:
        nsm_merkez_sayisi = max(2, pop_size // 2)
    if nsm_preset not in NSM_PRESETS:
        raise ValueError(f"nsm_preset 1..10 olmalı, verilen: {nsm_preset}")
    weights = NSM_PRESETS[nsm_preset]
    
    if resume_from is not None:
        with open(resume_from, 'rb') as f:
            state = pickle.load(f)
        rng = state['rng']
        pop = state['pop']
        fit = state['fit']
        n_eval = state['n_eval']
        gbest = state['gbest']
        gbest_fit = state['gbest_fit']
        history = state['history']
        start_iter = state['iter'] + 1
        nsm_applied = state.get('nsm_applied', 0)
        greedy_applied = state.get('greedy_applied', 0)
        nsm_accepted = state.get('nsm_accepted', 0)
        greedy_accepted = state.get('greedy_accepted', 0)
        if verbose:
            print(f"  Resumed NSM from {resume_from} @ iter {start_iter}")
    else:
        rng = np.random.RandomState(seed)
        pop = encoding.init_population(pop_size, rng)
        fit = np.array([fitness_fn(s) for s in pop])
        n_eval = pop_size
        bi = int(np.argmin(fit))
        gbest = _copy_solution(pop[bi])
        gbest_fit = float(fit[bi])
        history = [gbest_fit]
        start_iter = 0
        nsm_applied = greedy_applied = nsm_accepted = greedy_accepted = 0
    
    stopped_early = False
    logger = _IterLogger(log_file, log_every) if log_file else None
    
    for it in range(start_iter, max_iter):
        scores = fdb_scores(pop, fit, encoding, method=method)
        probs = _selection_probs(scores)
        switch_prob = sigmoid_switch(it + 1, max_iter)
        
        for _ in range(pop_size):
            # Parent seçimi
            if method == 'classic':
                p1_idx = _tournament(rng, fit, k=tournament_k)
                p2_idx = _tournament(rng, fit, k=tournament_k)
            else:
                p1_idx = int(rng.choice(pop_size, p=probs))
                p2_idx = int(rng.choice(pop_size, p=probs))
                tries = 0
                while p2_idx == p1_idx and tries < 5:
                    p2_idx = int(rng.choice(pop_size, p=probs))
                    tries += 1
            
            p1, p2 = pop[p1_idx], pop[p2_idx]
            
            if rng.rand() < p_crossover:
                c1, _c2 = encoding.crossover(rng, p1, p2)
            else:
                c1 = _copy_solution(p1)
            c1 = encoding.mutate(rng, c1, mutation_rate)
            
            f_child = float(fitness_fn(c1))
            n_eval += 1
            
            # Replace target = parent1 (steady-state); ama gbest'i koru
            target_idx = p1_idx
            elite_idx = int(np.argmin(fit))
            if target_idx == elite_idx:
                non_elite = [j for j in range(pop_size) if j != elite_idx]
                target_idx = int(rng.choice(non_elite))
            x_target = pop[target_idx]
            f_target = float(fit[target_idx])
            
            # Sigmoid switch — bu replacement NSM mi greedy mi?
            if rng.rand() < switch_prob:
                nsm_applied += 1
                accept = nsm_decide(
                    x_target, f_target, c1, f_child,
                    pop, fit, encoding,
                    merkez_sayisi=nsm_merkez_sayisi, weights=weights,
                )
                if accept:
                    nsm_accepted += 1
            else:
                greedy_applied += 1
                accept = (f_child < f_target)
                if accept:
                    greedy_accepted += 1
            
            if accept:
                pop[target_idx] = c1
                fit[target_idx] = f_child
                if f_child < gbest_fit:
                    gbest = _copy_solution(c1)
                    gbest_fit = f_child
        
        history.append(gbest_fit)
        if logger:
            logger.log(it + 1, gbest_fit, fit, n_eval)
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f} "
                  f"(NSM applied={nsm_applied}, accepted={nsm_accepted})")
        
        if _check_early_stop(history, early_stopping_patience, early_stopping_tol):
            stopped_early = True
            if verbose:
                print(f"  Early stop @ iter {it+1}: son {early_stopping_patience} "
                      f"iter'de iyileşme < {early_stopping_tol}")
            break
        
        # Checkpointing
        if (checkpoint_every is not None and checkpoint_path is not None
                and (it + 1) % checkpoint_every == 0):
            state = {
                'rng': rng, 'pop': pop, 'fit': fit, 'n_eval': n_eval,
                'gbest': gbest, 'gbest_fit': gbest_fit,
                'history': history, 'iter': it,
                'nsm_applied': nsm_applied, 'greedy_applied': greedy_applied,
                'nsm_accepted': nsm_accepted, 'greedy_accepted': greedy_accepted,
                '_meta': {'mode': 'nsm', 'pop_size': pop_size, 'method': method, 'seed': seed},
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            if verbose:
                print(f"  NSM checkpoint @ iter {it+1} → {checkpoint_path}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
        'stopped_early': stopped_early,
        '_nsm_applied': nsm_applied,
        '_greedy_applied': greedy_applied,
        '_nsm_accepted': nsm_accepted,
        '_greedy_accepted': greedy_accepted,
    }


def _run_bga_epm(fitness_fn, encoding, method, pop_size, max_iter,
                 p_crossover, mutation_rate, seed, verbose, n_jobs=1,
                 early_stopping_patience=None, early_stopping_tol=1e-8,
                 checkpoint_every=None, checkpoint_path=None, resume_from=None,
                 log_file=None, log_every=1):
    """BGA + EPM (B2 fazı): H1 epoch + H2 popülasyon update + H3 mate selection.
    
    method:
        'classic'      → EPM Method-2 (tek-best FDB referansı)
        'fdb' / 'rfdb' → EPM Method-1 (her parent için ayrı FDB)
    
    n_jobs > 1: H1 epoch sonunda P_child fitness'ları paralel hesaplanır.
    """
    import pickle
    
    # EPM forbidden-list mekaniği çift pop_size gerektirir (n/2 fit-half + n/2 fdb-half)
    if pop_size % 2 != 0:
        import warnings
        warnings.warn(f"EPM modu çift pop_size gerektirir; {pop_size} → {pop_size + 1} ayarlandı.")
        pop_size = pop_size + 1
    n = pop_size
    
    if resume_from is not None:
        with open(resume_from, 'rb') as f:
            state = pickle.load(f)
        rng = state['rng']
        P = state['P']
        F = state['F']
        n_eval = state['n_eval']
        gbest = state['gbest']
        gbest_fit = state['gbest_fit']
        history = state['history']
        start_epoch = state['epoch'] + 1
        if verbose:
            print(f"  Resumed EPM from {resume_from} @ epoch {start_epoch}")
    else:
        rng = np.random.RandomState(seed)
        P = encoding.init_population(n, rng)
        F = _eval_pop(fitness_fn, P, n_jobs=n_jobs)
        n_eval = n
        bi = int(np.argmin(F))
        gbest = _copy_solution(P[bi])
        gbest_fit = float(F[bi])
        history = [gbest_fit]
        start_epoch = 0
        # İlk epoch için fitness sırasıyla yarıla
        sort_idx = np.argsort(F)
        P = [_copy_solution(P[k]) for k in sort_idx]
        F = F[sort_idx].copy()
    
    stopped_early = False
    logger = _IterLogger(log_file, log_every) if log_file else None
    
    for epoch in range(start_epoch, max_iter):
        # ─── H1: epoch içinde child'lar üret, update YOK ───
        # Önce tüm child'ları sıralı oluştur (rng deterministik)
        P_child = []
        for i in range(n):
            mate_idx, _secondary = epm_guide_select(i, n, rng=rng)
            p1, p2 = P[i], P[mate_idx]
            
            if rng.rand() < p_crossover:
                c1, _c2 = encoding.crossover(rng, p1, p2)
            else:
                c1 = _copy_solution(p1)
            c1 = encoding.mutate(rng, c1, mutation_rate)
            P_child.append(c1)
        
        # Sonra fitness'ları toplu (paralel) hesapla — H1 izin veriyor
        F_child = _eval_pop(fitness_fn, P_child, n_jobs=n_jobs)
        n_eval += n
        
        # U-matrix bazlı gbest update
        all_fits = np.concatenate([F, F_child])
        all_pop = list(P) + P_child
        u_best = int(np.argmin(all_fits))
        if all_fits[u_best] < gbest_fit:
            gbest = _copy_solution(all_pop[u_best])
            gbest_fit = float(all_fits[u_best])
        
        # H2: yeni P (n/2 fitness + n/2 FDB seçim)
        new_P, new_F, _fit_h, _fdb_h = epm_population_update(
            P, F, P_child, F_child, encoding, method=method,
        )
        P = new_P
        F = np.asarray(new_F, dtype=float)
        
        history.append(gbest_fit)
        if logger:
            logger.log(epoch + 1, gbest_fit, F, n_eval)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
        
        if _check_early_stop(history, early_stopping_patience, early_stopping_tol):
            stopped_early = True
            if verbose:
                print(f"  Early stop @ epoch {epoch+1}: son {early_stopping_patience} "
                      f"epoch'ta iyileşme < {early_stopping_tol}")
            break
        
        # Checkpointing
        if (checkpoint_every is not None and checkpoint_path is not None
                and (epoch + 1) % checkpoint_every == 0):
            state = {
                'rng': rng, 'P': P, 'F': F, 'n_eval': n_eval,
                'gbest': gbest, 'gbest_fit': gbest_fit,
                'history': history, 'epoch': epoch,
                '_meta': {'mode': 'epm', 'pop_size': pop_size, 'method': method, 'seed': seed},
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state, f)
            if verbose:
                print(f"  EPM checkpoint @ epoch {epoch+1} → {checkpoint_path}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
        'stopped_early': stopped_early,
    }


def _run_bga_epm_nsm(fitness_fn, encoding, method, pop_size, max_iter,
                     p_crossover, mutation_rate, seed, verbose, n_jobs,
                     nsm_preset, nsm_merkez_sayisi,
                     early_stopping_patience=None, early_stopping_tol=1e-8,
                     log_file=None, log_every=1):
    """BGA + EPM + NSM (B3 hibrit fazı): EPM tabanı + fit-half NSM süzgeci.
    
    Yapı (B2 ile aynı, sadece H2 farklı):
        H1 (epoch): epoch içinde update yok (EPM ile birebir)
        H2 (population update): epm_nsm_population_update kullanılır
            - fit-half: sigmoid_switch ile NSM süzgeçli (geç epoch'ta NSM hakim)
            - fdb-half: klasik EPM Method-1/2 (NSM yok)
        H3 (mate selection): EPM ile birebir aynı (fit-half ↔ fdb-half eşleşme)
    
    Sigmoid switch: erken epoch ~0 (klasik EPM), geç epoch ~1 (NSM süzgeçli).
    """
    rng = np.random.RandomState(seed)
    if pop_size % 2 != 0:
        import warnings
        warnings.warn(f"EPM-NSM çift pop_size gerektirir; {pop_size} → {pop_size + 1} ayarlandı.")
        pop_size = pop_size + 1
    n = pop_size
    
    if nsm_merkez_sayisi is None:
        nsm_merkez_sayisi = max(2, pop_size // 2)
    if nsm_preset not in NSM_PRESETS:
        raise ValueError(f"nsm_preset 1..10 olmalı, verilen: {nsm_preset}")
    nsm_weights = NSM_PRESETS[nsm_preset]
    
    P = encoding.init_population(n, rng)
    F = _eval_pop(fitness_fn, P, n_jobs=n_jobs)
    n_eval = n
    
    bi = int(np.argmin(F))
    gbest = _copy_solution(P[bi])
    gbest_fit = float(F[bi])
    history = [gbest_fit]
    stopped_early = False
    
    sort_idx = np.argsort(F)
    P = [_copy_solution(P[k]) for k in sort_idx]
    F = F[sort_idx].copy()
    
    nsm_filtered = 0  # debug: kaç fit-half kararına NSM uygulandı
    logger = _IterLogger(log_file, log_every) if log_file else None
    
    for epoch in range(max_iter):
        # H1: epoch içinde child üret, update yok
        P_child = []
        for i in range(n):
            mate_idx, _ = epm_guide_select(i, n, rng=rng)
            p1, p2 = P[i], P[mate_idx]
            if rng.rand() < p_crossover:
                c1, _c2 = encoding.crossover(rng, p1, p2)
            else:
                c1 = _copy_solution(p1)
            c1 = encoding.mutate(rng, c1, mutation_rate)
            P_child.append(c1)
        F_child = _eval_pop(fitness_fn, P_child, n_jobs=n_jobs)
        n_eval += n
        
        # gbest update
        all_fits = np.concatenate([F, F_child])
        all_pop = list(P) + P_child
        u_best = int(np.argmin(all_fits))
        if all_fits[u_best] < gbest_fit:
            gbest = _copy_solution(all_pop[u_best])
            gbest_fit = float(all_fits[u_best])
        
        # H2: NSM süzgeçli population update
        switch_prob = sigmoid_switch(epoch + 1, max_iter)
        new_P, new_F, _fit_h, _fdb_h = epm_nsm_population_update(
            P, F, P_child, F_child, encoding,
            method=method, switch_prob=switch_prob,
            nsm_weights=nsm_weights, nsm_merkez_sayisi=nsm_merkez_sayisi,
            rng=rng,
        )
        if switch_prob > 0:
            nsm_filtered += 1
        P = new_P
        F = np.asarray(new_F, dtype=float)
        
        history.append(gbest_fit)
        if logger:
            logger.log(epoch + 1, gbest_fit, F, n_eval)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{max_iter}: best_fit = {gbest_fit:.6f} "
                  f"(switch_prob={switch_prob:.2f})")
        
        if _check_early_stop(history, early_stopping_patience, early_stopping_tol):
            stopped_early = True
            if verbose:
                print(f"  Early stop @ epoch {epoch+1}")
            break
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
        'stopped_early': stopped_early,
        '_nsm_filtered_epochs': nsm_filtered,
    }


def run_bde(fitness_fn: Callable, encoding: Encoding,
            method: str = 'classic',
            pop_size: int = 20, max_iter: int = 50,
            F: float = 0.5, CR: float = 0.9,
            noncont_mutation_rate: Optional[float] = None,
            seed: int = 42, verbose: bool = False,
            n_jobs: int = 1):
    """Universal Differential Evolution.
    
    Not: DE doğal olarak continuous için tasarlandı. Binary/Categorical 
    encoding'ler için crossover/mutation encoding'in kendisinden gelir 
    (yani bu DE değil, bir GA varyantına dönüşür). Saf DE davranışı için
    ContinuousEncoding kullan.
    
    ── UYARLAMA BEYANI (v0.3.5, 25 Eyl 2026) ─────────────────────────────
    Continuous dışı encoding'lerde (Binary/Categorical/Mixed/Permutation)
    trial = crossover(pop[r1], pop[i]) → mutate(rate=noncont_mutation_rate).
    v0.3.4 ve öncesinde mutasyon oranı olarak DE ölçek katsayısı `F` (0.5)
    geçiriliyordu; BinaryEncoding.mutate bunu BİT-ÇEVİRME OLASILIĞI olarak
    yorumladığından her adayın bitlerinin ~%50'si çevriliyor ve algoritma
    rastgele aramaya denk davranıyordu (OneMax n=60, 1000 FE: F=0.5 → hata
    14–20, rastgele arama 18–20, F=1/n → 0; MEALPY-DE 11.5). v0.3.5:
    `noncont_mutation_rate=None` → encoding varsayılanı (Binary: 1/n).
    `F` ve `CR` yalnız ContinuousEncoding'de kullanılır. RNG akışı
    continuous'ta değişmedi; binary'de DEĞİŞTİ (v0.3.4 binary BDE
    sonuçlarıyla aynı seed farklı sonuç verir).
    ────────────────────────────────────────────────────────────────────────
    """
    method = _normalize_method(method)
    rng = np.random.RandomState(seed)
    pop = encoding.init_population(pop_size, rng)
    fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
    n_eval = pop_size
    
    bi = int(np.argmin(fit))
    gbest = _copy_solution(pop[bi])
    gbest_fit = float(fit[bi])
    history = [gbest_fit]
    
    is_continuous = isinstance(encoding, ContinuousEncoding)
    
    for it in range(max_iter):
        scores = fdb_scores(pop, fit, encoding, method=method)
        probs = _selection_probs(scores)
        
        # Önce tüm trial'ları üret (rng deterministik), sonra paralel eval
        trials = []
        for i in range(pop_size):
            avail = [j for j in range(pop_size) if j != i]
            sub_p = np.delete(probs, i)
            sub_p_sum = sub_p.sum()
            if sub_p_sum < 1e-10 or np.count_nonzero(sub_p) < 3:
                r1, r2, r3 = rng.choice(avail, 3, replace=False)
            else:
                sub_p = sub_p / sub_p_sum
                r1, r2, r3 = rng.choice(avail, 3, replace=False, p=sub_p)
            
            if is_continuous:
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, encoding.bounds[:, 0], encoding.bounds[:, 1])
                trial = pop[i].copy()
                mask = rng.rand(encoding.dim) < CR
                mask[rng.randint(0, encoding.dim)] = True
                trial[mask] = mutant[mask]
            else:
                c1, _ = encoding.crossover(rng, pop[r1], pop[i])
                # v0.3.5: mutasyon oranı F DEĞİL (bkz. UYARLAMA BEYANI)
                trial = encoding.mutate(rng, c1, noncont_mutation_rate)
            trials.append(trial)
        
        # Paralel batch eval
        trial_fits = _eval_pop(fitness_fn, trials, n_jobs=n_jobs)
        n_eval += pop_size
        
        # Greedy selection
        new_pop = []
        new_fit = np.zeros(pop_size)
        for i in range(pop_size):
            if trial_fits[i] < fit[i]:
                new_pop.append(trials[i])
                new_fit[i] = trial_fits[i]
            else:
                new_pop.append(pop[i])
                new_fit[i] = fit[i]
        
        pop = new_pop
        fit = new_fit
        
        cur_best = int(np.argmin(fit))
        if fit[cur_best] < gbest_fit:
            gbest = _copy_solution(pop[cur_best])
            gbest_fit = float(fit[cur_best])
        history.append(gbest_fit)
        
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
    }


def run_bpso(fitness_fn: Callable, encoding: Encoding,
             method: str = 'classic',
             pop_size: int = 20, max_iter: int = 50,
             w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
             v_max: float = 4.0,
             seed: int = 42, verbose: bool = False,
             n_jobs: int = 1):
    """Universal Particle Swarm Optimization.
    
    BinaryEncoding ile sigmoid transfer fonksiyonu kullanılır (Kennedy & 
    Eberhart 1997). ContinuousEncoding ile klasik PSO. Categorical/Mixed 
    için PSO doğal değil; bu durumlarda BGA önerilir.
    """
    method = _normalize_method(method)
    rng = np.random.RandomState(seed)
    pop = encoding.init_population(pop_size, rng)
    fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
    n_eval = pop_size
    
    is_binary = isinstance(encoding, BinaryEncoding)
    is_continuous = isinstance(encoding, ContinuousEncoding)
    
    if not (is_binary or is_continuous):
        raise NotImplementedError(
            "PSO şu an sadece BinaryEncoding ve ContinuousEncoding ile "
            "çalışıyor. Categorical/Mixed için run_bga kullan."
        )
    
    # Hız vektörleri
    if is_binary:
        velocities = [rng.uniform(-v_max, v_max, encoding.dim) for _ in range(pop_size)]
    else:
        v_range = encoding.bounds[:, 1] - encoding.bounds[:, 0]
        velocities = [rng.uniform(-0.1, 0.1, encoding.dim) * v_range for _ in range(pop_size)]
    
    pbest = [_copy_solution(s) for s in pop]
    pbest_fit = fit.copy()
    
    bi = int(np.argmin(fit))
    gbest = _copy_solution(pop[bi])
    gbest_fit = float(fit[bi])
    history = [gbest_fit]
    
    for it in range(max_iter):
        # Rehber seçimi (v3.1: 'fdb' ve 'rfdb' artık farklı davranır —
        # v3.0'da ikisi de argmax'a düşüyordu)
        if method == 'classic':
            guide = gbest
        else:
            scores = fdb_scores(pop, fit, encoding, method=method)
            if method == 'fdb':
                # Deterministik: one-hot skorun argmax'ı = FDB-best
                guide_idx = int(np.argmax(scores))
            else:  # 'rfdb'
                # Olasılıksal: FDB skorları rulet ağırlığı
                guide_idx = int(rng.choice(pop_size, p=_selection_probs(scores)))
            guide = pop[guide_idx]
        
        for i in range(pop_size):
            r1, r2 = rng.rand(encoding.dim), rng.rand(encoding.dim)
            
            if is_binary:
                # Velocity güncelle
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest[i] - pop[i])
                                 + c2 * r2 * (guide - pop[i]))
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                # Sigmoid transfer
                prob = 1.0 / (1.0 + np.exp(-np.clip(velocities[i], -50, 50)))
                pop[i] = (rng.rand(encoding.dim) < prob).astype(int)
                if encoding.ensure_one and pop[i].sum() == 0:
                    pop[i][rng.randint(0, encoding.dim)] = 1
            else:
                # Continuous PSO
                velocities[i] = (w * velocities[i]
                                 + c1 * r1 * (pbest[i] - pop[i])
                                 + c2 * r2 * (guide - pop[i]))
                pop[i] = pop[i] + velocities[i]
                pop[i] = np.clip(pop[i], encoding.bounds[:, 0], encoding.bounds[:, 1])
            
            # Fitness güncelle
            fit[i] = fitness_fn(pop[i])
            n_eval += 1
            if fit[i] < pbest_fit[i]:
                pbest[i] = _copy_solution(pop[i])
                pbest_fit[i] = fit[i]
                if fit[i] < gbest_fit:
                    gbest = _copy_solution(pop[i])
                    gbest_fit = float(fit[i])
        
        history.append(gbest_fit)
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
    }


# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI: Solution kopyalama (numpy array, dict, vb. için)
# ═══════════════════════════════════════════════════════════════════════════


def _copy_solution(sol):
    """Solution tipine göre uygun kopyalama yap."""
    if isinstance(sol, dict):
        return {k: _copy_solution(v) for k, v in sol.items()}
    elif hasattr(sol, 'copy'):
        return sol.copy()
    else:
        import copy
        return copy.copy(sol)


# ═══════════════════════════════════════════════════════════════════════════
# YARDIMCI: Parallel fitness evaluation
# ═══════════════════════════════════════════════════════════════════════════
# 
# n_jobs=1 (default) → seri eval, mevcut davranış
# n_jobs>1           → multiprocessing.Pool ile paralel
# n_jobs=-1          → tüm CPU'lar
#
# Önemli: fitness_fn pickle-able olmalı (lambda işe yaramaz, top-level def
# kullan). __main__ scope'unda tanımlı fonksiyonlar Windows'ta çalışmayabilir.
# ═══════════════════════════════════════════════════════════════════════════


def _eval_pop(fitness_fn, population, n_jobs=1):
    """Popülasyonun fitness'ını seri veya paralel hesapla.
    
    Parameters
    ----------
    fitness_fn : callable(solution) -> float
        Pickle-able olmalı (top-level def, lambda DEĞİL — paralel modda).
    population : list
        Bireyler listesi.
    n_jobs : int, default=1
        1  → seri (mevcut davranış)
        >1 → multiprocessing.Pool worker sayısı
        -1 → os.cpu_count()
    
    Returns
    -------
    np.ndarray of shape (len(population),) — fitness değerleri.
    
    Notes
    -----
    Paralel modda her birey bağımsız değerlendirilir; karşılıklı bağımlılık
    olan algoritmalar (NSM steady-state) buradan paralelleşemez. Sadece batch
    evaluation noktalarında (init, generational replace) kullanılır.
    """
    if n_jobs == 1:
        return np.array([fitness_fn(s) for s in population])
    
    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1
    
    if n_jobs <= 1 or len(population) <= 1:
        return np.array([fitness_fn(s) for s in population])
    
    # multiprocessing.Pool
    from multiprocessing import Pool
    with Pool(processes=n_jobs) as pool:
        results = pool.map(fitness_fn, population)
    return np.array(results, dtype=float)


def _check_early_stop(history, patience, tol):
    """Son `patience` iter boyunca tol'dan büyük iyileşme yoksa True döner.
    
    History en az patience+1 uzun olmalı. Aksi halde False (henüz erken).
    """
    if patience is None or patience <= 0:
        return False
    if len(history) <= patience:
        return False
    # Son patience+1 nokta arasında en iyi - en kötü farkı tol'dan az mı?
    recent = history[-(patience + 1):]
    improvement = recent[0] - recent[-1]
    return improvement < tol


class _IterLogger:
    """Optimizasyon iterasyon loglayıcı (CSV).
    
    Header: iter,best_fit,mean_fit,std_fit,n_evaluations,elapsed_seconds
    """
    def __init__(self, log_file, log_every=1):
        self.log_file = log_file
        self.log_every = log_every
        self.t0 = None
        if log_file:
            import time
            self.t0 = time.time()
            with open(log_file, 'w') as f:
                f.write("iter,best_fit,mean_fit,std_fit,n_evaluations,elapsed_seconds\n")
    
    def log(self, iter_num, best_fit, fit_array, n_evaluations):
        if not self.log_file or (iter_num % self.log_every != 0):
            return
        import time
        elapsed = time.time() - self.t0
        mean_f = float(np.mean(fit_array))
        std_f = float(np.std(fit_array))
        with open(self.log_file, 'a') as f:
            f.write(f"{iter_num},{best_fit:.10f},{mean_f:.10f},{std_f:.10f},"
                    f"{n_evaluations},{elapsed:.4f}\n")




# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 4: EK ALGORİTMALAR (v3-beta)
# ═══════════════════════════════════════════════════════════════════════════


def run_babc(fitness_fn, encoding,
             method='classic',
             pop_size=20, max_iter=50,
             limit=None,
             seed=42, verbose=False,
             n_jobs=1):
    """Universal Artificial Bee Colony — encoding-agnostic.
    
    ABC'nin 3 fazı:
      1. Employed bees: her arı kendi yiyecek kaynağına komşu arar
      2. Onlooker bees: fitness'a göre rulet seçimi ile kaynaklara gider
      3. Scout bees: limit kez gelişmeyen kaynaklar terkedilir, yenileri keşfedilir
    
    Parameters
    ----------
    limit : int or None
        Bir kaynak limit kez gelişmezse terk edilir
        (default: max(5, pop_size * dim // 4))
    """
    method = _normalize_method(method)
    rng = np.random.RandomState(seed)
    pop = encoding.init_population(pop_size, rng)
    fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
    n_eval = pop_size
    
    if limit is None:
        limit = max(5, pop_size * encoding.dim // 4)
    
    trial_counters = np.zeros(pop_size, dtype=int)
    
    bi = int(np.argmin(fit))
    gbest = _copy_solution(pop[bi])
    gbest_fit = float(fit[bi])
    history = [gbest_fit]
    
    for it in range(max_iter):
        scores = fdb_scores(pop, fit, encoding, method=method)
        probs = _selection_probs(scores)
        
        # ─── Faz 1: Employed bees ───
        for i in range(pop_size):
            # Komşu seç (FDB-rehberli veya rastgele)
            if method == 'classic':
                k = rng.randint(0, pop_size)
                while k == i:
                    k = rng.randint(0, pop_size)
            else:
                avail = [j for j in range(pop_size) if j != i]
                sub_p = np.delete(probs, i)
                sub_p_sum = sub_p.sum()
                if sub_p_sum < 1e-10:
                    k = rng.choice(avail)
                else:
                    sub_p = sub_p / sub_p_sum
                    k = rng.choice(avail, p=sub_p)
            
            # Komşu çözüm üret: encoding'e göre uygun strateji
            new_sol = _abc_neighbor(rng, pop[i], pop[k], encoding)
            new_fit = fitness_fn(new_sol)
            n_eval += 1
            
            if new_fit < fit[i]:
                pop[i] = new_sol
                fit[i] = new_fit
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        # ─── Faz 2: Onlooker bees ───
        # Fitness'tan olasılık üret (min fitness istediğimiz için ters)
        f_max = float(fit.max())
        f_min = float(fit.min())
        if f_max == f_min:
            sel_probs = np.ones(pop_size) / pop_size
        else:
            inv_fit = (f_max + 1e-10) - fit
            sel_probs = inv_fit / inv_fit.sum()
        
        # FDB ile karıştır (eğer method != classic)
        if method != 'classic':
            sel_probs = 0.5 * sel_probs + 0.5 * probs
            sel_probs = sel_probs / sel_probs.sum()
        
        for _ in range(pop_size):
            i = int(rng.choice(pop_size, p=sel_probs))
            # Yine komşu seç
            k = rng.randint(0, pop_size)
            while k == i:
                k = rng.randint(0, pop_size)
            new_sol = _abc_neighbor(rng, pop[i], pop[k], encoding)
            new_fit = fitness_fn(new_sol)
            n_eval += 1
            if new_fit < fit[i]:
                pop[i] = new_sol
                fit[i] = new_fit
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
        
        # ─── Faz 3: Scout bees ───
        for i in range(pop_size):
            if trial_counters[i] >= limit:
                pop[i] = encoding.init_individual(rng)
                fit[i] = fitness_fn(pop[i])
                n_eval += 1
                trial_counters[i] = 0
        
        # Global best güncelle
        cur_best = int(np.argmin(fit))
        if fit[cur_best] < gbest_fit:
            gbest = _copy_solution(pop[cur_best])
            gbest_fit = float(fit[cur_best])
        history.append(gbest_fit)
        
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
    }


def _abc_neighbor(rng, current, partner, encoding):
    """ABC komşu çözüm üretici — encoding'e göre uygun strateji."""
    if isinstance(encoding, ContinuousEncoding):
        # Klasik ABC: x_new = x_i + φ * (x_i - x_k), φ ∈ [-1, 1]
        phi = rng.uniform(-1, 1, encoding.dim)
        new_sol = current + phi * (current - partner)
        new_sol = np.clip(new_sol, encoding.bounds[:, 0], encoding.bounds[:, 1])
        return new_sol
    elif isinstance(encoding, BinaryEncoding):
        # Binary ABC: rastgele bit flip + partner'dan bit alma
        new_sol = current.copy()
        # Birkaç bit flip yap (rastgele 1-3 bit)
        n_flip = rng.randint(1, max(2, encoding.dim // 5))
        flip_idx = rng.choice(encoding.dim, n_flip, replace=False)
        new_sol[flip_idx] ^= 1
        if encoding.ensure_one and new_sol.sum() == 0:
            new_sol[rng.randint(0, encoding.dim)] = 1
        return new_sol
    else:
        # Generic: encoding'in kendi crossover+mutation'ını kullan
        c1, _ = encoding.crossover(rng, current, partner)
        return encoding.mutate(rng, c1)


def run_bgwo(fitness_fn, encoding,
             method='classic',
             pop_size=20, max_iter=50,
             seed=42, verbose=False,
             n_jobs=1):
    """Universal Grey Wolf Optimizer (adım-bazlı V-shape + greedy seçim).
    
    Binary uzayda V-shape transfer fonksiyonu (|tanh|, bkz. Mirjalili & Lewis
    2013; Mafarja & Mirjalili 2018) ADIM büyüklüğüne uygulanır ve child
    birey-bazlı greedy ile kabul edilir (v3.1 düzeltmesi). Continuous için
    klasik GWO (Mirjalili et al. 2014).
    
    GWO'nun 3 lider yapısı (alpha, beta, delta) FDB ile birleşir:
    method='classic' → 3 lider klasik GWO (en iyi 3 fitness)
    method='fdb'     → deterministik: en yüksek FDB skorlu birey alpha
                       (one-hot skorla top-3 sıralaması alpha'yı FDB-best yapar)
    method='rfdb'    → 3 lider ham FDB skor sıralamasından (çeşitli liderler)
    """
    method = _normalize_method(method)
    rng = np.random.RandomState(seed)
    pop = encoding.init_population(pop_size, rng)
    fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
    n_eval = pop_size
    
    is_continuous = isinstance(encoding, ContinuousEncoding)
    is_binary = isinstance(encoding, BinaryEncoding)
    
    if not (is_continuous or is_binary):
        raise NotImplementedError(
            "GWO şu an sadece BinaryEncoding ve ContinuousEncoding ile "
            "çalışıyor. Diğer encoding'ler için run_bga kullan."
        )
    
    bi = int(np.argmin(fit))
    gbest = _copy_solution(pop[bi])
    gbest_fit = float(fit[bi])
    history = [gbest_fit]
    
    for it in range(max_iter):
        # Lider seçimi
        if method == 'classic':
            # En iyi 3 birey alpha, beta, delta
            sorted_idx = np.argsort(fit)
            alpha = pop[sorted_idx[0]]
            beta = pop[sorted_idx[1]]
            delta = pop[sorted_idx[2]] if pop_size >= 3 else pop[sorted_idx[1]]
        else:
            # FDB skorlu lider seç
            scores = fdb_scores(pop, fit, encoding, method=method)
            sorted_score_idx = np.argsort(-scores)
            alpha = pop[sorted_score_idx[0]]
            beta = pop[sorted_score_idx[1]] if pop_size > 1 else alpha
            delta = pop[sorted_score_idx[2]] if pop_size > 2 else beta
        
        # a parametresi: 2'den 0'a doğru lineer azalır (klasik GWO)
        a = 2 - 2 * it / max_iter
        
        new_pop = []
        for i in range(pop_size):
            if is_continuous:
                # Continuous GWO: standart formül
                A1 = 2 * a * rng.rand(encoding.dim) - a
                A2 = 2 * a * rng.rand(encoding.dim) - a
                A3 = 2 * a * rng.rand(encoding.dim) - a
                C1 = 2 * rng.rand(encoding.dim)
                C2 = 2 * rng.rand(encoding.dim)
                C3 = 2 * rng.rand(encoding.dim)
                
                X1 = alpha - A1 * np.abs(C1 * alpha - pop[i])
                X2 = beta - A2 * np.abs(C2 * beta - pop[i])
                X3 = delta - A3 * np.abs(C3 * delta - pop[i])
                
                new_x = (X1 + X2 + X3) / 3
                new_x = np.clip(new_x, encoding.bounds[:, 0], encoding.bounds[:, 1])
                new_pop.append(new_x)
            else:
                # Binary GWO — v3.1 düzeltmesi:
                # V-shape transfer POZİSYONA değil ADIMA uygulanır.
                # v3.0 hatası: prob = |tanh(avg_pozisyon)| + flip semantiği,
                # liderlerin '1' dediği bitleri yakınsamada bile ~%76 olasılıkla
                # flip'liyordu → algoritma yakınsayamıyordu (OneMax'ta ampirik
                # olarak doğrulandı). Literatür semantiği (Mirjalili & Lewis
                # 2013; Mafarja & Mirjalili 2018): transfer fonksiyonu DEĞİŞİM
                # büyüklüğüne uygulanır; yakınsamada adım→0 → flip olasılığı→0.
                A1 = 2 * a * rng.rand(encoding.dim) - a
                A2 = 2 * a * rng.rand(encoding.dim) - a
                A3 = 2 * a * rng.rand(encoding.dim) - a
                C1 = 2 * rng.rand(encoding.dim)
                C2 = 2 * rng.rand(encoding.dim)
                C3 = 2 * rng.rand(encoding.dim)
                
                X1_cont = alpha.astype(float) - A1 * np.abs(C1 * alpha.astype(float) - pop[i].astype(float))
                X2_cont = beta.astype(float) - A2 * np.abs(C2 * beta.astype(float) - pop[i].astype(float))
                X3_cont = delta.astype(float) - A3 * np.abs(C3 * delta.astype(float) - pop[i].astype(float))
                
                avg_cont = (X1_cont + X2_cont + X3_cont) / 3
                step = avg_cont - pop[i].astype(float)  # önerilen konum − mevcut konum
                # V-shape transfer: T(Δx) = |tanh(Δx)| → flip olasılığı
                prob = np.abs(np.tanh(step))
                flip_mask = rng.rand(encoding.dim) < prob
                new_x = pop[i].copy()
                new_x[flip_mask] ^= 1
                if encoding.ensure_one and new_x.sum() == 0:
                    new_x[rng.randint(0, encoding.dim)] = 1
                new_pop.append(new_x)
        
        new_fit = np.array([fitness_fn(s) for s in new_pop])
        n_eval += pop_size
        
        if is_binary:
            # v3.1: birey-bazlı greedy seçim — child ancak daha iyiyse geçer.
            # (v3.0'ın tüm-popülasyon değişimi + tek-birey elitizmi binary'de
            # yetersizdi; greedy ile OneMax performansı ~5 kat iyileşti.)
            for i in range(pop_size):
                if new_fit[i] >= fit[i]:
                    new_pop[i] = pop[i]
                    new_fit[i] = fit[i]
        else:
            # Continuous: kanonik GWO koşulsuz hareket eder; yalnız
            # elitizm koruması uygulanır (önceki en iyi kaybolmasın).
            worst_new = int(np.argmax(new_fit))
            best_old = int(np.argmin(fit))
            if fit[best_old] < new_fit[worst_new]:
                new_pop[worst_new] = _copy_solution(pop[best_old])
                new_fit[worst_new] = fit[best_old]
        
        pop = new_pop
        fit = new_fit
        
        cur_best = int(np.argmin(fit))
        if fit[cur_best] < gbest_fit:
            gbest = _copy_solution(pop[cur_best])
            gbest_fit = float(fit[cur_best])
        history.append(gbest_fit)
        
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
    }


# ─── Lévy Flight Yardımcıları (BCS için) ──────────────────────────────────
def _gamma_function(z):
    """math.gamma alternatifi (numpy 1.x'te kararlı)."""
    import math
    return math.gamma(z)


def _levy_flight_step(rng, beta=1.5):
    """Mantegna algoritması ile bir Lévy adımı (skaler).
    
    Heavy-tailed dağılımdan örnek: küçük adımlar sık, büyük adımlar nadir
    ama olası. Yerel araştırma + ara sıra büyük sıçrama.
    """
    sigma_u = (
        _gamma_function(1 + beta) * np.sin(np.pi * beta / 2)
        / (_gamma_function((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma_u)
    v = rng.normal(0, 1)
    return u / (abs(v) ** (1 / beta) + 1e-12)


def _levy_perturb(rng, x, encoding, scale=1.0):
    """Encoding-aware Lévy perturbation.
    
    BinaryEncoding   : Lévy step'in büyüklüğüne göre N bit flip
    ContinuousEncoding: x + step (her boyutta ayrı), bound'lara clip
    Categorical/Mixed: |step| → kaç boyut değişeceğini belirler, sonra mutate
    """
    step = _levy_flight_step(rng)
    
    if isinstance(encoding, BinaryEncoding):
        n_flip = int(np.clip(round(abs(step) * scale * 2.0), 1, encoding.n_features))
        new_x = x.copy()
        flip_idx = rng.choice(encoding.n_features, n_flip, replace=False)
        new_x[flip_idx] ^= 1
        if encoding.ensure_one and new_x.sum() == 0:
            new_x[rng.randint(0, encoding.n_features)] = 1
        return new_x
    
    if isinstance(encoding, ContinuousEncoding):
        # Her boyut için bağımsız Lévy step (ranges'a göre normalize)
        new_x = x.copy()
        for d in range(encoding.n_dim):
            lo, hi = encoding.bounds[d]
            r = (hi - lo) * 0.05 * scale  # %5'lik range birim
            new_x[d] = x[d] + _levy_flight_step(rng) * r
        return np.clip(new_x, encoding.bounds[:, 0], encoding.bounds[:, 1])
    
    if isinstance(encoding, CategoricalEncoding):
        # Lévy büyüklüğüne göre random reset miktarı
        n_change = int(np.clip(round(abs(step) * scale), 1, encoding.n_dim))
        rate = n_change / encoding.n_dim
        return encoding.mutate(rng, x, rate=rate)
    
    if isinstance(encoding, MixedEncoding):
        # Her bileşene ayrı Lévy uygula
        return {
            name: _levy_perturb(rng, x[name], enc, scale)
            for name, enc in encoding.components.items()
        }
    
    # Fallback: encoding'in mutate'ı
    return encoding.mutate(rng, x, rate=0.3)


def run_bcs(fitness_fn: Callable, encoding: Encoding,
            method: str = 'classic',
            pop_size: int = 20, max_iter: int = 50,
            pa: float = 0.25,
            levy_scale: float = 1.0,
            seed: int = 42, verbose: bool = False,
            n_jobs: int = 1):
    """Universal Cuckoo Search — encoding-aware Lévy flight.
    
    Standart Cuckoo Search'ün (Yang & Deb 2009) iki fazı:
        Phase 1 (Lévy flight): Her cuckoo bir Lévy adımı atar, rastgele
                               bir yuvayla rekabet eder. Heavy-tailed dağılım
                               yerel araştırmayla ara sıra büyük sıçramayı
                               birleştirir.
        Phase 2 (Nest abandonment): pa olasılığıyla yuvalar terkedilir.
            method='classic' → uniform random reset
            method='fdb'     → deterministik FDB rehberin komşusu
            method='rfdb'    → rulet-FDB ile seçilen rehberin komşusu
    
    Encoding-aware: `_levy_perturb` her tipte uygun adımı üretir.
    
    Parameters
    ----------
    fitness_fn : callable
        Minimize edilecek fitness.
    encoding : Encoding
        Solution uzayı.
    method : {'classic', 'fdb', 'rfdb'}
    pop_size : int, default=20
    max_iter : int, default=50
    pa : float, default=0.25
        Nest abandonment olasılığı.
    levy_scale : float, default=1.0
        Lévy adımı ölçeği.
    seed : int, default=42
    verbose : bool, default=False
    
    Returns
    -------
    dict with keys: best_sol, best_fit, history, n_evaluations
    """
    method = _normalize_method(method)
    rng = np.random.RandomState(seed)
    
    pop = encoding.init_population(pop_size, rng)
    fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
    n_eval = pop_size
    
    bi = int(np.argmin(fit))
    gbest = _copy_solution(pop[bi])
    gbest_fit = float(fit[bi])
    history = [gbest_fit]
    
    for it in range(max_iter):
        # ─── Phase 1: Lévy flight ─────────────────────────────────
        for i in range(pop_size):
            new_x = _levy_perturb(rng, pop[i], encoding, scale=levy_scale)
            j = rng.randint(0, pop_size)
            new_f = float(fitness_fn(new_x))
            n_eval += 1
            if new_f < fit[j]:
                pop[j] = new_x
                fit[j] = new_f
                if new_f < gbest_fit:
                    gbest = _copy_solution(new_x)
                    gbest_fit = new_f
        
        # ─── Phase 2: Nest abandonment ────────────────────────────
        scores = fdb_scores(pop, fit, encoding, method=method)
        probs = _selection_probs(scores) if method != 'classic' else None
        
        for i in range(pop_size):
            if rng.rand() < pa:
                if method == 'classic':
                    new_x = encoding.init_individual(rng)
                else:
                    # FDB rehber (fdb: one-hot → hep FDB-best; rfdb: rulet) + 1 mutasyon
                    guide_idx = int(rng.choice(pop_size, p=probs))
                    new_x = encoding.mutate(rng, _copy_solution(pop[guide_idx]),
                                           rate=1.0 / max(1, encoding.dim))
                
                new_f = float(fitness_fn(new_x))
                n_eval += 1
                pop[i] = new_x
                fit[i] = new_f
                if new_f < gbest_fit:
                    gbest = _copy_solution(new_x)
                    gbest_fit = new_f
        
        history.append(gbest_fit)
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
    }


_BSOS_PHASES = ('mutualism', 'commensalism', 'parasitism')


def _bsos_select_partner(rng, pop, fit, encoding, i, method, phase,
                         fdb_phases, fdb_exclude_self, raw_scores):
    """BSOS partner seçimi (v0.3.3).

    phase, fdb_phases içinde değilse veya method='classic' ise partner
    uniform-random ≠ i seçilir (orijinal SOS ve resmi FDB-SOS kodunun
    FDB-dışı fazlardaki davranışı).

    FDB'li fazlarda:
    - 'fdb'  → HAM skor argmax'ı (deterministik). fdb_exclude_self=True ise
      i maskelenir (self hariç en iyi); False ise resmi kod gibi j==i olabilir.
    - 'rfdb' → ham skorlar rulet ağırlığıdır; fdb_exclude_self=True ise i'nin
      ağırlığı sıfırlanır. Dejenere toplamda uniform'a düşülür.

    raw_scores: fdb_scores(..., method='rfdb') çıktısı (ham nf+nd vektörü);
    tazeleme kararı (iter/selection) çağıran tarafta verilir.
    """
    pop_size = len(pop)
    if method == 'classic' or phase not in fdb_phases:
        avail = [j for j in range(pop_size) if j != i]
        return int(rng.choice(avail))
    if method == 'fdb':
        if fdb_exclude_self:
            masked = np.asarray(raw_scores, dtype=float).copy()
            masked[i] = -np.inf
            return int(np.argmax(masked))
        return int(np.argmax(raw_scores))
    # method == 'rfdb'
    w = np.maximum(np.asarray(raw_scores, dtype=float), 0.0).copy()
    if fdb_exclude_self:
        w[i] = 0.0
    s = w.sum()
    if s <= 1e-12:
        cand = ([j for j in range(pop_size) if j != i]
                if fdb_exclude_self else list(range(pop_size)))
        return int(rng.choice(cand))
    return int(rng.choice(pop_size, p=w / s))


def run_bsos(fitness_fn: Callable, encoding: Encoding,
             method: str = 'classic',
             pop_size: int = 20, max_iter: int = 50,
             seed: int = 42, verbose: bool = False,
             n_jobs: int = 1,
             fdb_phases=_BSOS_PHASES,
             fdb_refresh: str = 'iter',
             fdb_exclude_self: bool = True,
             fdb_reference: bool = False):
    """Universal Symbiotic Organisms Search — 3 fazlı (Cheng & Prayogo 2014).
    
    Phase 1 (Mutualism):
        İki organizma (i, j) karşılıklı fayda ile gbest'e yaklaşır.
        v3-final tasarım: encoding.crossover ile gbest-yönlü 2 child üret;
        her ikisini değerlendir, kabul edersen replace.
    
    Phase 2 (Commensalism):
        i, j'den etkilenmeden gbest'e yaklaşır.
        v3-final tasarım: i ile gbest'i crossover'la, child i'ye replace
        (yalnızca daha iyiyse).
    
    Phase 3 (Parasitism):
        i'den parazit (yüksek mutasyon) üretilir, partner j ile rekabet eder.
        v3-final tasarım: encoding.mutate(i, rate=0.3) → parazit,
        j'den iyiyse onun yerini alır.
    
    method:
        'classic' → tüm fazlarda partner = uniform random ≠ self
        'fdb'     → partner = FDB skoru argmax'ı (deterministik)
        'rfdb'    → partner = rulet-FDB ile seçilir
    
    FDB davranış parametreleri (v0.3.3 — resmi kod hizası için):
    fdb_phases : tuple, default = üç faz da
        FDB rehberliğinin uygulanacağı fazlar. Resmi FDB-SOS kodu (FEX
        #72311) YALNIZ ('mutualism',) kullanır; kütüphanenin tarihsel
        varsayılanı üç fazdır (FDB-uyarlaması). Bilinmeyen faz adı raise.
    fdb_refresh : {'iter', 'selection'}, default 'iter'
        'iter' → skorlar iterasyon başında bir kez hesaplanır (v3 davranışı).
        'selection' → her seçim çağrısında güncel popülasyonla yeniden
        hesaplanır (resmi kod davranışı).
    fdb_exclude_self : bool, default True
        True → partner asla i olamaz (v3 davranışı). False → resmi kod gibi
        argmax/rulet i'yi de seçebilir.
    fdb_reference : bool, default False
        True → üç parametreyi resmi kod hizasına zorlar:
        fdb_phases=('mutualism',), fdb_refresh='selection',
        fdb_exclude_self=False. ContinuousEncoding kullanılıyorsa
        distance_metric='manhattan' ŞARTTIR (değilse raise — sessiz metrik
        değişimi yapılmaz). Bu preset ile koşum, raporlarda
        "FDB-SOS (Kahraman 2020, resmi kod hizası)" olarak etiketlenebilir;
        preset dışı kombinasyonlar "FDB-uyarlaması"dır (faz3 §3.5).
    
    Encoding-aware: Tüm operatörler encoding.crossover, encoding.mutate
    üzerinden geliyor; Binary-OR/XOR varsayımları yok.
    
    Parameters
    ----------
    fitness_fn : callable
    encoding : Encoding
    method : {'classic', 'fdb', 'rfdb'}
    pop_size : int, default=20
    max_iter : int, default=50
    seed : int, default=42
    verbose : bool, default=False
    
    Returns
    -------
    dict with keys: best_sol, best_fit, history, n_evaluations
    """
    method = _normalize_method(method)
    if fdb_reference:
        fdb_phases = ('mutualism',)
        fdb_refresh = 'selection'
        fdb_exclude_self = False
        if (isinstance(encoding, ContinuousEncoding)
                and encoding.distance_metric != 'manhattan'):
            raise ValueError(
                "fdb_reference=True resmi MATLAB koduyla (FEX #72311) birebir "
                "hiza demektir ve HAM Manhattan mesafesi gerektirir. Encoding'i "
                "ContinuousEncoding(bounds, distance_metric='manhattan') olarak "
                "kurun. (Sessiz metrik değişimi yapılmaz — faz3 §3.5.)"
            )
    fdb_phases = tuple(fdb_phases)
    for _p in fdb_phases:
        if _p not in _BSOS_PHASES:
            raise ValueError(
                f"Bilinmeyen fdb_phases öğesi: {_p!r}. Geçerli: {_BSOS_PHASES}"
            )
    if fdb_refresh not in ('iter', 'selection'):
        raise ValueError(
            f"Bilinmeyen fdb_refresh: {fdb_refresh!r}. "
            f"Geçerli: 'iter' | 'selection'"
        )
    rng = np.random.RandomState(seed)
    
    pop = encoding.init_population(pop_size, rng)
    fit = _eval_pop(fitness_fn, pop, n_jobs=n_jobs)
    n_eval = pop_size
    
    bi = int(np.argmin(fit))
    gbest = _copy_solution(pop[bi])
    gbest_fit = float(fit[bi])
    history = [gbest_fit]
    
    # Parazit mutasyon oranı (encoding'e göre adapte)
    parasite_rate = max(0.2, 1.5 / max(1, encoding.dim))
    
    # v3.1.1: ContinuousEncoding için ORİJİNAL SOS aritmetiği (Cheng &
    # Prayogo 2014). Diğer encoding'lerde crossover-tabanlı UYARLAMA sürer
    # (aritmetik hamle binary/kategorik/permütasyonda tanımsız).
    is_continuous = isinstance(encoding, ContinuousEncoding)
    if is_continuous:
        _lo = encoding.bounds[:, 0]
        _hi = encoding.bounds[:, 1]
    
    for it in range(max_iter):
        # 'iter' modunda skorlar iterasyon başında BİR KEZ hesaplanır;
        # 'selection' modunda her seçim çağrısında güncel pop/fit ile
        # yeniden hesaplanır (resmi kod davranışı).
        raw_iter = (fdb_scores(pop, fit, encoding, method='rfdb')
                    if (method != 'classic' and fdb_refresh == 'iter') else None)
        
        def _partner(phase, i):
            raw = raw_iter
            if (method != 'classic' and phase in fdb_phases
                    and fdb_refresh == 'selection'):
                raw = fdb_scores(pop, fit, encoding, method='rfdb')
            return _bsos_select_partner(rng, pop, fit, encoding, i, method,
                                        phase, fdb_phases, fdb_exclude_self,
                                        raw)
        
        for i in range(pop_size):
            # ─── Phase 1: Mutualism ─────────────────────────────────
            j = _partner('mutualism', i)
            if is_continuous:
                # Orijinal: MV=(Xi+Xj)/2; BF∈{1,2};
                # Xnew = X + rand(0,1)·(Xbest − MV·BF)
                mv = (np.asarray(pop[i]) + np.asarray(pop[j])) / 2.0
                bf1 = rng.randint(1, 3)
                bf2 = rng.randint(1, 3)
                new_i = np.clip(pop[i] + rng.rand(encoding.dim) * (gbest - mv * bf1), _lo, _hi)
                new_j = np.clip(pop[j] + rng.rand(encoding.dim) * (gbest - mv * bf2), _lo, _hi)
            else:
                # UYARLAMA: i ve j ayrı ayrı gbest'e yaklaşır (crossover)
                new_i, _ = encoding.crossover(rng, pop[i], gbest)
                new_j, _ = encoding.crossover(rng, pop[j], gbest)
            
            f_i = float(fitness_fn(new_i)); n_eval += 1
            f_j = float(fitness_fn(new_j)); n_eval += 1
            if f_i < fit[i]:
                pop[i] = new_i; fit[i] = f_i
                if f_i < gbest_fit:
                    gbest = _copy_solution(new_i); gbest_fit = f_i
            if f_j < fit[j]:
                pop[j] = new_j; fit[j] = f_j
                if f_j < gbest_fit:
                    gbest = _copy_solution(new_j); gbest_fit = f_j
            
            # ─── Phase 2: Commensalism ──────────────────────────────
            j = _partner('commensalism', i)
            if is_continuous:
                # Orijinal: Xi_new = Xi + rand(−1,1)·(Xbest − Xj)
                new_i = np.clip(pop[i] + rng.uniform(-1.0, 1.0, encoding.dim)
                                * (gbest - np.asarray(pop[j])), _lo, _hi)
            else:
                # UYARLAMA: i, gbest'e doğru çekilir (j hareket etmez)
                new_i, _ = encoding.crossover(rng, pop[i], gbest)
            f_i = float(fitness_fn(new_i)); n_eval += 1
            if f_i < fit[i]:
                pop[i] = new_i; fit[i] = f_i
                if f_i < gbest_fit:
                    gbest = _copy_solution(new_i); gbest_fit = f_i
            
            # ─── Phase 3: Parasitism ────────────────────────────────
            j = _partner('parasitism', i)
            if is_continuous:
                # Orijinal: parazit = Xi kopyası, rastgele boyut alt-kümesi
                # uniform-random yeniden örneklenir; j ile rekabet eder.
                parasite = np.asarray(_copy_solution(pop[i]), dtype=float)
                n_mut = rng.randint(1, encoding.dim + 1)
                dims = rng.choice(encoding.dim, size=n_mut, replace=False)
                parasite[dims] = _lo[dims] + rng.rand(n_mut) * (_hi[dims] - _lo[dims])
            else:
                parasite = encoding.mutate(rng, _copy_solution(pop[i]),
                                           rate=parasite_rate)
            f_par = float(fitness_fn(parasite)); n_eval += 1
            if f_par < fit[j]:
                pop[j] = parasite; fit[j] = f_par
                if f_par < gbest_fit:
                    gbest = _copy_solution(parasite); gbest_fit = f_par
        
        history.append(gbest_fit)
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: best_fit = {gbest_fit:.6f}")
    
    return {
        'best_sol': gbest,
        'best_fit': gbest_fit,
        'history': history,
        'n_evaluations': n_eval,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 5: MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)
# ═══════════════════════════════════════════════════════════════════════════
# 
# NSGA-II (Deb et al. 2002) — multi-objective genetic algorithm.
# Pareto-optimal çözüm kümesi (Pareto front) üretir.
#
# Kullanım:
#   def multi_obj_fit(sol):
#       return np.array([rmse(sol), n_features(sol), cost(sol)])
#
#   result = run_nsga2(fitness_fn=multi_obj_fit, encoding=enc, n_objectives=3)
#   result['pareto_front']     # liste: non-dominated solutions
#   result['pareto_fitness']   # array: corresponding fitness vectors
# ═══════════════════════════════════════════════════════════════════════════


def _dominates(fit_a, fit_b):
    """fit_a, fit_b'yi domine eder mi? (minimization)
    
    a dominates b ↔ ∀i: a[i] <= b[i] AND ∃i: a[i] < b[i]
    """
    return np.all(fit_a <= fit_b) and np.any(fit_a < fit_b)


def _fast_non_dominated_sort(fitness_matrix):
    """NSGA-II'nin O(MN²) sıralaması.
    
    Parameters
    ----------
    fitness_matrix : ndarray, shape (N, M)
        N birey, M objektif
    
    Returns
    -------
    fronts : list of list of int
        fronts[i] = i. front'taki birey indeksleri.
        fronts[0] = Pareto front (non-dominated)
    """
    N = len(fitness_matrix)
    domination_count = np.zeros(N, dtype=int)  # kaç birey beni domine ediyor
    dominated_set = [[] for _ in range(N)]  # ben kimleri domine ediyorum
    rank = np.zeros(N, dtype=int)
    
    fronts = [[]]
    
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if _dominates(fitness_matrix[p], fitness_matrix[q]):
                dominated_set[p].append(q)
            elif _dominates(fitness_matrix[q], fitness_matrix[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    fronts.pop()  # son boş front
    return fronts


def _crowding_distance(fitness_matrix, front_indices):
    """NSGA-II crowding distance hesabı.
    
    Aynı front'taki bireylerin diversity'si için.
    """
    n = len(front_indices)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)
    
    n_obj = fitness_matrix.shape[1]
    distances = np.zeros(n)
    
    for m in range(n_obj):
        # Bu objektif için sırala
        sorted_idx = np.argsort(fitness_matrix[front_indices, m])
        
        # Boundary'lerin distance'ı sonsuz
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        
        f_min = fitness_matrix[front_indices[sorted_idx[0]], m]
        f_max = fitness_matrix[front_indices[sorted_idx[-1]], m]
        f_range = f_max - f_min
        if f_range < 1e-12:
            continue
        
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                fitness_matrix[front_indices[sorted_idx[k+1]], m] -
                fitness_matrix[front_indices[sorted_idx[k-1]], m]
            ) / f_range
    
    return distances


def _crowded_tournament(rng, ranks, distances):
    """NSGA-II tournament: önce rank, eşitse crowding distance."""
    n = len(ranks)
    i, j = rng.choice(n, 2, replace=False)
    if ranks[i] < ranks[j]:
        return i
    elif ranks[j] < ranks[i]:
        return j
    else:
        # Aynı front: daha büyük crowding distance kazansın
        if distances[i] > distances[j]:
            return i
        else:
            return j


def run_nsga2(fitness_fn, encoding,
              n_objectives,
              pop_size=50, max_iter=100,
              p_crossover=0.9,
              mutation_rate=None,
              constraints=None,
              seed=42, verbose=False):
    """NSGA-II — Multi-objective genetic algorithm with Pareto front.
    
    Parameters
    ----------
    fitness_fn : callable(solution) -> ndarray of shape (n_objectives,)
        Her objektif için bir değer döndürür (hepsi minimize edilir).
    encoding : Encoding
    n_objectives : int
        Objektif sayısı.
    pop_size : int, default=50
    max_iter : int, default=100
    p_crossover : float, default=0.9
    mutation_rate : float or None
    constraints : list of callable(solution) -> float or None
        Her constraint negatif → ihlal yok, pozitif → ihlal miktarı.
        Constraint domination kullanılır.
    seed : int, default=42
    
    Returns
    -------
    dict with keys:
        pareto_front     : list of solutions (Pareto-optimal)
        pareto_fitness   : ndarray (n_pareto, n_objectives)
        all_solutions    : final popülasyondaki tüm çözümler
        all_fitness      : ndarray (pop_size, n_objectives)
        history          : list of dicts, her gen için pareto front size
        n_evaluations    : toplam fitness değerlendirme
    """
    rng = np.random.RandomState(seed)
    pop = encoding.init_population(pop_size, rng)
    
    # Fitness değerlendirme (multi-objective)
    fitness_matrix = np.array([fitness_fn(s) for s in pop])
    n_eval = pop_size
    
    if fitness_matrix.shape[1] != n_objectives:
        raise ValueError(
            f"fitness_fn {fitness_matrix.shape[1]} objektif döndürdü ama "
            f"n_objectives={n_objectives} dendi"
        )
    
    # Constraint violation (eğer varsa)
    if constraints is not None:
        violations = np.array([
            sum(max(0, c(s)) for c in constraints) for s in pop
        ])
    else:
        violations = np.zeros(pop_size)
    
    history = []
    
    for it in range(max_iter):
        # ─── 1. Non-dominated sort ───
        fronts = _fast_non_dominated_sort(fitness_matrix)
        ranks = np.zeros(pop_size, dtype=int)
        for r, front in enumerate(fronts):
            for i in front:
                ranks[i] = r
        
        # ─── 2. Crowding distance ───
        all_distances = np.zeros(pop_size)
        for front in fronts:
            d = _crowding_distance(fitness_matrix, front)
            for k, idx in enumerate(front):
                all_distances[idx] = d[k]
        
        # ─── 3. Yeni nesil üret ───
        new_pop = []
        while len(new_pop) < pop_size:
            # Crowded tournament selection
            p1_idx = _crowded_tournament(rng, ranks, all_distances)
            p2_idx = _crowded_tournament(rng, ranks, all_distances)
            
            if rng.rand() < p_crossover:
                c1, c2 = encoding.crossover(rng, pop[p1_idx], pop[p2_idx])
            else:
                c1, c2 = _copy_solution(pop[p1_idx]), _copy_solution(pop[p2_idx])
            
            c1 = encoding.mutate(rng, c1, mutation_rate)
            c2 = encoding.mutate(rng, c2, mutation_rate)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        # Yeni fitness'lar
        new_fitness = np.array([fitness_fn(s) for s in new_pop])
        n_eval += pop_size
        
        # ─── 4. Combine + select (μ + λ) ───
        combined_pop = pop + new_pop
        combined_fit = np.vstack([fitness_matrix, new_fitness])
        
        # Constraint violation (eğer varsa)
        if constraints is not None:
            new_viol = np.array([
                sum(max(0, c(s)) for c in constraints) for s in new_pop
            ])
            combined_viol = np.concatenate([violations, new_viol])
        else:
            combined_viol = np.zeros(len(combined_pop))
        
        # Combined üzerinde sıralama
        combined_fronts = _fast_non_dominated_sort_with_constraints(
            combined_fit, combined_viol)
        
        # En iyi pop_size'ı seç
        next_pop_idx = []
        for front in combined_fronts:
            if len(next_pop_idx) + len(front) <= pop_size:
                next_pop_idx.extend(front)
            else:
                # Crowding distance ile sırala, sığacak kadar al
                d = _crowding_distance(combined_fit, front)
                sorted_by_distance = sorted(zip(front, d), key=lambda x: -x[1])
                remaining = pop_size - len(next_pop_idx)
                next_pop_idx.extend([idx for idx, _ in sorted_by_distance[:remaining]])
                break
        
        pop = [combined_pop[i] for i in next_pop_idx]
        fitness_matrix = combined_fit[next_pop_idx]
        violations = combined_viol[next_pop_idx]
        
        # History
        cur_pareto_size = len([i for i in range(len(pop)) 
                               if all(not _dominates(fitness_matrix[j], fitness_matrix[i])
                                      for j in range(len(pop)) if j != i)])
        history.append({
            'iter': it,
            'pareto_front_size': cur_pareto_size,
            'mean_fitness': fitness_matrix.mean(axis=0).tolist(),
        })
        
        if verbose and (it + 1) % 10 == 0:
            print(f"  iter {it+1}/{max_iter}: Pareto front size = {cur_pareto_size}")
    
    # ─── Final Pareto front ───
    final_fronts = _fast_non_dominated_sort(fitness_matrix)
    pareto_idx = final_fronts[0]
    pareto_solutions = [_copy_solution(pop[i]) for i in pareto_idx]
    pareto_fitness = fitness_matrix[pareto_idx]
    
    return {
        'pareto_front': pareto_solutions,
        'pareto_fitness': pareto_fitness,
        'all_solutions': pop,
        'all_fitness': fitness_matrix,
        'history': history,
        'n_evaluations': n_eval,
    }


def _fast_non_dominated_sort_with_constraints(fitness_matrix, violations):
    """NSGA-II constrained non-dominated sort (Deb 2000)."""
    N = len(fitness_matrix)
    domination_count = np.zeros(N, dtype=int)
    dominated_set = [[] for _ in range(N)]
    rank = np.zeros(N, dtype=int)
    fronts = [[]]
    
    def constraint_dominates(p, q):
        # p constraint-dominates q if:
        # - p is feasible and q is not, OR
        # - both are infeasible but p has less violation, OR
        # - both are feasible and p Pareto-dominates q
        p_feasible = violations[p] <= 1e-12
        q_feasible = violations[q] <= 1e-12
        if p_feasible and not q_feasible:
            return True
        if not p_feasible and q_feasible:
            return False
        if not p_feasible and not q_feasible:
            return violations[p] < violations[q]
        # Both feasible: Pareto domination
        return _dominates(fitness_matrix[p], fitness_matrix[q])
    
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if constraint_dominates(p, q):
                dominated_set[p].append(q)
            elif constraint_dominates(q, p):
                domination_count[p] += 1
        if domination_count[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    fronts.pop()
    return fronts


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 6: NSM ve EPM MODÜLLERİ (Encoding-Aware)
# ═══════════════════════════════════════════════════════════════════════════
#
# NSM (Natural Survivor Method, Kahraman et al. 2023, EAAI 122:106121):
#   Survivor selection adımında klasik greedy karşılaştırma yerine
#   3 bileşenli (OFS = fitness, MPS, PS) ağırlıklı skor kullanır.
#
#   ⚠ UYARLAMA BEYANI (bu implementasyon makaleyle BİREBİR DEĞİLDİR):
#   1. Makalede MPS = mating pool'daki rehberlere uzaklık; burada TEK best
#      bireye uzaklık ile yaklaşıklanır.
#   2. Makalede PS = P popülasyonuna katkı (bireylere uzaklık); burada
#      top-K bireyin merkezine uzaklık ile yaklaşıklanır.
#   3. nsm_decide'daki erken-çıkış kuralları (f_new < f_best → kabul,
#      f_old == f_best → koru) makalede yoktur; eklenmiş elitizm korumalarıdır.
#   4. sigmoid_switch'in "erken greedy → geç NSM" yönü ve sabit 15 eğimi bir
#      TASARIM SEÇİMİDİR; makale linear/step/sigmoid/kaotik sinyalleri iki
#      yönde de tarar ve en iyi strateji problem-bağımlıdır (mak. Table 2).
#   Bir yayında bu modülü kullanırken "NSM-uyarlaması" olarak adlandır ve
#   atfı orijinale ver (faz3 §3.5 varyant adlandırma kuralı).
#
# EPM (Evolutionary Population Management, Üstünsoy et al. 2025 KBS 328:114221):
#   3 hipotez:
#     H1: Epoch boyunca update yok (tüm child'lar üretilir, sonra topluca seçim)
#     H2: U = P ∪ Pchild'dan yarısı fitness-bazlı, yarısı FDB-bazlı seçilir
#         (forbidden list ile parent-child eşleri ayrılır)
#     H3: Yeni P'nin fit-half ve fdb-half'ı convergence için eşleştirilir
#         (crossover'da parent2 = epm_guide_select sonucu)
#
#   ✔ RESMİ KOD DOĞRULAMASI (10 Tem 2026, FEX #181754, v0.3.4):
#   epm_population_update() seçim indeksleri resmi pop_productions.m
#   (Method-1) ve pop_productions_2.m (Method-2) transkriptleriyle 10.000
#   rastgele popülasyonda BİREBİR özdeş (validation/epm_matlab_crosscheck.py
#   T1–T3). Resmi kod makale pseudo-kodundaki iki belirsizliği şöyle çözer
#   ve kütüphane AYNI taraftadır: (a) U = [P; Pchild] — ebeveynler önce
#   (makale Eq. 6'nın tersi); (b) akraba ofseti ±n (Algorithm-2'deki
#   "n/2" yazımında n = |U| = 2n'dir). Method-1/2 FDB skoru resmi kodda
#   HAM Manhattan kullanır → continuous'ta hiza için
#   ContinuousEncoding(distance_metric='manhattan') gerekir.
#
#   ⚠ UYARLAMA BEYANI (kalan sapmalar — bilinçli):
#   1. Eşit-fitness kenar durumu: resmi kod randperm (non-deterministik);
#      kütüphane deterministik davranır (kanonik sayı üretimi önceliği).
#   2. Doldurma prosedürü: FL nedeniyle fdb-yarı dolamazsa kütüphane kalan
#      en iyi bireyle doldurur; resmi kodda tanımsız.
#   3. H3 rehber eşleşmesi: resmi kod algoritma/Case-özgü stratejiler
#      kullanır (ör. EPM-AGDE V2'de h↔h+45, AGDE'nin ilk-5/son-5 şablonu);
#      kütüphane makale Fig. 4 kanonik eşleşmesini (i↔i+n/2) genelleştirir.
#   4. H1 ayrışması: resmi EPM-AGDE kodu epoch İÇİNDE açgözlü ebeveyn
#      değiştirmeyi (DE seçimi) KORUR — makale Algorithm-4'te bu satırlar
#      çıkarılmıştır (makale-kod farkı; SAVUNMA_DOSYASI kalemi). Kütüphane
#      makale tarafındadır: epoch içinde P güncellenmez.
#   Yayın çerçevesi: H2 çekirdeği "resmi kodla doğrulanmış"; algoritma-bütünü
#   iddiaları için ad "EPM-uyarlaması", atıf orijinale (faz3 §3.5).
#
# Encoding-Aware Tasarım Notu:
#   v2'deki NSM/EPM kodları binary varsayıyordu (Manhattan distance, XOR-Hamming).
#   v3'te tüm distance'lar `encoding.distance(a,b)` üzerinden, popülasyon
#   merkezi `encoding.population_center(pop)` üzerinden geliyor.
#   Sonuç: NSM/EPM artık Continuous, Categorical, Mixed uzaylarda da çalışıyor.
# ═══════════════════════════════════════════════════════════════════════════

# ─── NSM Tuning Preset'leri ───────────────────────────────────────────────
# katsayi[k] = (w_OFS, w_PS, w_MPS)
# Kaynak: Kahraman grubunun paylaştığı MATLAB NSM.m dosyasından aktarım
# (ders projesi döneminde doğrulandı). Bu sayılar 2023 NSM makalesinin
# METNİNDE yer almaz — provenance MATLAB kodudur.
NSM_PRESETS = {
    1:  (0.9548, 0.1114, 0.1188),
    2:  (0.9388, 0.8586, 0.2677),
    3:  (0.8402, 0.4538, 0.1253),
    4:  (0.8883, 0.2704, 0.1291),
    5:  (0.7290, 0.3029, 0.1034),
    6:  (0.9882, 0.6862, 0.6868),  # ← TLABC Case-1'de kullanılan (default)
    7:  (0.6899, 0.1294, 0.1152),
    8:  (0.9012, 0.2746, 0.1838),
    9:  (0.9055, 0.4235, 0.1602),
    10: (0.8271, 0.3483, 0.2723),
}


def _norm_minmax_safe(values: np.ndarray) -> np.ndarray:
    """Min-max normalize; max==min ise sıfır vektör döner."""
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def sigmoid_switch(t: int, T: int) -> float:
    """t/T oranına göre NSM uygulama olasılığı.
    
    Erken iter (t/T küçük) → ~0 (greedy hakim)
    Geç iter   (t/T büyük) → ~1 (NSM hakim)
    sigmoid(15·(progress − 0.5)) ile yumuşak geçiş; orta noktada %50.

    Not: 15 eğimi ve "geç iter'de NSM" yönü bu kütüphanenin tasarım
    seçimidir; Kahraman 2023 makalesinin kanonik stratejisi değildir
    (makale birden çok switch sinyali tarar).
    """
    if T <= 0:
        return 0.0
    p = t / T
    z = 15.0 * (p - 0.5)
    z = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def dynamic_weights(fes: int, max_fes: int, frequency: int = 4):
    """Dinamik FDB ağırlıkları (MATLAB calculateFddbMultiplier birebir).
    
    frequency=0 → sabit (0.6, 0.25, 0.15) döner.
    frequency>0 → testereyle artan w_OFS profili.
    
    Returns
    -------
    (w_OFS, w_PS, w_MPS)
    """
    if frequency == 0:
        return (0.6, 0.25, 0.15)
    fx = max(1, round(max_fes / frequency))
    y = fes % fx
    w_ofs = (y / fx) * (1 - 0.4) + 0.4
    w_ps = (1 - w_ofs) / 2.4
    w_mps = 1.4 * w_ps
    return (w_ofs, w_ps, w_mps)


def _top_k_center(population: list, fit: np.ndarray, encoding: 'Encoding',
                  merkez_sayisi: int):
    """Top-K (en iyi) bireyin encoding-spesifik merkezi.
    
    v2'deki `_pop_merkez(pop, fit, K)` fonksiyonunun encoding-aware versiyonu.
    K<=1 ise K=2 (en az 2 birey gerekli).
    """
    k = max(2, int(merkez_sayisi))
    k = min(k, len(population))
    sort_idx = np.argsort(fit)  # küçük=iyi (minimization)
    top_k = [population[i] for i in sort_idx[:k]]
    return encoding.population_center(top_k)


def nsm_decide(x_old, f_old: float, x_new, f_new: float,
               population: list, fit: np.ndarray,
               encoding: 'Encoding',
               merkez_sayisi: int = 5,
               weights: tuple = None) -> bool:
    """X_old → X_new geçişine NSM skor karşılaştırmasıyla karar verir.
    
    Parameters
    ----------
    x_old, x_new : Any
        Encoding'in kabul ettiği iki birey (eski ve yeni aday).
    f_old, f_new : float
        Fitness değerleri (küçük = iyi).
    population : list
        Mevcut popülasyon (NSM bileşenlerinin hesabında kullanılır).
    fit : np.ndarray
        Popülasyon fitness'i (population ile aynı sırada).
    encoding : Encoding
        distance() ve population_center() metotlarını sağlayan encoding.
    merkez_sayisi : int, default=5
        PS bileşenindeki top-K merkez hesabı için K.
    weights : (w_OFS, w_PS, w_MPS), default=NSM_PRESETS[6]
        Ağırlıklar.
    
    Returns
    -------
    bool : True → x_new'i kabul et, False → x_old'u koru.
    
    Notes
    -----
    Erken çıkış kuralları (v2 NSM.m birebir):
        - Popülasyon dejenere (f_best == f_worst) → True
        - f_new < f_best (yeni en iyiyi yener) → True
        - f_old == f_best (eski zaten en iyi, dokunma) → False
    Aksi halde 3 bileşenli ağırlıklı skor karşılaştırması.
    """
    if weights is None:
        weights = NSM_PRESETS[6]
    
    fit = np.asarray(fit, dtype=float)
    f_best = float(fit.min())
    f_worst = float(fit.max())
    
    # ─── Erken çıkış kuralları ───
    if f_best == f_worst:
        return True
    if f_new < f_best:
        return True
    if f_old == f_best:
        return False
    
    w_ofs, w_ps, w_mps = weights
    
    # X_best (popülasyondaki en iyi birey)
    best_idx = int(np.argmin(fit))
    x_best = population[best_idx]
    
    # Top-K merkez (encoding-aware)
    x_merkez = _top_k_center(population, fit, encoding, merkez_sayisi)
    
    # OFS: 1 − (f − f_best) / (f_worst − f_best)
    denom = f_worst - f_best
    ofs_old = 1.0 - (f_old - f_best) / denom
    ofs_new = 1.0 - (f_new - f_best) / denom
    
    # MPS: x_best'e encoding.distance() — pop maxine bölerek normalize
    pop_dists_to_best = np.array([encoding.distance(p, x_best) for p in population])
    pop_dist_max = float(pop_dists_to_best.max())
    if pop_dist_max <= 1e-12:
        mps_old = mps_new = 0.0
    else:
        mps_old = encoding.distance(x_old, x_best) / pop_dist_max
        mps_new = encoding.distance(x_new, x_best) / pop_dist_max
    
    # PS: pop merkez'e encoding.distance() — min-max normalize
    pop_dists_to_merkez = np.array([encoding.distance(p, x_merkez) for p in population])
    pmm_min = float(pop_dists_to_merkez.min())
    pmm_range = float(pop_dists_to_merkez.max()) - pmm_min
    if pmm_range <= 1e-12:
        ps_old = ps_new = 0.0
    else:
        ps_old = (encoding.distance(x_old, x_merkez) - pmm_min) / pmm_range
        ps_new = (encoding.distance(x_new, x_merkez) - pmm_min) / pmm_range
    
    score_old = w_ofs * ofs_old + w_ps * ps_old + w_mps * mps_old
    score_new = w_ofs * ofs_new + w_ps * ps_new + w_mps * mps_new
    
    return score_new > score_old


# ─── EPM: U-matrix üzerinde FDB skoru ─────────────────────────────────────
def _fdb_scores_to_ref(U_pop: list, U_fit: np.ndarray, ref_idx: int,
                       encoding: 'Encoding') -> np.ndarray:
    """U[ref_idx]'e göre FDB skoru — score_j = (1 − normFitness_j) + normDist(U_j, U_ref).
    
    Encoding-aware: distance metric encoding'ten gelir.
    """
    fmin, fmax = float(U_fit.min()), float(U_fit.max())
    if fmax == fmin:
        nf = np.ones(len(U_fit))
    else:
        nf = 1.0 - (U_fit - fmin) / (fmax - fmin)
    
    ref = U_pop[ref_idx]
    dist = np.array([encoding.distance(s, ref) for s in U_pop], dtype=float)
    dmin, dmax = float(dist.min()), float(dist.max())
    if dmax == dmin:
        nd = np.zeros(len(dist))
    else:
        nd = (dist - dmin) / (dmax - dmin)
    
    return nf + nd


def _fdb_scores_to_best(U_pop: list, U_fit: np.ndarray,
                        encoding: 'Encoding') -> np.ndarray:
    """En iyi U bireyine göre FDB skoru (Method-2)."""
    best_idx = int(np.argmin(U_fit))
    return _fdb_scores_to_ref(U_pop, U_fit, best_idx, encoding)


def epm_population_update(P: list, F: np.ndarray,
                          P_child: list, F_child: np.ndarray,
                          encoding: 'Encoding',
                          method: str = 'classic'):
    """EPM Hipotez-2: U = P ∪ Pchild'dan yarısı fitness, yarısı FDB ile seçilir.
    
    Parameters
    ----------
    P, F           : parent popülasyonu (n birey) ve fitness'ı
    P_child, F_child: çocuk popülasyonu (n birey) ve fitness'ı
    encoding       : distance() metric için
    method         : 'classic'      → Method-2 (tek-best FDB referansı)
                     'fdb' / 'rfdb' → Method-1 (her parent için ayrı FDB)
    
    Returns
    -------
    new_P, new_F, fit_half_picks, fdb_half_picks
        new_P/new_F : yeni P (n birey) ve fitness
        fit_half_picks : ilk n/2 indekslerin "U üzerindeki orijinal indeksleri"
        fdb_half_picks : ikinci n/2 indekslerin "U üzerindeki orijinal indeksleri"
    
    Notes
    -----
    U-matrix sıralaması: [P_0..P_{n-1}, Pchild_0..Pchild_{n-1}]
    → U-index i (i<n) ile (i+n) parent-child çiftidir (forbidden list).
    """
    n = len(P)
    half = n // 2
    
    U_pop = list(P) + list(P_child)  # 2n birey (encoding'e bağlı tip)
    U_fit = np.concatenate([np.asarray(F, dtype=float),
                            np.asarray(F_child, dtype=float)])
    
    # ─── Yarı 1: Fitness-bazlı seçim + forbidden list ───
    rank_idx = np.argsort(U_fit)  # küçük=iyi
    forbidden = set()
    fit_half_picks = []
    
    for i in rank_idx:
        i = int(i)
        if i in forbidden:
            continue
        fit_half_picks.append(i)
        if i < n:
            forbidden.add(i)
            forbidden.add(i + n)
        else:
            forbidden.add(i)
            forbidden.add(i - n)
        if len(fit_half_picks) >= half:
            break
    
    # ─── Yarı 2: FDB-bazlı seçim + forbidden list ───
    fdb_half_picks = []
    
    if method == 'classic':
        # Method-2: tek FDB skor matrisi (U-best referans)
        scores = _fdb_scores_to_best(U_pop, U_fit, encoding)
        ordered = np.argsort(-scores)
        for j in ordered:
            j = int(j)
            if j in forbidden:
                continue
            fdb_half_picks.append(j)
            if j < n:
                forbidden.add(j)
                forbidden.add(j + n)
            else:
                forbidden.add(j)
                forbidden.add(j - n)
            if len(fdb_half_picks) >= half:
                break
    else:
        # Method-1: her fit-half üyesi için ayrı FDB skoru
        for ref_i in fit_half_picks:
            if len(fdb_half_picks) >= half:
                break
            scores = _fdb_scores_to_ref(U_pop, U_fit, ref_i, encoding)
            ordered = np.argsort(-scores)
            for j in ordered:
                j = int(j)
                if j in forbidden:
                    continue
                fdb_half_picks.append(j)
                if j < n:
                    forbidden.add(j)
                    forbidden.add(j + n)
                else:
                    forbidden.add(j)
                    forbidden.add(j - n)
                break  # her fit-half üyesi için sadece 1 FDB eşi
    
    # ─── Yetersiz seçim durumunda doldur ───
    while len(fdb_half_picks) < half:
        avail = [k for k in range(2 * n) if k not in forbidden]
        if avail:
            best_avail = int(min(avail, key=lambda k: U_fit[k]))
            fdb_half_picks.append(best_avail)
            forbidden.add(best_avail)
        else:
            fdb_half_picks.append(int(np.argmin(U_fit)))
    
    # ─── Yeni P ve F oluştur ───
    all_picks = fit_half_picks + fdb_half_picks
    new_P = [_copy_solution(U_pop[k]) for k in all_picks]
    new_F = U_fit[all_picks].copy()
    
    return new_P, new_F, fit_half_picks, fdb_half_picks


def epm_guide_select(i: int, n: int, rng=None):
    """EPM Hipotez-3: i-inci bireyin convergence eşi.
    
    Yeni P'nin ilk yarısı [0..n/2-1] = fit-half, ikinci yarısı [n/2..n-1] = fdb-half.
    H2 sürecinde fit_half_picks[k] ile fdb_half_picks[k] aynı k'da eşleştirildi.
    
        - i fit-half'ta (i < n/2) ise, FDB eşi: index (i + n/2)
        - i fdb-half'ta (i >= n/2) ise, fit eşi: index (i − n/2)
    
    Returns
    -------
    (mate_idx, secondary_idx)
        primary FDB/fit eşi ve yedek random partner (rng verilmişse)
    """
    half = n // 2
    if i < half:
        mate = i + half
    else:
        mate = i - half
    
    if rng is None:
        return mate, mate
    
    others = [k for k in range(n) if k != i and k != mate]
    if not others:
        return mate, mate
    secondary = int(rng.choice(others))
    return mate, secondary


def epm_nsm_population_update(P: list, F: np.ndarray,
                              P_child: list, F_child: np.ndarray,
                              encoding: 'Encoding',
                              method: str = 'classic',
                              switch_prob: float = 0.0,
                              nsm_weights: tuple = None,
                              nsm_merkez_sayisi: int = 5,
                              rng: np.random.RandomState = None):
    """EPM Algoritma 2'nin NSM-süzgeçli versiyonu (B3 hibrit yardımcısı).
    
    EPM'in fit-half seçim kararına sigmoid switch ile NSM süzgeci ekler.
    fdb-half tarafı klasik EPM ile aynı kalır (NSM yok).
    
    Parameters
    ----------
    P, F, P_child, F_child : EPM input
    encoding : distance() ve population_center() için
    method : 'classic' (Method-2) veya 'fdb'/'rfdb' (Method-1)
    switch_prob : NSM uygulama olasılığı [0,1]. 0 → klasik EPM birebir.
    nsm_weights : (w_OFS, w_PS, w_MPS) — None → NSM_PRESETS[6]
    nsm_merkez_sayisi : NSM PS top-K
    rng : Bernoulli kararı için RandomState (None → seed=0)
    
    Returns
    -------
    new_P, new_F, fit_half_picks, fdb_half_picks
    """
    if nsm_weights is None:
        nsm_weights = NSM_PRESETS[6]
    if rng is None:
        rng = np.random.RandomState(0)
    
    n = len(P)
    half = n // 2
    
    U_pop = list(P) + list(P_child)
    U_fit = np.concatenate([np.asarray(F, dtype=float),
                            np.asarray(F_child, dtype=float)])
    
    # ─── Yarı 1: Fitness-bazlı + NSM süzgeci + forbidden list ───
    rank_idx = np.argsort(U_fit)
    forbidden = set()
    fit_half_picks = []
    
    for cand in rank_idx:
        cand = int(cand)
        if cand in forbidden:
            continue
        
        # İlk eleman her zaman alınır (en iyi U bireyi — elite koruma)
        if len(fit_half_picks) == 0:
            accept = True
        else:
            use_nsm = (rng.rand() < switch_prob)
            if use_nsm:
                # cand (yeni) ile şimdiye kadar alınmış en zayıf fit-half üyesini karşılaştır
                weakest_idx = fit_half_picks[-1]
                accept = nsm_decide(
                    U_pop[weakest_idx], float(U_fit[weakest_idx]),
                    U_pop[cand], float(U_fit[cand]),
                    U_pop, U_fit, encoding,
                    merkez_sayisi=nsm_merkez_sayisi, weights=nsm_weights,
                )
            else:
                accept = True  # klasik EPM greedy
        
        if accept:
            fit_half_picks.append(cand)
            if cand < n:
                forbidden.add(cand)
                forbidden.add(cand + n)
            else:
                forbidden.add(cand)
                forbidden.add(cand - n)
        
        if len(fit_half_picks) >= half:
            break
    
    # NSM çok sıkı reddederse fit_half eksik kalabilir
    while len(fit_half_picks) < half:
        avail = [k for k in range(2 * n) if k not in forbidden]
        if not avail:
            fit_half_picks.append(int(np.argmin(U_fit)))
            break
        best_avail = int(min(avail, key=lambda k: U_fit[k]))
        fit_half_picks.append(best_avail)
        if best_avail < n:
            forbidden.add(best_avail)
            forbidden.add(best_avail + n)
        else:
            forbidden.add(best_avail)
            forbidden.add(best_avail - n)
    
    # ─── Yarı 2: FDB-bazlı (B2 ile birebir, NSM yok) ───
    fdb_half_picks = []
    
    if method == 'classic':
        scores = _fdb_scores_to_best(U_pop, U_fit, encoding)
        ordered = np.argsort(-scores)
        for j in ordered:
            j = int(j)
            if j in forbidden:
                continue
            fdb_half_picks.append(j)
            if j < n:
                forbidden.add(j)
                forbidden.add(j + n)
            else:
                forbidden.add(j)
                forbidden.add(j - n)
            if len(fdb_half_picks) >= half:
                break
    else:
        for ref_i in fit_half_picks:
            if len(fdb_half_picks) >= half:
                break
            scores = _fdb_scores_to_ref(U_pop, U_fit, ref_i, encoding)
            ordered = np.argsort(-scores)
            for j in ordered:
                j = int(j)
                if j in forbidden:
                    continue
                fdb_half_picks.append(j)
                if j < n:
                    forbidden.add(j)
                    forbidden.add(j + n)
                else:
                    forbidden.add(j)
                    forbidden.add(j - n)
                break
    
    while len(fdb_half_picks) < half:
        avail = [k for k in range(2 * n) if k not in forbidden]
        if avail:
            best_avail = int(min(avail, key=lambda k: U_fit[k]))
            fdb_half_picks.append(best_avail)
            forbidden.add(best_avail)
        else:
            fdb_half_picks.append(int(np.argmin(U_fit)))
    
    all_picks = fit_half_picks + fdb_half_picks
    new_P = [_copy_solution(U_pop[k]) for k in all_picks]
    new_F = U_fit[all_picks].copy()
    
    return new_P, new_F, fit_half_picks, fdb_half_picks


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 7: NATIVE CONSTRAINT HANDLING
# ═══════════════════════════════════════════════════════════════════════════
# 
# Single-objective algoritmalara constraint desteği eklemek için wrapper.
#
# 3 strateji desteklenir:
#   1. 'penalty'      → manuel penalty (eski v2 yaklaşımı)
#   2. 'death_penalty' → infeasible çözümler aşırı yüksek fitness alır
#   3. 'feasibility_rules' → Deb's feasibility rules (önerilen):
#       - Iki feasible çözüm: daha düşük fitness kazanır
#       - Bir feasible + bir infeasible: feasible kazanır
#       - Iki infeasible: daha az ihlal eden kazanır
#
# Kullanım:
#   def fit(sol):
#       return -(sol * values).sum()  # max value
#
#   def weight_constraint(sol):
#       return (sol * weights).sum() - capacity  # >0 = ihlal
#
#   wrapped_fit = make_constrained_fitness(
#       fitness_fn=fit,
#       constraints=[weight_constraint],
#       method='feasibility_rules',
#   )
#   result = run_bga(fitness_fn=wrapped_fit, encoding=enc, ...)
# ═══════════════════════════════════════════════════════════════════════════


def make_constrained_fitness(fitness_fn, constraints, method='feasibility_rules',
                              penalty_factor=1e6):
    """Constraint'li single-objective fitness wrapper'ı oluşturur.
    
    Parameters
    ----------
    fitness_fn : callable(solution) -> float
        Asıl objektif (minimize edilir).
    constraints : list of callable(solution) -> float
        Her constraint fonksiyonu:
        - sonuç <= 0: feasible (ihlal yok)
        - sonuç > 0: infeasible, değer ihlal miktarı
    method : {'feasibility_rules', 'death_penalty', 'penalty'}
        - 'feasibility_rules': Deb 2000 — infeasible çözümler her feasibledan kötü
        - 'death_penalty': infeasible → 1e10 (sabit yüksek değer)
        - 'penalty': fitness + penalty_factor * total_violation
    penalty_factor : float, default=1e6
        Sadece 'penalty' modunda kullanılır.
    
    Returns
    -------
    wrapped_fn : callable(solution) -> float
        run_bga, run_bde gibi algoritmalara verilebilir.
    """
    if method == 'feasibility_rules':
        # Trick: feasible'lar normal fitness, infeasible'lar
        # daima daha kötü (yüksek) ve aralarında ihlal miktarına göre sıralanır.
        # Pratik uygulama: feasible için fitness, infeasible için
        # (worst_feasible_value + 1) + violation_sum şeklinde.
        # Ama biz fitness range'i bilmediğimiz için basit bir yöntem:
        # fitness + 1e8 * violation_indicator + 1e6 * violation_amount
        def wrapped(sol):
            base = fitness_fn(sol)
            violations = [c(sol) for c in constraints]
            total_viol = sum(max(0, v) for v in violations)
            if total_viol > 0:
                # Infeasible: çok büyük penalty + ihlal miktarına göre sıralama
                return 1e8 + total_viol * 1e3 + base * 1e-6
            return base
        return wrapped
    
    elif method == 'death_penalty':
        def wrapped(sol):
            for c in constraints:
                if c(sol) > 0:
                    return 1e10
            return fitness_fn(sol)
        return wrapped
    
    elif method == 'penalty':
        def wrapped(sol):
            base = fitness_fn(sol)
            total_viol = sum(max(0, c(sol)) for c in constraints)
            return base + penalty_factor * total_viol
        return wrapped
    
    else:
        raise ValueError(f"Bilinmeyen constraint method: {method}. "
                         f"'feasibility_rules', 'death_penalty', 'penalty' kullanın.")


def evaluate_constraints(solution, constraints):
    """Bir çözümün constraint durumunu raporlar.
    
    Returns
    -------
    dict with keys:
        feasible : bool
        violations : list of float (her constraint için)
        total_violation : float
    """
    violations = [float(c(solution)) for c in constraints]
    return {
        'feasible': all(v <= 1e-12 for v in violations),
        'violations': violations,
        'total_violation': sum(max(0, v) for v in violations),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 8: AUTO-TUNING META-LOOP
# ═══════════════════════════════════════════════════════════════════════════
# 
# Otomatik algoritma + hyperparameter seçimi.
# Sabit fitness eval bütçesi içinde (algoritma × method × pop_size) tüm
# kombinasyonları küçük denemelerle test eder, en iyiyi tam bütçeyle koşar.
# ═══════════════════════════════════════════════════════════════════════════


def tune_and_run(fitness_fn, encoding,
                 algorithms=('bga', 'bde', 'babc'),
                 methods=('classic', 'fdb'),
                 pop_sizes=(10, 20),
                 n_trials_per_config=2,
                 budget=2000,
                 seed=42,
                 n_jobs=1,
                 verbose=False):
    """Algoritma + hyperparameter konfigürasyonu otomatik seçer ve final koşum yapar.
    
    İki aşamalı:
        1. Trial fazı: Her (alg, method, pop_size) kombinasyonunu küçük bütçeyle
           (toplam_budget'in %30'u) n_trials_per_config kez koşar, ortalama fit'i alır.
        2. Final faz: En iyi konfigürasyonu kalan bütçenin tamamı (~%70) ile koşar.
    
    Parameters
    ----------
    fitness_fn : callable
    encoding : Encoding
    algorithms : iterable of str
        'bga', 'bde', 'bpso', 'babc', 'bgwo', 'bcs', 'bsos'
    methods : iterable of str
        'classic', 'fdb', 'rfdb'
    pop_sizes : iterable of int
    n_trials_per_config : int
        Her konfigürasyon için bağımsız koşum sayısı (farklı seed'lerle).
    budget : int
        Toplam fitness eval bütçesi.
    seed : int
        Master seed (her konfigürasyon için seed+offset kullanılır).
    verbose : bool
    
    Returns
    -------
    dict with keys:
        best_config : (alg, method, pop_size) tuple — en iyi
        best_result : final koşumun result dict'i
        trial_results : list of trial özetleri
        total_evaluations : kullanılan toplam eval sayısı
    """
    methods = tuple(_normalize_method(m) for m in methods)  # fail-fast
    runners = {
        'bga': run_bga, 'bde': run_bde, 'bpso': run_bpso,
        'babc': run_babc, 'bgwo': run_bgwo,
        'bcs': run_bcs, 'bsos': run_bsos,
    }
    
    # 1. Trial faz: %30 bütçe, her config'i hızlı dene
    trial_budget = int(budget * 0.3)
    final_budget = budget - trial_budget
    
    trial_results = []
    n_configs = len(algorithms) * len(methods) * len(pop_sizes)
    per_trial_budget = max(1, trial_budget // (n_configs * n_trials_per_config))
    
    if verbose:
        print(f"Auto-tune: {n_configs} configs × {n_trials_per_config} trials = "
              f"{n_configs * n_trials_per_config} koşum, her biri ~{per_trial_budget} eval")
    
    eval_count = 0
    for alg in algorithms:
        if alg not in runners:
            continue
        runner = runners[alg]
        for method in methods:
            for ps in pop_sizes:
                # max_iter = budget / pop_size (yaklaşık)
                trial_iters = max(2, per_trial_budget // ps)
                trial_fits = []
                for trial_idx in range(n_trials_per_config):
                    try:
                        r = runner(fitness_fn, encoding,
                                   method=method, pop_size=ps,
                                   max_iter=trial_iters,
                                   seed=seed + trial_idx,
                                   n_jobs=n_jobs)
                        trial_fits.append(r['best_fit'])
                        eval_count += r['n_evaluations']
                    except (NotImplementedError, ValueError):
                        # Bu config bu encoding'de desteklenmiyor (BPSO+Categorical vb.)
                        trial_fits.append(float('inf'))
                
                mean_fit = float(np.mean(trial_fits))
                trial_results.append({
                    'alg': alg, 'method': method, 'pop_size': ps,
                    'mean_fit': mean_fit, 'fits': trial_fits,
                })
                if verbose:
                    print(f"  {alg:5s} {method:11s} ps={ps:3d}: mean_fit={mean_fit:.6f}")
    
    # En iyi konfig
    valid_results = [r for r in trial_results if r['mean_fit'] != float('inf')]
    if not valid_results:
        raise RuntimeError("Hiçbir konfigürasyon çalışmadı.")
    best = min(valid_results, key=lambda r: r['mean_fit'])
    
    if verbose:
        print(f"En iyi config: {best['alg']} + {best['method']} + ps={best['pop_size']} "
              f"(mean_fit={best['mean_fit']:.6f})")
    
    # 2. Final koşum: en iyi config, kalan bütçe tamamı
    final_iters = max(10, final_budget // best['pop_size'])
    final_runner = runners[best['alg']]
    final_result = final_runner(
        fitness_fn, encoding,
        method=best['method'], pop_size=best['pop_size'],
        max_iter=final_iters, seed=seed, n_jobs=n_jobs,
    )
    eval_count += final_result['n_evaluations']
    
    return {
        'best_config': (best['alg'], best['method'], best['pop_size']),
        'best_result': final_result,
        'trial_results': trial_results,
        'total_evaluations': eval_count,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BÖLÜM 9: VERİ-ODAKLI HIZLI BAŞLANGIÇ (v3.1)
# ═══════════════════════════════════════════════════════════════════════════
#
# Amaç: "Elimde bir CSV var (sağlık verisi, oyun telemetrisi, satış verisi...),
# hangi sütunlar önemli?" sorusunu TEK fonksiyon çağrısıyla cevaplamak.
# Kütüphanenin çekirdeği (run_* fonksiyonları) sklearn'e bağımlı DEĞİLDİR;
# bu bölüm opsiyoneldir ve sklearn/pandas yoksa açıklayıcı hata verir.
# ═══════════════════════════════════════════════════════════════════════════


def load_csv_dataset(path, target, drop=None, encoding='utf-8'):
    """CSV dosyasından (X, y) çifti üretir. pandas gerektirir.

    Sayısal olmayan özellik sütunları otomatik düşürülür (uyarıyla) —
    kategorik sütunları kullanmak istiyorsan önce kendin one-hot/ordinal
    kodla, sonra select_features'a DataFrame olarak ver.

    Parameters
    ----------
    path : str
        CSV dosya yolu.
    target : str
        Hedef değişkenin sütun adı.
    drop : list of str or None
        Özellik olarak KULLANILMAYACAK sütunlar (ID, isim, tarih vb.).
    encoding : str, default='utf-8'
        Dosya encoding'i.

    Returns
    -------
    (X, y) : (pandas.DataFrame, pandas.Series)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "load_csv_dataset pandas gerektirir: pip install pandas"
        )
    df = pd.read_csv(path, encoding=encoding)
    if target not in df.columns:
        raise ValueError(
            f"target='{target}' CSV'de yok. Mevcut sütunlar: {list(df.columns)}"
        )
    drop = list(drop) if drop else []
    missing_drop = [c for c in drop if c not in df.columns]
    if missing_drop:
        raise ValueError(f"drop listesindeki sütun(lar) CSV'de yok: {missing_drop}")

    y = df[target]
    X = df.drop(columns=[target] + drop)

    non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print(f"[uyarı] Sayısal olmayan {len(non_numeric)} sütun düşürüldü: "
              f"{non_numeric}. Kullanmak istiyorsan önce sayısal kodla "
              f"(pd.get_dummies vb.).")
        X = X.select_dtypes(include=['number'])
    if X.shape[1] == 0:
        raise ValueError("Hiç sayısal özellik sütunu kalmadı.")
    return X, y


def _infer_task(y):
    """y'den görev tipini çıkarır: 'classification' veya 'regression'."""
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in ('U', 'S', 'O', 'b'):  # string/object/bool
        return 'classification'
    uniq = np.unique(y_arr[~np.isnan(y_arr.astype(float))]) \
        if y_arr.dtype.kind == 'f' else np.unique(y_arr)
    # Tam sayı değerli ve az sayıda sınıf → classification
    if len(uniq) <= 20 and np.allclose(uniq, np.round(uniq)):
        return 'classification'
    return 'regression'


def select_features(X, y, task='auto', algorithm='bga', method='fdb',
                    model=None, cv=5, scoring=None,
                    pop_size=20, max_iter=30, seed=42, n_jobs=1,
                    verbose=True, **algo_kwargs):
    """Herhangi bir tablosal veri setinde meta-sezgisel özellik seçimi.

    Veri kaynağı fark etmez: sağlık kayıtları, oyun telemetrisi, sensör
    verisi... X sayısal bir tablo, y bir hedef sütun olduğu sürece çalışır.
    scikit-learn gerektirir (yalnızca bu fonksiyon; çekirdek kütüphane değil).

    Parameters
    ----------
    X : pandas.DataFrame veya numpy.ndarray, shape (n_samples, n_features)
        Özellik matrisi. DataFrame verilirse sütun adları sonuçta korunur.
        NaN içeriyorsa medyan ile doldurulur (uyarı basılır).
    y : array-like, shape (n_samples,)
        Hedef değişken.
    task : {'auto', 'classification', 'regression'}
        'auto' → y'den çıkarılır (≤20 tam-sayı benzersiz değer → classification).
    algorithm : {'bga','bde','bpso','babc','bgwo','bcs','bsos'}
    method : {'classic', 'fdb', 'rfdb'}
    model : sklearn estimator or None
        None → RandomForest(Classifier/Regressor)(n_estimators=100).
        Kendi modelini verebilirsin (XGBoost, SVM...).
    cv : int, default=5
        K-fold sayısı (shuffle + seed ile deterministik).
    scoring : str or None
        sklearn scoring adı. None → classification: 'accuracy',
        regression: 'r2'. Fitness = 1 − ortalama skor (minimize edilir).
    pop_size, max_iter, seed : algoritma parametreleri.
    n_jobs : int
        Paralel fitness değerlendirme (yalnız destekleyen algoritma/modlarda).
    verbose : bool
        Bütçe özeti ve ilerleme bas.
    **algo_kwargs
        run_* fonksiyonuna aynen geçirilir (use_nsm=True, use_epm=True vb.).

    Returns
    -------
    dict:
        selected      : seçilen özellik adları (DataFrame ise) / indeksleri
        mask          : 0/1 numpy vektörü
        n_selected    : seçilen özellik sayısı
        best_score    : seçilen alt-kümenin ortalama CV skoru (1 − best_fit)
        task, scoring : kullanılan görev/metrik
        result        : algoritmanın ham çıktı dict'i (history dahil)

    Notes
    -----
    Toplam model eğitimi sayısı ≈ pop_size × (max_iter + 1) × cv.
    Büyük veri setlerinde önce küçük bir pilotla (pop_size=5, max_iter=3)
    süreyi ölç, sonra tam bütçeye geç.
    """
    try:
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError:
        raise ImportError(
            "select_features scikit-learn gerektirir: pip install scikit-learn"
        )

    method = _normalize_method(method)
    runners = {
        'bga': run_bga, 'bde': run_bde, 'bpso': run_bpso,
        'babc': run_babc, 'bgwo': run_bgwo, 'bcs': run_bcs, 'bsos': run_bsos,
    }
    if algorithm not in runners:
        raise ValueError(f"Bilinmeyen algorithm: '{algorithm}'. "
                         f"Geçerli: {sorted(runners)}")

    # Özellik adlarını yakala, numpy'a çevir
    feature_names = None
    if hasattr(X, 'columns'):
        feature_names = list(X.columns)
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError(f"X 2-boyutlu olmalı, verilen shape: {X_arr.shape}")
    if len(y_arr) != X_arr.shape[0]:
        raise ValueError(f"X ({X_arr.shape[0]} satır) ile y ({len(y_arr)}) uyumsuz.")

    # NaN → medyan doldurma (uyarıyla)
    if np.isnan(X_arr).any():
        n_nan = int(np.isnan(X_arr).sum())
        col_median = np.nanmedian(X_arr, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median)
        nan_r, nan_c = np.where(np.isnan(X_arr))
        X_arr[nan_r, nan_c] = col_median[nan_c]
        print(f"[uyarı] X'te {n_nan} NaN bulundu → sütun medyanı ile dolduruldu. "
              f"Daha iyi bir imputation istiyorsan veriyi önceden hazırla.")

    # Görev tipi
    if task == 'auto':
        task = _infer_task(y_arr)
        if verbose:
            print(f"[bilgi] task='auto' → '{task}' olarak belirlendi "
                  f"(y: {len(np.unique(y_arr))} benzersiz değer).")
    if task not in ('classification', 'regression'):
        raise ValueError(f"task 'auto'/'classification'/'regression' olmalı: {task}")

    # Model + metrik
    if model is None:
        model = (RandomForestClassifier(n_estimators=100, random_state=seed)
                 if task == 'classification'
                 else RandomForestRegressor(n_estimators=100, random_state=seed))
    if scoring is None:
        scoring = 'accuracy' if task == 'classification' else 'r2'

    if task == 'classification':
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)

    n_feat = X_arr.shape[1]
    est_fits = pop_size * (max_iter + 1) * cv
    if verbose:
        print(f"[bilgi] {n_feat} özellik, {X_arr.shape[0]} örnek | "
              f"{algorithm}+{method} | tahmini ~{est_fits} model eğitimi "
              f"(pop {pop_size} × iter {max_iter} × cv {cv}).")

    from sklearn.base import clone

    def fitness(mask):
        idx = np.flatnonzero(np.asarray(mask, dtype=int))
        if len(idx) == 0:
            return 2.0  # hiç özellik yok → en kötü (1 − skor üst sınırı üstü)
        scores = cross_val_score(clone(model), X_arr[:, idx], y_arr,
                                 cv=splitter, scoring=scoring)
        return 1.0 - float(np.mean(scores))

    enc = BinaryEncoding(n_features=n_feat)
    result = runners[algorithm](fitness, enc, method=method,
                                pop_size=pop_size, max_iter=max_iter,
                                seed=seed, n_jobs=n_jobs, verbose=verbose,
                                **algo_kwargs)

    mask = np.asarray(result['best_sol'], dtype=int)
    idx = np.flatnonzero(mask)
    selected = ([feature_names[i] for i in idx] if feature_names is not None
                else idx.tolist())
    out = {
        'selected': selected,
        'mask': mask,
        'n_selected': int(mask.sum()),
        'best_score': 1.0 - float(result['best_fit']),
        'task': task,
        'scoring': scoring,
        'result': result,
    }
    if verbose:
        print(f"[sonuç] {out['n_selected']}/{n_feat} özellik seçildi | "
              f"{scoring} = {out['best_score']:.4f}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Encoding sınıfları
    "Encoding",
    "BinaryEncoding",
    "ContinuousEncoding",
    "CategoricalEncoding",
    "MixedEncoding",
    "PermutationEncoding",
    # Single-objective algoritmalar
    "run_bga",
    "run_bde",
    "run_bpso",
    "run_babc",
    "run_bgwo",
    "run_bcs",
    "run_bsos",
    # Multi-objective algoritma
    "run_nsga2",
    # Constraint handling
    "make_constrained_fitness",
    "evaluate_constraints",
    # NSM & EPM modülleri
    "NSM_PRESETS",
    "nsm_decide",
    "sigmoid_switch",
    "dynamic_weights",
    "epm_population_update",
    "epm_nsm_population_update",  # v3-final-rc2
    "epm_guide_select",
    # Auto-tuning (v3-final-rc2)
    "tune_and_run",
    # Veri-odaklı hızlı başlangıç (v3.1)
    "select_features",
    "load_csv_dataset",
    # Utility
    "fdb_scores",
    "VALID_METHODS",
    "SHOW_METHOD_HINTS",
    # Versiyon
    "__version__", "__author__", "__license__",
]
