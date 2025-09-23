# ⚡ STRATÉGIE 4H SPRINT - CRITEO INTERVIEW

> **Mode Intensif : 240 minutes pour être prêt à 100%**
> **Chaque minute compte. Suivez ce plan à la lettre.**

---

## 🎯 VOTRE MISSION EN 4H

**Objectif:** Maîtriser l'essentiel pour passer l'entretien Criteo avec succès.

**Ce que vous DEVEZ savoir après 4h:**
1. ✅ Résoudre 2 problèmes DSA (hash/graph) en 40 min
2. ✅ Écrire 1 requête SQL complexe avec window functions
3. ✅ Expliquer CTR pipeline + métriques
4. ✅ Dessiner DeepKNN architecture
5. ✅ Comprendre first-price bidding
6. ✅ Pitcher en 60 secondes

---

## ⏱️ PLANNING MINUTE PAR MINUTE

### 🚀 PHASE 0: SETUP (10 min) - [09:00 - 09:10]

**Actions immédiates:**
```bash
□ Mode avion ON
□ Timer lancé (utiliser sprint_timer_4h.py)
□ Eau + snacks prêts
□ Ouvrir: START_HERE.md
□ IDE prêt avec les fichiers
```

**3 Objectifs mesurables pour aujourd'hui:**
1. _________________________________
2. _________________________________
3. _________________________________

---

### 💪 PHASE 1: DSA POWER HOUR (55 min) - [09:10 - 10:05]

#### 🎯 1A: Hash/Array Pattern (25 min)

**Problem 1: Two Sum All Pairs (12 min)**
```python
# TEMPLATE À COMPLÉTER
def two_sum_all_pairs(nums, target):
    """
    Criteo use case: Find all ad pairs summing to budget
    Time: O(n), Space: O(n)
    """
    seen = {}
    pairs = []

    # VOTRE CODE ICI (5 min)
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            for j in seen[complement]:
                pairs.append((j, i))

        if num not in seen:
            seen[num] = []
        seen[num].append(i)

    return pairs

# TEST (2 min)
test_cases = [
    ([100, 200, 100, 150], 300),  # Expected: [(0,1), (2,1)]
    ([1, 2, 3, 4, 5], 5),          # Expected: [(0,3), (1,2)]
]

for nums, target in test_cases:
    print(f"Input: {nums}, Target: {target}")
    print(f"Pairs: {two_sum_all_pairs(nums, target)}\n")
```

**Problem 2: Sliding Window Maximum (13 min)**
```python
# CONCEPT CLÉ: Monotonic deque pour CTR rolling window
from collections import deque

def sliding_window_max(nums, k):
    """
    Criteo: Max CTR in rolling time windows
    Time: O(n), Space: O(k)
    """
    dq = deque()  # stores indices
    result = []

    # MÉMORISER CE PATTERN
    for i, num in enumerate(nums):
        # Remove outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain decreasing
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

#### 🎯 1B: Graph BFS/DFS (30 min)

**Problem 3: User Network Clustering (15 min)**
```python
def find_user_clusters(edges):
    """
    Criteo: Find connected user groups for lookalike audiences
    Pattern: BFS for connected components
    """
    from collections import defaultdict, deque

    # Build graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # Find all clusters
    visited = set()
    clusters = []

    def bfs(start):
        cluster = []
        queue = deque([start])
        visited.add(start)

        while queue:
            user = queue.popleft()
            cluster.append(user)

            for neighbor in graph[user]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return cluster

    # Process all users
    all_users = set()
    for u, v in edges:
        all_users.add(u)
        all_users.add(v)

    for user in all_users:
        if user not in visited:
            cluster = bfs(user)
            clusters.append(cluster)

    return clusters

# TEST RAPIDE
edges = [(1,2), (2,3), (4,5), (5,6), (7,8)]
print(f"Clusters: {find_user_clusters(edges)}")
# Expected: [[1,2,3], [4,5,6], [7,8]]
```

**Problem 4: Campaign Dependencies (15 min)**
```python
def campaign_order(n, dependencies):
    """
    Criteo: Schedule campaigns with dependencies
    Pattern: Topological sort (Kahn's algorithm)
    """
    from collections import defaultdict, deque

    graph = defaultdict(list)
    in_degree = [0] * n

    for prereq, campaign in dependencies:
        graph[prereq].append(campaign)
        in_degree[campaign] += 1

    # Start with no dependencies
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []

    while queue:
        campaign = queue.popleft()
        result.append(campaign)

        for next_campaign in graph[campaign]:
            in_degree[next_campaign] -= 1
            if in_degree[next_campaign] == 0:
                queue.append(next_campaign)

    return result if len(result) == n else []  # [] if cycle

# TEST
deps = [(0,1), (0,2), (1,3), (2,3)]
print(f"Order: {campaign_order(4, deps)}")
# Expected: [0, 1, 2, 3] or [0, 2, 1, 3]
```

### ☕ BREAK (5 min) - [10:05 - 10:10]
- Stand up, stretch
- Hydrate
- Quick review what you learned

---

### 📊 PHASE 2: SQL ANALYTICS (35 min) - [10:10 - 10:45]

#### 🎯 2A: Master Pattern - Rolling Window CTR (15 min)

```sql
-- MÉMORISER CE TEMPLATE
WITH daily_metrics AS (
    SELECT
        date,
        campaign_id,
        SUM(clicks) / NULLIF(SUM(impressions), 0) as daily_ctr
    FROM events
    GROUP BY date, campaign_id
)
SELECT
    date,
    campaign_id,
    daily_ctr,

    -- Rolling 7-day average (KEY PATTERN)
    AVG(daily_ctr) OVER (
        PARTITION BY campaign_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as ctr_7d_avg,

    -- Week-over-week change (KEY PATTERN)
    daily_ctr - LAG(daily_ctr, 7) OVER (
        PARTITION BY campaign_id ORDER BY date
    ) as wow_change,

    -- Rank by performance (KEY PATTERN)
    DENSE_RANK() OVER (
        PARTITION BY date
        ORDER BY daily_ctr DESC
    ) as daily_rank

FROM daily_metrics;
```

#### 🎯 2B: Attribution Query (10 min)

```sql
-- PATTERN ESSENTIEL: Multi-touch attribution
WITH touchpoints AS (
    SELECT
        conversion_id,
        channel,
        conversion_value,
        ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY timestamp) as first_touch,
        ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY timestamp DESC) as last_touch,
        COUNT(*) OVER (PARTITION BY conversion_id) as total_touches
    FROM paths
)
SELECT
    channel,
    -- Last-click
    SUM(CASE WHEN last_touch = 1 THEN conversion_value END) as last_click_value,
    -- First-click
    SUM(CASE WHEN first_touch = 1 THEN conversion_value END) as first_click_value,
    -- Linear (equal credit)
    SUM(conversion_value / total_touches) as linear_value
FROM touchpoints
GROUP BY channel;
```

#### 🎯 2C: Quick Practice (10 min)

**Exercices rapides - Écrivez de tête:**
1. Calculer le rang d'un utilisateur par revenue
2. Trouver le top 10% des campagnes
3. Calculer la médiane avec PERCENTILE_CONT

---

### 🤖 PHASE 3: CTR & ML CORE (30 min) - [10:45 - 11:15]

#### 🎯 3A: Feature Engineering Essentials (10 min)

```python
# CRITEO FEATURES - MÉMORISER
"""
Dataset Criteo:
- 13 numerical features (num_1 to num_13)
- 26 categorical features (cat_1 to cat_26)
- 45M samples
- Target: click (0/1)
"""

# HASH TRICK - PATTERN CLÉ
def hash_trick(value, buckets=1_000_000):
    """Pour gérer high cardinality (milliards de valeurs uniques)"""
    import hashlib
    if pd.isna(value):
        return 0
    hash_val = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
    return hash_val % buckets

# FEATURE INTERACTIONS - TOP 3
def create_interactions(df):
    # 1. Categorical crosses (le plus important)
    df['cat_1_2'] = df['cat_1'].astype(str) + '_' + df['cat_2'].astype(str)

    # 2. Numerical products
    df['num_product'] = df['num_1'] * df['num_2']

    # 3. Statistical features
    df['num_mean'] = df[[f'num_{i}' for i in range(1,14)]].mean(axis=1)

    return df
```

#### 🎯 3B: Metrics & Evaluation (10 min)

```python
# MÉTRIQUES CRITEO - VALEURS CIBLES
"""
OBJECTIFS:
- LogLoss < 0.44 (primary metric)
- AUC > 0.80
- Calibration ratio ≈ 1.0
"""

from sklearn.metrics import log_loss, roc_auc_score

def evaluate_ctr_model(y_true, y_pred_proba):
    metrics = {
        'logloss': log_loss(y_true, y_pred_proba),      # < 0.44
        'auc': roc_auc_score(y_true, y_pred_proba),     # > 0.80
        'calibration': y_pred_proba.mean() / y_true.mean()  # ≈ 1.0
    }

    # LIFT @ 10% (business metric)
    threshold = np.percentile(y_pred_proba, 90)
    top_10_ctr = y_true[y_pred_proba >= threshold].mean()
    baseline_ctr = y_true.mean()
    metrics['lift_10'] = top_10_ctr / baseline_ctr  # Target > 2.0

    return metrics
```

#### 🎯 3C: Model Choice Quick Reference (10 min)

```python
# DECISION TREE - QUEL MODÈLE CHOISIR?
"""
Latency Budget < 20ms:
  → Logistic Regression (5ms)
  → LightGBM small (10ms)

Latency Budget 20-50ms:
  → LightGBM large (25ms)
  → XGBoost (30ms)

Latency Budget > 50ms:
  → Deep Learning (60ms+)
  → But use DeepKNN offline instead!

CRITEO CHOICE:
- Offline: DeepKNN embeddings (deep learning)
- Online: LightGBM ranking (fast, 15ms)
"""

# EXEMPLE LIGHTGBM OPTIMISÉ
import lightgbm as lgb

params_criteo = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 255,        # Power of 2 - 1
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,        # Regularization
    'lambda_l2': 0.1,
    'max_depth': 8,           # Prevent overfitting
    'min_data_in_leaf': 100   # For 45M samples
}
```

### ☕ BREAK (5 min) - [11:15 - 11:20]

---

### 🚀 PHASE 4: DEEPKNN & RETRIEVAL (35 min) - [11:20 - 11:55]

#### 🎯 4A: Architecture à Dessiner (10 min)

```
DEEPKNN ARCHITECTURE - DESSINEZ CECI:

     USER                           ITEM
       ↓                             ↓
  [Features]                    [Features]
       ↓                             ↓
  [Dense 256]                   [Dense 256]
       ↓                             ↓
  [Dense 128]                   [Dense 128]
       ↓                             ↓
  [Embedding]                   [Embedding]
       ↓                             ↓
  [L2 Normalize]                [L2 Normalize]
       ↓                             ↓
       └──────→ [Dot Product] ←──────┘
                      ↓
                 [Similarity]

OFFLINE:                        ONLINE:
- Generate embeddings          - User embedding (5ms)
- Index with Faiss            - kNN search (15ms)
- Update daily                - Re-rank (10ms)
                              Total: 30ms < 50ms ✓
```

#### 🎯 4B: Code Pattern Essentiel (15 min)

```python
# PATTERN TWO-TOWER - MÉMORISER
class TwoTowerModel:
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim

    def build_model(self):
        # User tower
        user_input = Input(shape=(50,))
        user_hidden = Dense(256, activation='relu')(user_input)
        user_hidden = Dropout(0.2)(user_hidden)
        user_embedding = Dense(self.embedding_dim)(user_hidden)
        user_embedding = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(user_embedding)

        # Item tower (similar)
        item_input = Input(shape=(30,))
        item_hidden = Dense(256, activation='relu')(item_input)
        item_hidden = Dropout(0.2)(item_hidden)
        item_embedding = Dense(self.embedding_dim)(item_hidden)
        item_embedding = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(item_embedding)

        # Similarity
        similarity = Dot(axes=1)([user_embedding, item_embedding])

        return Model(inputs=[user_input, item_input], outputs=similarity)

# FAISS INDEXING - PATTERN CLÉ
import faiss

def build_vector_index(embeddings, index_type='IVF'):
    dim = embeddings.shape[1]

    if index_type == 'Flat':
        index = faiss.IndexFlatIP(dim)  # Exact search
    elif index_type == 'IVF':
        nlist = int(np.sqrt(len(embeddings)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embeddings)  # Required for IVF

    faiss.normalize_L2(embeddings)  # For cosine similarity
    index.add(embeddings)

    return index

# SEARCH - TEMPS RÉEL
def search_top_k(index, query_embedding, k=100):
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return indices[0], distances[0]  # Top k items
```

#### 🎯 4C: Points Clés Interview (10 min)

**Questions fréquentes - Réponses rapides:**

1. **"Pourquoi two-tower?"**
   → "Séparation user/item permet caching et serving indépendant"

2. **"Comment gérer cold-start?"**
   → "Fallback sur features contextuelles + fast online learning"

3. **"Pourquoi Faiss?"**
   → "Optimisé SIMD, support GPU, scales to billions"

4. **"Trade-off latence/recall?"**
   → "IVF avec nprobe=10 donne 95% recall à 10ms"

---

### 💰 PHASE 5: BIDDING & AUCTIONS (20 min) - [11:55 - 12:15]

#### 🎯 5A: First-Price Essentials (10 min)

```python
# CONCEPT CLÉ: BID SHADING
"""
Sans shading: Bid = pCTR × pCVR × Value × (1 - Margin)
Problème: Surpaye de 30% en first-price!

Avec shading: Bid_final = Bid × Shading_Factor
Résultat: 15-20% économies (Criteo numbers)
"""

def calculate_shaded_bid(pCTR, pCVR, conversion_value, margin=0.3):
    # 1. Calculate max bid
    expected_value = pCTR * pCVR * conversion_value
    max_bid = expected_value * (1 - margin)

    # 2. Predict clearing price (ML model in practice)
    predicted_clearing = max_bid * 0.7  # Simplified

    # 3. Optimal shading
    shading_factor = 0.8  # Typically 0.75-0.85

    # 4. Final bid
    shaded_bid = max_bid * shading_factor

    # 5. Ensure above floor
    floor_price = 0.1  # Example
    final_bid = max(shaded_bid, floor_price * 1.01)

    return {
        'max_bid': max_bid,
        'shaded_bid': final_bid,
        'savings': max_bid - final_bid,
        'savings_pct': (max_bid - final_bid) / max_bid * 100
    }

# TEST
result = calculate_shaded_bid(
    pCTR=0.05,           # 5% CTR
    pCVR=0.02,           # 2% CVR
    conversion_value=50,  # $50 conversion
    margin=0.3           # 30% margin target
)
print(f"Max bid: ${result['max_bid']:.2f}")
print(f"Shaded bid: ${result['shaded_bid']:.2f}")
print(f"Savings: {result['savings_pct']:.1f}%")
```

#### 🎯 5B: Quick Reference (10 min)

```python
# TIMELINE À RETENIR
"""
2019: Industry moves to first-price
Why: Header bidding complexity + transparency

CRITEO APPROACH:
1. ML model predicts clearing price
2. Optimal shading via expected profit maximization
3. Results: 15-20% cost savings, same win rate
"""

# HEADER BIDDING - CONCEPT CLÉ
"""
Traditional: Sequential auctions
Header: Parallel auctions (all DSPs bid simultaneously)
Impact: Higher yield for publishers, need for smarter bidding
"""
```

---

### 🎯 PHASE 6: FAIRNESS & SYSTEM DESIGN (20 min) - [12:15 - 12:35]

#### 🎯 6A: Fairness Metrics (5 min)

```python
# FAIRJOB DATASET - MÉTRIQUES CRITEO
"""
Dataset: Job ads fairness
Metric: Demographic Parity (DP)
Target: < 0.7% deviation
Result: No performance loss!
"""

def demographic_parity(y_pred, sensitive_attribute):
    """
    DP = |P(Y=1|A=0) - P(Y=1|A=1)|
    Criteo target: < 0.007
    """
    group_0_rate = y_pred[sensitive_attribute == 0].mean()
    group_1_rate = y_pred[sensitive_attribute == 1].mean()

    dp = abs(group_0_rate - group_1_rate)

    return {
        'dp': dp,
        'passes': dp < 0.007,
        'group_0_rate': group_0_rate,
        'group_1_rate': group_1_rate
    }
```

#### 🎯 6B: System Design Template (15 min)

```
REAL-TIME CTR SYSTEM - DESSINEZ:

    Request
       ↓
  [Load Balancer]
       ↓
  [Feature Service] ←→ [Redis Cache]
       ↓
  [Prediction Service]
       ↓
    [Model A/B]
    /         \
[Model v1]  [Model v2]
       ↓
  [Ranking Service]
       ↓
  [Business Rules]
       ↓
    Response

SCALE NUMBERS:
- QPS: 1M+
- Latency: p99 < 100ms
- Models: ~100MB each
- Cache hit rate: > 90%

BOTTLENECKS:
1. Feature retrieval → Cache
2. Model inference → Batching
3. Network I/O → Connection pooling
```

---

### 🏁 PHASE 7: PITCH & INTEGRATION (20 min) - [12:35 - 12:55]

#### 🎯 7A: Pitch 60 Secondes (5 min)

**Apprenez par cœur:**

"Je suis [Nom], ingénieur ML spécialisé en systèmes publicitaires temps réel.

J'ai récemment optimisé un pipeline CTR réduisant la latence p99 de 200ms à 45ms tout en maintenant 0.82 AUC, en utilisant une approche similaire à votre DeepKNN.

Expert en first-price bidding, j'ai implémenté un système de bid shading ML-based générant 18% d'économies.

Ce qui m'attire chez Criteo : votre leadership technique - DeepKNN en production, FairJob pour l'éthique, et l'innovation continue.

J'ai étudié vos publications récentes et je suis enthousiaste à l'idée de contribuer à ces défis à grande échelle."

**Chronométrez: Doit faire < 60 secondes!**

#### 🎯 7B: Top 5 Questions à Poser (5 min)

1. "Comment gérez-vous les embeddings updates dans DeepKNN sans downtime?"
2. "Votre bid shading est-il global ou par-publisher?"
3. "Quel trade-off acceptez-vous entre fairness et business metrics?"
4. "Post-cookies, quelle est la stratégie beyond contextual?"
5. "Opportunités de publier en conférences pour l'équipe?"

#### 🎯 7C: Speed Review Final (10 min)

**Checklist - Pouvez-vous:**
- [ ] Écrire two-pointer pattern de mémoire?
- [ ] SQL window function avec PARTITION BY?
- [ ] Expliquer hash trick en 30 secondes?
- [ ] Dessiner two-tower architecture?
- [ ] Calculer bid shading savings?
- [ ] Définir Demographic Parity?
- [ ] Pitcher en < 60 secondes?

---

### ✅ PHASE 8: CLOSING (5 min) - [12:55 - 13:00]

**Actions finales:**
1. Screenshot vos notes
2. Réviser les weak points identifiés
3. Tester votre setup technique
4. Respirer profondément
5. Vous êtes PRÊT!

---

## 📊 TRACKING CARD - À REMPLIR

```
START TIME: _______
END TIME: _______

PROBLEMS SOLVED:
□ Two Sum Pairs
□ Sliding Window
□ User Clustering
□ Topological Sort
□ SQL Rolling Window
□ SQL Attribution

CONCEPTS CLEAR:
□ Hash Trick
□ CTR Metrics
□ Two-Tower Architecture
□ Faiss Indexing
□ Bid Shading
□ Demographic Parity

PITCH PRACTICED:
□ < 60 seconds
□ Smooth delivery
□ Key points covered

CONFIDENCE LEVEL: ___/10
```

---

## 🚨 CHEAT SHEET - GARDEZ SOUS LA MAIN

### Numbers to Remember:
- **Dataset:** 45M samples, 13 num + 26 cat
- **Targets:** LogLoss < 0.44, AUC > 0.80
- **Latency:** < 50ms p99 (DeepKNN)
- **Savings:** 15-20% (bid shading)
- **Fairness:** DP < 0.7%
- **Scale:** 1M+ QPS

### Code Patterns:
```python
# Two-pointer
left, right = 0, len(arr) - 1

# Window function
OVER (PARTITION BY x ORDER BY y ROWS BETWEEN...)

# Hash trick
hash(value) % buckets

# Two-tower
normalize(user_emb) · normalize(item_emb)

# Bid shading
bid * 0.8 (typical factor)
```

### One-Liners:
- "DeepKNN = sophistication offline, simplicité online"
- "First-price needs intelligence to not overpay"
- "Fairness improves generalization"

---

## 💪 YOU'VE GOT THIS!

**4 heures intenses, mais vous serez prêt.**

**Suivez le plan. Trust the process. Ace the interview.**

---

*Stratégie 4H créée pour maximiser votre préparation Criteo*

*GO CRUSH IT! 🚀*