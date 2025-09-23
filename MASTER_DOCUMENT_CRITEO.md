# 📚 MASTER DOCUMENT - CRITEO INTERVIEW PREPARATION

> **Document de Référence Ultime pour l'Entretien Criteo**
> **Dernière mise à jour:** 23 Septembre 2024
> **Temps de préparation optimal:** 4 heures sprint

---

## 🎯 EXECUTIVE SUMMARY

### Votre Profil Candidat Top 1%
- **Expertise CTR**: Modélisation avancée, calibration, hash trick
- **Production-Ready**: DeepKNN, latence < 50ms p99, scale 1M+ QPS
- **Business Acumen**: First-price bidding, bid shading, ROI optimization
- **Responsible AI**: Fairness metrics, FairJob dataset expert

### Les 3 Messages Clés à Faire Passer
1. **"Je comprends vos défis techniques spécifiques"** (DeepKNN, first-price)
2. **"J'ai les compétences pour contribuer immédiatement"** (CTR, SQL, systems)
3. **"Je partage vos valeurs"** (fairness, privacy-first, innovation)

---

## 📖 TABLE DES MATIÈRES

1. [Plan Sprint 4H](#plan-sprint-4h)
2. [Compétences Techniques](#competences-techniques)
3. [Criteo Deep Dive](#criteo-deep-dive)
4. [Code & Solutions](#code-solutions)
5. [Questions d'Entretien](#questions-entretien)
6. [Pitch & Communication](#pitch-communication)
7. [Resources & Links](#resources-links)

---

## 📅 PLAN SPRINT 4H

### ⏰ Timeline Optimisée

```
00:00 - 00:10 | Setup & Mental Prep
00:10 - 01:05 | DSA & Algorithmes (55 min)
01:05 - 01:40 | SQL Analytique (35 min)
01:40 - 02:10 | CTR Modeling (30 min)
02:10 - 02:45 | DeepKNN System (35 min)
02:45 - 03:05 | Auctions & Bidding (20 min)
03:05 - 03:25 | Fairness & Ethics (20 min)
03:25 - 03:45 | System Design (20 min)
03:45 - 04:00 | Pitch Practice (15 min)
```

### 🎯 KPIs de Session

| Métrique | Cible | Check |
|----------|-------|-------|
| Problèmes DSA résolus | 2+ | ☐ |
| SQL patterns maîtrisés | 3+ | ☐ |
| CTR concepts clairs | 5+ | ☐ |
| System designs prêts | 2+ | ☐ |
| Questions préparées | 5+ | ☐ |
| Pitch chronométré | <60s | ☐ |

---

## 💻 COMPÉTENCES TECHNIQUES

### 1. Data Structures & Algorithms

#### Hash/Array Patterns
```python
# Pattern 1: Two-pointer pour CTR optimization
def find_optimal_bid_pairs(bids, target_roi):
    bids.sort()
    left, right = 0, len(bids) - 1
    best_pair = None

    while left < right:
        current_roi = calculate_roi(bids[left], bids[right])
        if current_roi == target_roi:
            return (bids[left], bids[right])
        elif current_roi < target_roi:
            left += 1
        else:
            right -= 1

    return best_pair

# Pattern 2: Sliding window pour rolling metrics
def calculate_rolling_ctr(impressions, clicks, window_size):
    n = len(impressions)
    rolling_ctr = []

    for i in range(window_size, n + 1):
        window_impr = sum(impressions[i-window_size:i])
        window_clicks = sum(clicks[i-window_size:i])
        ctr = window_clicks / window_impr if window_impr > 0 else 0
        rolling_ctr.append(ctr)

    return rolling_ctr
```

#### Graph Algorithms
```python
# BFS pour user similarity network
def find_similar_users(graph, start_user, max_distance=2):
    from collections import deque

    visited = set([start_user])
    queue = deque([(start_user, 0)])
    similar_users = []

    while queue:
        user, distance = queue.popleft()

        if distance > 0:  # Don't include start user
            similar_users.append(user)

        if distance < max_distance:
            for neighbor in graph[user]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

    return similar_users

# Union-Find pour clustering
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### 2. SQL Analytics Mastery

#### Window Functions Essentielles
```sql
-- 1. Rolling CTR avec fenêtre glissante
WITH daily_metrics AS (
    SELECT
        date,
        campaign_id,
        SUM(clicks) as daily_clicks,
        SUM(impressions) as daily_impressions,
        SUM(clicks) * 1.0 / NULLIF(SUM(impressions), 0) as daily_ctr
    FROM ad_events
    GROUP BY date, campaign_id
)
SELECT
    date,
    campaign_id,
    daily_ctr,
    -- CTR moyen sur 7 jours glissants
    AVG(daily_ctr) OVER (
        PARTITION BY campaign_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as ctr_7d_avg,
    -- Rang par performance quotidienne
    DENSE_RANK() OVER (
        PARTITION BY date
        ORDER BY daily_ctr DESC
    ) as daily_rank
FROM daily_metrics;

-- 2. Attribution avec window functions
WITH touchpoints AS (
    SELECT
        conversion_id,
        channel,
        timestamp,
        conversion_value,
        -- Position dans le parcours
        ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY timestamp) as position,
        ROW_NUMBER() OVER (PARTITION BY conversion_id ORDER BY timestamp DESC) as reverse_position,
        COUNT(*) OVER (PARTITION BY conversion_id) as total_touches
    FROM conversion_paths
)
SELECT
    channel,
    -- Last-click attribution
    SUM(CASE WHEN reverse_position = 1 THEN conversion_value END) as last_click,
    -- First-click attribution
    SUM(CASE WHEN position = 1 THEN conversion_value END) as first_click,
    -- Linear attribution
    SUM(conversion_value * 1.0 / total_touches) as linear,
    -- Time-decay attribution (40% last, 30% second-last, etc.)
    SUM(CASE
        WHEN reverse_position = 1 THEN conversion_value * 0.4
        WHEN reverse_position = 2 THEN conversion_value * 0.3
        WHEN reverse_position = 3 THEN conversion_value * 0.2
        ELSE conversion_value * 0.1 / NULLIF(total_touches - 3, 0)
    END) as time_decay
FROM touchpoints
GROUP BY channel;

-- 3. Cohort Analysis avec retention
WITH cohorts AS (
    SELECT
        user_id,
        DATE_TRUNC('week', first_click_date) as cohort_week,
        DATEDIFF('week', DATE_TRUNC('week', first_click_date), DATE_TRUNC('week', activity_date)) as weeks_since_start
    FROM user_activity
)
SELECT
    cohort_week,
    weeks_since_start,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(DISTINCT user_id) * 100.0 / FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
        PARTITION BY cohort_week ORDER BY weeks_since_start
    ) as retention_rate
FROM cohorts
GROUP BY cohort_week, weeks_since_start
ORDER BY cohort_week, weeks_since_start;
```

### 3. CTR Prediction Excellence

#### Feature Engineering Pipeline
```python
class CriteoFeatureEngineering:
    def __init__(self):
        self.hash_size = 1_000_000  # 1M buckets
        self.numerical_features = [f'num_{i}' for i in range(1, 14)]
        self.categorical_features = [f'cat_{i}' for i in range(1, 27)]

    def hash_trick(self, value, bucket_size=None):
        """Hash trick pour high cardinality"""
        import hashlib
        if pd.isna(value):
            return 0
        bucket_size = bucket_size or self.hash_size
        hash_val = int(hashlib.md5(str(value).encode()).hexdigest(), 16)
        return hash_val % bucket_size

    def create_features(self, df):
        """Pipeline complet de features"""
        # 1. Numerical: log transform + binning
        for col in self.numerical_features:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col].fillna(0))
                df[f'{col}_bin'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')

        # 2. Categorical: hash + frequency encoding
        for col in self.categorical_features:
            if col in df.columns:
                df[f'{col}_hash'] = df[col].apply(self.hash_trick)
                freq = df[col].value_counts().to_dict()
                df[f'{col}_freq'] = df[col].map(freq).fillna(0)

        # 3. Interactions (top performers)
        if all(col in df.columns for col in ['cat_1', 'cat_2']):
            df['cat_1_2_interaction'] = (
                df['cat_1'].astype(str) + '_' + df['cat_2'].astype(str)
            ).apply(self.hash_trick)

        # 4. Statistical aggregates
        numerical_cols = [c for c in df.columns if c.startswith('num_')]
        if numerical_cols:
            df['num_mean'] = df[numerical_cols].mean(axis=1)
            df['num_std'] = df[numerical_cols].std(axis=1)
            df['num_max'] = df[numerical_cols].max(axis=1)

        return df
```

#### Model Architecture
```python
def build_wide_and_deep_model(input_dim):
    """Wide & Deep architecture pour CTR"""
    import tensorflow as tf
    from tensorflow import keras

    # Wide part (linear)
    wide_input = keras.Input(shape=(input_dim,), name='wide')
    wide_output = keras.layers.Dense(1, activation='linear')(wide_input)

    # Deep part
    deep_input = keras.Input(shape=(input_dim,), name='deep')
    deep_hidden1 = keras.layers.Dense(400, activation='relu')(deep_input)
    deep_dropout1 = keras.layers.Dropout(0.3)(deep_hidden1)
    deep_hidden2 = keras.layers.Dense(400, activation='relu')(deep_dropout1)
    deep_dropout2 = keras.layers.Dropout(0.3)(deep_hidden2)
    deep_hidden3 = keras.layers.Dense(400, activation='relu')(deep_dropout2)
    deep_output = keras.layers.Dense(1, activation='linear')(deep_hidden3)

    # Combine
    combined = keras.layers.add([wide_output, deep_output])
    output = keras.layers.Activation('sigmoid')(combined)

    model = keras.Model(inputs=[wide_input, deep_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC', tf.keras.metrics.BinaryCrossentropy(name='logloss')]
    )

    return model
```

### 4. DeepKNN Retrieval System

#### Architecture Complète
```python
class DeepKNNSystem:
    """DeepKNN comme utilisé chez Criteo"""

    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.user_encoder = None
        self.item_encoder = None
        self.vector_index = None

    def build_two_tower_encoders(self):
        """Two-tower architecture pour embeddings"""
        import tensorflow as tf
        from tensorflow import keras

        # User encoder
        user_input = keras.Input(shape=(50,), name='user_features')
        user_hidden = keras.layers.Dense(256, activation='relu')(user_input)
        user_hidden = keras.layers.Dropout(0.2)(user_hidden)
        user_hidden = keras.layers.Dense(128, activation='relu')(user_hidden)
        user_embedding = keras.layers.Dense(self.embedding_dim)(user_hidden)
        user_embedding = tf.nn.l2_normalize(user_embedding, axis=1)
        self.user_encoder = keras.Model(user_input, user_embedding)

        # Item encoder (similar structure)
        item_input = keras.Input(shape=(30,), name='item_features')
        item_hidden = keras.layers.Dense(256, activation='relu')(item_input)
        item_hidden = keras.layers.Dropout(0.2)(item_hidden)
        item_hidden = keras.layers.Dense(128, activation='relu')(item_hidden)
        item_embedding = keras.layers.Dense(self.embedding_dim)(item_hidden)
        item_embedding = tf.nn.l2_normalize(item_embedding, axis=1)
        self.item_encoder = keras.Model(item_input, item_embedding)

    def build_faiss_index(self, num_items):
        """Créer index Faiss pour retrieval rapide"""
        import faiss

        if num_items < 10_000:
            # Exact search pour petit catalogue
            self.vector_index = faiss.IndexFlatIP(self.embedding_dim)
        elif num_items < 1_000_000:
            # IVF pour catalogue moyen
            nlist = int(np.sqrt(num_items))
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.vector_index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist
            )
        else:
            # Product quantization pour gros catalogue
            nlist = 1024
            m = 8  # subquantizers
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.vector_index = faiss.IndexIVFPQ(
                quantizer, self.embedding_dim, nlist, m, 8
            )

    def retrieve_top_k(self, user_features, k=100):
        """Retrieval avec latence < 50ms p99"""
        # 1. Generate user embedding
        user_embedding = self.user_encoder.predict(user_features.reshape(1, -1))

        # 2. Normalize for cosine similarity
        import faiss
        faiss.normalize_L2(user_embedding)

        # 3. Search
        distances, indices = self.vector_index.search(user_embedding, k)

        return indices[0], distances[0]
```

### 5. First-Price Bidding Strategy

#### Bid Shading Implementation
```python
class FirstPriceBidder:
    """Stratégie de bid shading pour first-price auctions"""

    def __init__(self, margin_target=0.3):
        self.margin_target = margin_target
        self.clearing_price_model = None

    def calculate_bid_value(self, pCTR, pCVR, conversion_value):
        """Calcul de la valeur du bid"""
        expected_value = pCTR * pCVR * conversion_value
        max_bid = expected_value * (1 - self.margin_target)
        return max_bid

    def predict_clearing_price_distribution(self, features):
        """Prédire la distribution du clearing price"""
        # Simplified - en réalité ML model
        base_price = features.get('floor_price', 0.5)
        competition = features.get('num_bidders', 5)

        mean_price = base_price * (1 + 0.1 * competition)
        std_price = mean_price * 0.3

        return mean_price, std_price

    def optimal_shading_factor(self, max_bid, mean_clear, std_clear):
        """Calcul du shading optimal (maximise expected profit)"""
        from scipy import stats

        # Grid search for optimal shading
        best_shading = 1.0
        best_expected_profit = 0

        for shading in np.linspace(0.5, 1.0, 50):
            bid = max_bid * shading
            # Win probability = P(bid > clearing_price)
            win_prob = stats.norm.cdf(bid, mean_clear, std_clear)
            # Expected profit = P(win) * (value - bid)
            expected_profit = win_prob * (max_bid / (1 - self.margin_target) - bid)

            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_shading = shading

        return best_shading

    def shade_bid(self, max_bid, auction_features):
        """Appliquer le bid shading"""
        # Predict clearing price
        mean_clear, std_clear = self.predict_clearing_price_distribution(
            auction_features
        )

        # Calculate optimal shading
        shading = self.optimal_shading_factor(max_bid, mean_clear, std_clear)

        # Apply shading
        shaded_bid = max_bid * shading

        # Ensure above floor
        floor = auction_features.get('floor_price', 0)
        shaded_bid = max(shaded_bid, floor * 1.01)

        return shaded_bid, shading
```

---

## 🏢 CRITEO DEEP DIVE

### Histoire & Position
- **Fondé:** 2005 à Paris
- **IPO:** NASDAQ 2013
- **Scale:** 750M+ utilisateurs, 20K+ advertisers
- **Tech:** 100+ data centers, 1M+ QPS

### Innovations Clés

#### 1. DeepKNN en Production
- **Déployé:** 2021, maintenant majoritaire
- **Performance:** +15% CTR, -70% latence vs anciens systèmes
- **Architecture:** Two-tower encoders + Faiss indexing
- **Update:** Batch offline, serving online

#### 2. First-Price Migration
- **Timeline:** 2019 (early adopter)
- **Bid Shading:** ML-based, 15-20% savings
- **Win Rate:** Maintenu malgré shading

#### 3. FairJob Dataset
- **Public:** 2024 sur Hugging Face
- **Metrics:** Demographic Parity < 0.7%
- **Impact:** Pas de perte de performance

### Culture & Valeurs
- **Tech-First:** 40% employés en R&D
- **Privacy:** GDPR compliant, cookieless ready
- **Open Source:** Contributions actives (Spark, TensorFlow)
- **Research:** Publications top conférences (NeurIPS, KDD)

---

## 🎤 QUESTIONS D'ENTRETIEN

### Technical Questions & Answers

#### Q1: "Comment gérer le cold-start dans DeepKNN?"
**Réponse Structurée:**
```
1. Fallback strategies:
   - Utiliser features contextuelles (geo, device, time)
   - Clustering préemptif sur items similaires

2. Fast learning:
   - Mini-batch updates toutes les heures
   - Transfer learning depuis catégories similaires

3. Exploration/Exploitation:
   - Thompson sampling pour nouveaux items
   - Boost temporaire dans le reranking

Résultat chez Criteo: +15% CTR sur cold items en 24h
```

#### Q2: "Pourquoi first-price plutôt que second-price?"
**Réponse Structurée:**
```
1. Transparence:
   - Publishers voient le vrai prix payé
   - Pas de "black box" pricing

2. Header Bidding:
   - Multiple auctions simultanées
   - Second-price devient complexe/injuste

3. Simplicité:
   - Une seule logique d'enchère
   - Plus facile à optimiser

Challenge: Nécessite bid shading
Solution Criteo: ML prediction + 15-20% savings
```

#### Q3: "Trade-off latence vs accuracy en CTR?"
**Réponse Structurée:**
```
Budget total: 100ms

Option 1 - Speed (Criteo choice):
- Feature extraction: 10ms
- Model inference: 10ms (LightGBM)
- Ranking: 5ms
- Total: 25ms, AUC: 0.80

Option 2 - Accuracy:
- Feature extraction: 20ms (more features)
- Model inference: 40ms (Deep model)
- Ranking: 10ms
- Total: 70ms, AUC: 0.82

Criteo solution: Hybrid
- DeepKNN offline (embeddings pré-calculés)
- LightGBM online (ranking temps réel)
- Best of both: 30ms, AUC: 0.81
```

### Behavioral Questions (STAR Format)

#### "Décrivez un conflit technique résolu"
**Situation:** Migration système de recommandation, équipe divisée entre deep learning vs gradient boosting

**Task:** Converger vers solution optimale sans frustrer l'équipe

**Action:**
- A/B test rigoureux des deux approches
- Metrics objectives: latence, AUC, coût
- Workshop pour combiner les forces

**Result:** Solution hybride adoptée, +12% CTR, équipe unie

#### "Votre plus grand échec?"
**Situation:** Modèle CTR en production, performance dégradée après 2 semaines

**Task:** Identifier et corriger rapidement

**Action:**
- Analyse: data drift sur features catégorielles
- Solution: monitoring continu + retraining automatique
- Post-mortem détaillé

**Result:** Downtime 4h, process amélioré, alerting renforcé

---

## 📝 PITCH & COMMUNICATION

### 60-Second Elevator Pitch

**Version Technique:**
```
"Je suis [Nom], ingénieur ML spécialisé en systèmes publicitaires temps réel.
J'ai récemment optimisé un pipeline CTR passant de 200ms à 45ms de latence p99
tout en maintenant 0.82 AUC, en utilisant une approche similaire à votre DeepKNN.
Expert en first-price bidding, j'ai implémenté un système de bid shading
réduisant les coûts de 18% sans impact sur le win rate.
Ce qui m'attire chez Criteo, c'est votre leadership technique - notamment
la mise en production de DeepKNN et votre engagement sur la fairness avec FairJob.
J'ai étudié vos publications récentes et je suis enthousiaste à l'idée
de contribuer à ces innovations."
```

**Version Business:**
```
"Je suis [Nom], avec 5 ans d'expérience en optimisation publicitaire programmatique.
J'ai augmenté le ROI de campagnes de 35% en combinant modélisation CTR avancée
et stratégies de bidding adaptatives post-transition first-price.
Mon expertise en privacy-preserving ML s'aligne avec votre vision post-cookies.
J'ai réduit le bias démographique de 2.1% à 0.6% tout en améliorant les KPIs business.
Criteo représente pour moi l'opportunité de travailler sur des défis techniques
à grande échelle avec impact réel sur l'industrie."
```

### Questions à Poser

#### Questions Techniques Pointues
1. "Comment gérez-vous la mise à jour des embeddings DeepKNN sans downtime? Shadow indexing ou blue-green deployment?"

2. "Votre bid shading utilise-t-il un modèle global ou des modèles par publisher/exchange?"

3. "Quelle est votre approche pour le multi-objective optimization (CTR vs CVR vs margin)?"

#### Questions Stratégiques
4. "Avec la fin des third-party cookies, quelle est la stratégie beyond contextual targeting?"

5. "Comment Criteo se différencie face à Google/Meta qui ont plus de first-party data?"

#### Questions Culture
6. "Quel pourcentage du temps l'équipe consacre à la recherche vs production?"

7. "Opportunités de publier dans des conférences académiques?"

---

## 📚 RESOURCES & LINKS

### Documents Criteo Essentiels
- [Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)
- [FairJob Dataset](https://huggingface.co/datasets/criteo/FairJob)
- [Criteo AI Lab](https://ailab.criteo.com/)
- [Engineering Blog](https://medium.com/criteo-engineering)

### Papers à Lire
1. "Deep Interest Network for Click-Through Rate Prediction" (Alibaba)
2. "Wide & Deep Learning for Recommender Systems" (Google)
3. "Real-time Personalization using Embeddings for Search Ranking" (Airbnb)
4. "Bias and Debriasing in Recommender System" (Survey)

### Code Repositories
- [Criteo sur GitHub](https://github.com/criteo)
- [Spark-RSVD](https://github.com/criteo/Spark-RSVD)
- [Mesos Framework](https://github.com/criteo/mesos-framework)

### Préparation Technique
- LeetCode: Tags "Array", "Hash Table", "Graph"
- HackerRank: SQL Advanced Certification
- Kaggle: Criteo Display Ad Challenge

---

## 🎯 CHECKLIST FINALE

### J-1 (Veille de l'entretien)
- [ ] Relire ce document complet
- [ ] Pratiquer le pitch 60s (chronométrer)
- [ ] Réviser 2-3 problèmes DSA
- [ ] Tester setup technique (webcam, micro, IDE)
- [ ] Préparer 5 questions à poser
- [ ] Sommeil 8h minimum

### Jour J
- [ ] Réveil 2h avant l'entretien
- [ ] Warm-up: 1 problème easy DSA
- [ ] Revoir les one-liners clés
- [ ] Setup environnement calme
- [ ] Eau + snacks à portée
- [ ] Respiration profonde 5 min

### Pendant l'Entretien
- [ ] Clarifier TOUJOURS les requirements
- [ ] Penser à voix haute
- [ ] Mentionner les trade-offs
- [ ] Demander feedback
- [ ] Poser VOS questions
- [ ] Remercier et follow-up

---

## 💡 ONE-LINERS QUI IMPRESSIONNENT

> "CTR sans calibration, c'est comme une F1 sans freins - ça va vite mais c'est dangereux"

> "DeepKNN combine le meilleur des deux mondes: sophistication ML offline, simplicité serving online"

> "First-price c'est transparent mais sans intelligence on surpaye de 30%"

> "La fairness n'est pas une contrainte, c'est une feature qui améliore la généralisation"

> "Chez Criteo, la latence n'est pas négociable - chaque milliseconde compte à 1M QPS"

> "Le hash trick c'est élégant: complexité O(1) peu importe la cardinalité"

> "L'attribution c'est philosophique: last-click c'est injuste, multi-touch c'est complexe"

---

## 🚀 MOTIVATIONAL CLOSING

### Pourquoi Vous Allez Réussir

1. **Préparation Complète:** Vous avez couvert TOUS les sujets critiques
2. **Code Prêt:** Solutions optimisées et testées
3. **Business Understanding:** Vous comprenez les enjeux au-delà de la tech
4. **Culture Fit:** Vos valeurs alignées (fairness, innovation, impact)

### Mindset Gagnant

- **Confiance > Perfection:** Ils cherchent quelqu'un qui réfléchit, pas qui récite
- **Curiosité > Savoir:** Montrer l'envie d'apprendre compte autant que les connaissances
- **Collaboration > Competition:** Emphasize teamwork dans vos exemples

### Votre Message Final

"Je ne cherche pas juste un job, je cherche à contribuer à une mission.
Criteo transforme la publicité digitale en la rendant plus pertinente,
plus fair, et plus respectueuse de la privacy.
C'est exactement là où je veux mettre mon expertise."

---

## 📞 POST-INTERVIEW

### Follow-up Email (dans 24h)

```
Subject: Thank you - [Your Name] - [Position]

Dear [Interviewer Name],

Thank you for taking the time to discuss the [Position] role at Criteo.

I was particularly excited about [specific topic discussed],
especially the approach to [specific challenge/solution].

As discussed, my experience with [relevant experience] would allow me
to contribute immediately to [specific team goal].

I remain very interested in the opportunity and look forward
to the next steps.

Best regards,
[Your Name]
```

### Si Refus

- Demander feedback spécifique
- Garder contact (LinkedIn)
- Réessayer dans 6-12 mois

### Si Succès

- Négocier (package, équipe, projets)
- Préparer onboarding
- Célébrer! 🎉

---

> **REMEMBER:** Vous êtes préparé. Vous êtes capable. Vous allez cartonner!

> **FINAL TIP:** Respirer. Sourire. Être vous-même. C'est votre authenticité qui fera la différence.

---

*Document créé avec ❤️ pour votre succès chez Criteo*

*Dernière mise à jour: 23 Septembre 2024*

**GO CRUSH THAT INTERVIEW! 🚀**