# 📅 GUIDE D'ÉTUDE 7 JOURS - CRITEO INTERVIEW

> Plan d'étude structuré pour maîtriser tous les aspects de l'entretien Criteo
> **Temps total:** 7 jours × 4 heures = 28 heures
> **Format:** Sessions quotidiennes de 4h avec pauses

---

## 🎯 OBJECTIFS PAR DOMAINE

| Domaine | Heures | Priorité | Objectif Final |
|---------|--------|----------|----------------|
| DSA & Coding | 8h | ⭐⭐⭐⭐⭐ | 20+ problèmes, patterns maîtrisés |
| SQL Analytics | 4h | ⭐⭐⭐⭐ | 10+ requêtes complexes |
| ML/CTR Theory | 4h | ⭐⭐⭐⭐⭐ | Concepts clairs, code prêt |
| System Design | 4h | ⭐⭐⭐⭐ | 3 designs complets |
| Criteo Specific | 4h | ⭐⭐⭐⭐⭐ | DeepKNN, FairJob, First-price |
| Behavioral | 2h | ⭐⭐⭐ | 5 stories STAR |
| Mock Practice | 2h | ⭐⭐⭐⭐ | 2 sessions complètes |

---

## 📆 JOUR 1 - FONDAMENTAUX DSA

### 🕐 09:00-13:00 - Data Structures & Algorithms Basics

#### Session 1: Arrays & Hashing (2h)
```python
# Problems à résoudre:
1. Two Sum (avec toutes variantes)
2. Sliding Window Maximum
3. Subarray Sum Equals K
4. Top K Frequent Elements
5. LRU Cache

# Pattern à maîtriser:
- Two pointers
- Sliding window
- Hash map optimizations
- Prefix sums
```

**Exercices Pratiques:**
- [ ] LeetCode #1 - Two Sum (15 min)
- [ ] LeetCode #239 - Sliding Window Maximum (20 min)
- [ ] LeetCode #560 - Subarray Sum Equals K (20 min)
- [ ] LeetCode #347 - Top K Frequent Elements (20 min)
- [ ] LeetCode #146 - LRU Cache (25 min)

#### Session 2: Graphs Basics (2h)
```python
# Problems à résoudre:
1. BFS/DFS traversal
2. Connected Components
3. Cycle Detection
4. Shortest Path (Dijkstra)
5. Topological Sort

# Criteo contexts:
- User similarity networks
- Campaign dependencies
- Attribution paths
```

**Exercices Pratiques:**
- [ ] LeetCode #200 - Number of Islands (20 min)
- [ ] LeetCode #207 - Course Schedule (25 min)
- [ ] LeetCode #743 - Network Delay Time (25 min)
- [ ] LeetCode #399 - Evaluate Division (30 min)

### 📊 Métriques Jour 1
- Problems solved: ___/10
- Patterns understood: ___/5
- Time complexity mastered: ⬜ Yes ⬜ No
- Space complexity clear: ⬜ Yes ⬜ No

---

## 📆 JOUR 2 - SQL MASTERY & ADVANCED DSA

### 🕐 09:00-11:00 - SQL Analytics Deep Dive

#### Window Functions Mastery
```sql
-- Exercices progressifs:
1. Running totals & averages
2. Ranking (ROW_NUMBER, RANK, DENSE_RANK)
3. Lead/Lag comparisons
4. Percentile calculations
5. Moving windows (ROWS vs RANGE)
```

**Pratique sur données Criteo-like:**
- [ ] Calculer CTR rolling 7 jours
- [ ] Attribution multi-touch
- [ ] Cohort analysis avec retention
- [ ] A/B test significance
- [ ] Budget pacing queries

### 🕐 11:00-13:00 - Advanced DSA Patterns

#### Patterns Avancés
```python
# Focus sur problèmes Criteo-relevant:
1. Union-Find (user clustering)
2. Segment Trees (range queries)
3. Trie (autocomplete)
4. Binary Search variations
5. Dynamic Programming basics
```

**Exercices:**
- [ ] LeetCode #547 - Friend Circles (Union-Find)
- [ ] LeetCode #307 - Range Sum Query (Segment Tree)
- [ ] LeetCode #208 - Implement Trie
- [ ] LeetCode #300 - Longest Increasing Subsequence (DP)

### 📊 Métriques Jour 2
- SQL patterns: ___/5
- Advanced DSA: ___/4
- Optimization techniques: ___/3

---

## 📆 JOUR 3 - CTR MODELING & ML

### 🕐 09:00-13:00 - CTR Prediction Deep Dive

#### Session 1: Feature Engineering (2h)
```python
# Implémenter from scratch:
class CriteoFeatureProcessor:
    def __init__(self):
        # 13 numerical + 26 categorical
        pass

    def hash_trick(self, value, buckets=1000000):
        # Implement hashing for high cardinality
        pass

    def create_interactions(self, features):
        # Cross features critical for CTR
        pass

    def handle_missing(self, data):
        # Strategy for missing values
        pass
```

**Coding Tasks:**
- [ ] Implement hash trick
- [ ] Create feature interactions
- [ ] Build preprocessing pipeline
- [ ] Add feature statistics

#### Session 2: Models & Evaluation (2h)
```python
# Implémenter et comparer:
1. Logistic Regression baseline
2. LightGBM/XGBoost
3. Simple Wide & Deep
4. Calibration techniques

# Metrics à calculer:
- LogLoss
- AUC-ROC
- Calibration plots
- Lift curves
```

**Pratique:**
- [ ] Train on sample Criteo data
- [ ] Compare model performances
- [ ] Analyze feature importance
- [ ] Optimize for latency

### 📊 Métriques Jour 3
- Feature engineering complete: ⬜
- Models trained: ___/4
- Metrics understood: ⬜
- Production considerations: ⬜

---

## 📆 JOUR 4 - DEEPKNN & RETRIEVAL SYSTEMS

### 🕐 09:00-13:00 - Modern Retrieval Architecture

#### Session 1: Embedding Systems (2h)
```python
# Build Two-Tower Architecture:
class TwoTowerModel:
    def build_user_tower(self):
        # User embedding network
        pass

    def build_item_tower(self):
        # Item embedding network
        pass

    def training_loop(self):
        # Contrastive learning
        pass
```

**Implementation:**
- [ ] User encoder network
- [ ] Item encoder network
- [ ] Similarity computation
- [ ] Negative sampling strategy

#### Session 2: Vector Search (2h)
```python
# Implement retrieval pipeline:
class VectorSearchSystem:
    def build_index(self):
        # Faiss index creation
        pass

    def search_topk(self, query, k=100):
        # Efficient kNN search
        pass

    def rerank(self, candidates):
        # Business logic layer
        pass
```

**Tasks:**
- [ ] Setup Faiss index
- [ ] Implement search
- [ ] Add caching layer
- [ ] Measure latency

### 📊 Métriques Jour 4
- Embedding model built: ⬜
- Vector search working: ⬜
- Latency < 50ms: ⬜
- Recall@100 calculated: ⬜

---

## 📆 JOUR 5 - BIDDING & SYSTEM DESIGN

### 🕐 09:00-11:00 - First-Price Auction Mastery

#### Bid Shading Implementation
```python
class BidOptimizer:
    def predict_clearing_price(self, features):
        # ML model for price prediction
        pass

    def calculate_optimal_shading(self, value, predicted_price):
        # Maximize expected profit
        pass

    def apply_pacing(self, bid, budget_remaining):
        # Budget pacing logic
        pass
```

**Practice:**
- [ ] Implement shading algorithm
- [ ] Simulate auctions
- [ ] Calculate savings
- [ ] Optimize for different objectives

### 🕐 11:00-13:00 - System Design Patterns

#### Design Problems:
1. **Real-time CTR Prediction System**
   - Requirements gathering
   - Architecture diagram
   - Component design
   - Scaling considerations

2. **Ad Retrieval Pipeline**
   - Offline vs online
   - Caching strategies
   - Load balancing
   - Monitoring

**Whiteboard Practice:**
- [ ] Draw complete architecture
- [ ] Identify bottlenecks
- [ ] Propose optimizations
- [ ] Estimate capacity

### 📊 Métriques Jour 5
- Bidding logic complete: ⬜
- System designs: ___/2
- Trade-offs documented: ⬜
- Scaling addressed: ⬜

---

## 📆 JOUR 6 - CRITEO SPECIFICS & BEHAVIORAL

### 🕐 09:00-11:00 - Criteo Deep Knowledge

#### Topics à Maîtriser:
1. **DeepKNN Details**
   - Architecture specifics
   - Production challenges
   - Performance metrics

2. **FairJob Dataset**
   - Metrics used
   - Bias mitigation
   - Business impact

3. **Header Bidding**
   - Criteo's approach
   - Technical implementation
   - Market impact

**Research & Practice:**
- [ ] Read Criteo blog posts
- [ ] Study FairJob paper
- [ ] Understand business model
- [ ] Prepare intelligent questions

### 🕐 11:00-13:00 - Behavioral Preparation

#### STAR Stories à Préparer:
1. **Technical Challenge**
   - Complex problem solved
   - Impact quantified

2. **Team Conflict**
   - Resolution approach
   - Lessons learned

3. **Failure/Learning**
   - What went wrong
   - How you recovered

4. **Leadership**
   - Initiative taken
   - Results achieved

5. **Innovation**
   - Creative solution
   - Implementation

**Practice:**
- [ ] Write out each story
- [ ] Time to 2-3 minutes
- [ ] Practice delivery
- [ ] Prepare variations

### 📊 Métriques Jour 6
- Criteo knowledge solid: ⬜
- STAR stories ready: ___/5
- Questions prepared: ___/10
- Confidence level: ___/10

---

## 📆 JOUR 7 - INTEGRATION & MOCK INTERVIEWS

### 🕐 09:00-11:00 - Full Review & Integration

#### Quick Review All Topics:
- [ ] DSA patterns checklist
- [ ] SQL templates ready
- [ ] CTR model clear
- [ ] DeepKNN explained
- [ ] Bidding understood
- [ ] System design practiced

#### Speed Rounds:
- 10 min: Solve array problem
- 10 min: Write complex SQL
- 10 min: Explain CTR pipeline
- 10 min: Draw system architecture
- 10 min: Behavioral question
- 10 min: Criteo specifics

### 🕐 11:00-13:00 - Mock Interview Sessions

#### Mock Interview 1 (1h):
- 5 min: Introduction
- 20 min: Coding problem
- 15 min: System design
- 15 min: ML concepts
- 5 min: Questions

#### Mock Interview 2 (1h):
- 5 min: Introduction
- 15 min: SQL problem
- 20 min: Behavioral
- 15 min: Criteo specific
- 5 min: Closing

### 📊 Métriques Jour 7
- All topics reviewed: ⬜
- Mock interviews complete: ___/2
- Weak areas identified: _______
- Ready for interview: ⬜

---

## 📈 PROGRESSION TRACKING

### Daily Checklist Template
```markdown
Date: ___________
Study Hours: _____
Energy Level: ⬜ High ⬜ Medium ⬜ Low

Topics Covered:
- [ ] _________________
- [ ] _________________
- [ ] _________________

Problems Solved: ___
New Concepts: ___
Questions Noted: ___

Tomorrow's Priority:
_____________________
```

### Weekly Progress Matrix

| Skill | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 | Day 7 |
|-------|-------|-------|-------|-------|-------|-------|-------|
| DSA | 🔴 | 🟡 | 🟡 | 🟢 | 🟢 | 🟢 | ✅ |
| SQL | 🔴 | 🟡 | 🟢 | 🟢 | 🟢 | 🟢 | ✅ |
| ML/CTR | 🔴 | 🔴 | 🟡 | 🟢 | 🟢 | 🟢 | ✅ |
| Systems | 🔴 | 🔴 | 🔴 | 🟡 | 🟢 | 🟢 | ✅ |
| Criteo | 🔴 | 🔴 | 🟡 | 🟡 | 🟢 | 🟢 | ✅ |

Legend: 🔴 Not Started | 🟡 In Progress | 🟢 Good | ✅ Mastered

---

## 🎯 FINAL PREPARATION TIPS

### 3 Days Before
- Review all weak areas
- Do 2-3 medium problems daily
- Read recent Criteo news

### 1 Day Before
- Light review only
- Practice pitch
- Prepare outfit & setup
- Sleep 8+ hours

### Day Of Interview
- Warm up with 1 easy problem
- Review key formulas
- Test technical setup
- Breathe & stay confident

---

## 💪 MOTIVATION MANTRAS

> "Preparation breeds confidence"

> "Every problem solved is a step closer"

> "Focus on progress, not perfection"

> "You've got this!"

---

**Remember: Consistent daily practice > Cramming**

**Goal: Be so prepared that success is inevitable!**

---

*Guide created for Criteo Interview Excellence*
*Follow the plan, trust the process, ace the interview!*