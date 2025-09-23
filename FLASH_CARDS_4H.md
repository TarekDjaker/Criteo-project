# 📇 FLASH CARDS - RÉVISION MINUTE CRITEO

> **À imprimer ou garder sur téléphone**
> **Révision rapide entre les phases**

---

## 🎯 CARD 1: NUMBERS TO MEMORIZE

```
Dataset: 45M samples
Features: 13 numerical + 26 categorical
Hash buckets: 1M (10^6)

LogLoss target: < 0.44
AUC target: > 0.80
Calibration: ≈ 1.0

Latency: < 50ms p99
QPS: 1M+ requests/sec
Scale: 750M+ users

First-price: Since 2019
Bid shading: 15-20% savings
Shading factor: 0.75-0.85

Fairness DP: < 0.7%
FairJob: Public dataset
Impact: No perf loss
```

---

## 💻 CARD 2: CODE PATTERNS

### Two-Pointer
```python
left, right = 0, len(arr)-1
while left < right:
    # logic
```

### Sliding Window
```python
for i in range(len(arr)):
    # Remove outside window
    while window and condition:
        window.popleft()
    # Add current
    window.append(i)
```

### BFS Template
```python
queue = deque([start])
visited = {start}
while queue:
    node = queue.popleft()
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)
```

### Hash Trick
```python
hash(str(value)) % buckets
```

---

## 📊 CARD 3: SQL PATTERNS

### Rolling Window
```sql
AVG(metric) OVER (
    PARTITION BY id
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING
    AND CURRENT ROW
)
```

### Attribution
```sql
ROW_NUMBER() OVER (
    PARTITION BY conversion_id
    ORDER BY timestamp DESC
) as last_touch
```

### Percentile
```sql
PERCENTILE_CONT(0.5)
WITHIN GROUP (ORDER BY value)
```

---

## 🤖 CARD 4: ML FORMULAS

### CTR Value
```
Bid = pCTR × pCVR × Value × (1-Margin)
```

### Bid Shading
```
Final_Bid = Max_Bid × 0.8
Savings = 15-20%
```

### Calibration
```
Calibration = mean(predicted) / mean(actual)
Target ≈ 1.0
```

### Lift@K
```
Lift = CTR_topK / CTR_baseline
Target > 2.0
```

### Demographic Parity
```
DP = |P(Y=1|A=0) - P(Y=1|A=1)|
Target < 0.007
```

---

## 🏗️ CARD 5: DEEPKNN ARCHITECTURE

```
USER → [Dense 256] → [Dense 128] → [Embed] → [L2 Norm]
                                                    ↓
                                            [Dot Product]
                                                    ↑
ITEM → [Dense 256] → [Dense 128] → [Embed] → [L2 Norm]

OFFLINE: Generate embeddings, Index with Faiss
ONLINE: User embed (5ms) + Search (15ms) + Rerank (10ms)
TOTAL: 30ms < 50ms ✓
```

---

## 📈 CARD 6: SYSTEM SCALE

### Real-Time Pipeline
```
Request → LB → Feature Service → Model → Ranking → Response
         ↓           ↓              ↓         ↓
      [Cache]    [Redis]      [A/B Test]  [Rules]
```

### Capacity Planning
```
1M QPS × 100ms latency = 100K concurrent
Memory: 100GB embeddings
Storage: 1TB/day logs
Models: 100MB each
```

---

## 🎤 CARD 7: PITCH BULLETS

### 60-Second Structure
```
1. Who (10s): Role + Experience
2. What (20s): Recent achievement
3. How (20s): Technical approach
4. Why (10s): Why Criteo
```

### Key Phrases
- "Optimized CTR pipeline: 200ms → 45ms"
- "0.82 AUC maintained"
- "Similar to your DeepKNN approach"
- "18% cost savings via bid shading"
- "Studied FairJob and your blog posts"

---

## ❓ CARD 8: TOP QUESTIONS

### Technical
1. "How do you handle embedding updates without downtime?"
2. "Is bid shading global or per-publisher?"
3. "Trade-off between fairness and ROI?"

### Strategic
4. "Post-cookies strategy beyond contextual?"
5. "Differentiation vs Google/Meta?"

### Culture
6. "% time on research vs production?"
7. "Publishing opportunities?"

---

## 🚨 CARD 9: EMERGENCY ANSWERS

### If stuck on coding:
"Let me think through the approach: First, I'd consider time/space complexity, then edge cases..."

### If don't know:
"I haven't worked with that specifically, but based on similar systems, I would approach it by..."

### If nervous:
"Let me take a moment to organize my thoughts..."
*Breathe, then structure: Problem → Approach → Trade-offs*

---

## ✅ CARD 10: FINAL CHECKLIST

### Technical Ready?
- [ ] Can code two-pointer
- [ ] Can write SQL window
- [ ] Know CTR metrics
- [ ] Can draw DeepKNN
- [ ] Understand bid shading

### Behavioral Ready?
- [ ] STAR stories (3)
- [ ] 60s pitch smooth
- [ ] Questions prepared (5)

### Logistics Ready?
- [ ] Setup tested
- [ ] Links ready
- [ ] Water nearby
- [ ] Notes accessible

---

## 🔥 ONE-LINERS TO REMEMBER

> "CTR without calibration is like F1 without brakes"

> "DeepKNN = sophistication offline, simplicity online"

> "First-price needs intelligence to not overpay"

> "Fairness is a feature, not a constraint"

> "At Criteo scale, every millisecond counts"

---

## 💪 CONFIDENCE BOOSTERS

- You have **4 hours** of focused prep
- You understand **core concepts**
- You can **code solutions**
- You know **Criteo specifics**
- You are **prepared**

---

## 🎯 LAST MINUTE REVIEW

30 seconds before interview:
1. Deep breath (4-7-8 technique)
2. Smile (changes your voice)
3. "I'm excited about this opportunity"
4. Have water ready
5. Notes visible but not distracting

---

**REMEMBER: They want you to succeed!**

**BE YOURSELF. BE CONFIDENT. YOU'VE GOT THIS! 🚀**