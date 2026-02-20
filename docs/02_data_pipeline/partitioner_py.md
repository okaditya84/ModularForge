# `partitioner.py` — Data Partitioning for Expert Specialization

> **Source:** [partitioner.py](file:///Users/aditya/Documents/Aditya/Python/LLM_In_Parts/modularforge/data/partitioner.py) · **Lines:** 444 · **Prereqs:** `tokenizer.py`, `dataset.py`

---

## What This File Does

Splits the training corpus into **N disjoint partitions** — one for each expert module. How you split the data determines what each expert specializes in.

```
Full corpus: 🐱🐶🌳🌺🚗🏠📱🎮🧬🔬...
partition 0: 🐱🐶🧬         ← Expert 0: biology
partition 1: 🌳🌺🌍         ← Expert 1: nature
partition 2: 🚗🏠📱         ← Expert 2: technology
partition 3: 🎮🎬🎵         ← Expert 3: entertainment
partition 4: 🔬📊💰         ← Expert 4: science & finance
```

---

## Three Partitioning Strategies

### 1. `"random"` — Shuffle and Deal

```
Articles: [A, B, C, D, E, F, G, H, I, J]
Shuffle:  [E, C, H, A, J, F, B, I, D, G]
Deal:     Expert0=[E,J]  Expert1=[C,F]  Expert2=[H,B]  Expert3=[A,I]  Expert4=[D,G]
```

**Pros:** Fair, unbiased, no external dependencies.
**Cons:** Experts don't specialize — they all learn a random mix of topics.
**When to use:** Baseline comparison, when specialization doesn't matter.

### 2. `"clustered"` — Semantic Grouping (Default)

Uses `sentence-transformers` to compute embeddings for each article, then `scikit-learn` KMeans to group similar articles together.

```
Articles → Embed with all-MiniLM-L6-v2 → 384-dim vectors → KMeans(k=5) → Clusters
```

**Pros:** Natural topic specialization. Expert 0 might learn "science", Expert 1 "sports," etc.
**Cons:** Requires `sentence-transformers` + `scikit-learn` (heavy dependencies). Embedding all articles takes time.
**When to use:** Production use — gives the best MoE results.

### 3. `"curriculum"` — Complexity-Based

Sorts articles by "complexity" (approximated by average word length) and assigns progressively harder articles to later experts.

```
Easy articles → Expert 0
Medium articles → Expert 1
...
Hard articles → Expert 4
```

**Pros:** Interesting curriculum learning dynamics.
**Cons:** "Complexity by word length" is a crude proxy. Could be improved with perplexity-based scoring.

---

## Key Parameters

### `overlap_ratio` (default: 0.1)

Each expert gets 10% extra articles from other partitions' data. This creates a "bridge" of shared knowledge between experts.

```
Without overlap (0.0):   Expert0=[A,B,C]  Expert1=[D,E,F]   (disjoint)
With overlap (0.1):      Expert0=[A,B,C,D] Expert1=[D,E,F,C] (D and C are shared)
```

**Why overlap matters:** Without overlap, experts might learn incompatible representations for common words. A little overlap ensures coherent behavior when experts are combined during inference.

**Why 10%?** Empirically, 5-15% overlap gives good coherence without reducing specialization too much. At 50%, experts see too much shared data and don't specialize.

---

## Code Walkthrough

### The `partition()` Method (Line 120-170)

```python
def partition(self, texts: list[str]) -> list[list[str]]:
    if len(texts) < self.n_partitions:
        raise ValueError(...)

    if self.strategy == "random":
        base_partitions = self._random_partition(texts)
    elif self.strategy == "clustered":
        base_partitions = self._clustered_partition(texts)
    elif self.strategy == "curriculum":
        base_partitions = self._curriculum_partition(texts)

    if self.overlap_ratio > 0:
        final = self._add_overlap(texts, base_partitions)
    else:
        final = base_partitions

    self._rebalance(final)
    return final
```

**Why rebalance?** After adding overlap, some partitions may end up significantly larger than others. Rebalancing trims oversized partitions to keep training time uniform across experts.

### Clustered Partitioning (Lines 220-308)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)
```

**`all-MiniLM-L6-v2`:** A tiny (80MB) sentence embedding model that maps sentences to 384-dimensional vectors. Fast enough to embed 10K+ articles.

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=self.n_partitions, random_state=self.seed)
labels = kmeans.fit_predict(embeddings)
```

**Why KMeans?** Simple, well-understood, and produces hard cluster assignments. Each article goes to exactly one cluster.
**Alternative:** `HDBSCAN` (density-based) can find natural cluster counts, but we KNOW how many clusters we want (= n_experts), making KMeans ideal.

```python
except ImportError:
    logger.warning(
        "sentence-transformers or scikit-learn not installed. "
        "Falling back to random partitioning."
    )
    return self._random_partition(texts)
```

**Graceful degradation:** If the heavy dependencies aren't installed, fall back to random instead of crashing. This lets the pipeline run (with reduced quality) on minimal installations.

---

## Edge Cases

1. **Fewer articles than partitions:** Raises `ValueError`. You need at least 1 article per expert.
2. **All articles identical:** KMeans assigns them all to one cluster. Rebalancing redistributes them.
3. **Very short articles:** Get filtered out by `min_article_length` in config BEFORE partitioning. Articles shorter than ~100 characters don't provide enough signal for embedding.
4. **Seed consistency:** Setting `seed` in config ensures identical partitions across runs. Critical for reproducibility.

---

## Q&A

**Q: Why not use LDA (topic modeling) instead of KMeans on embeddings?**
A: LDA gives soft (probabilistic) topic assignments and requires tuning (number of topics, iterations, alpha/beta priors). KMeans on pre-trained embeddings is simpler and leverages the semantic quality of modern sentence embeddings, which outperform bag-of-words models like LDA.

**Q: Does partitioning order affect training?**
A: Yes. Expert 0 trains first and Expert 4 last. If the data order is biased (e.g., all easy data first), some experts get biased training. That's why `random` shuffles and `clustered` is order-independent.

**Q: What if an expert's partition is much smaller than others?**
A: The `_rebalance()` method redistributes articles from oversized to undersized partitions until all partitions are within 20% of the average size.
