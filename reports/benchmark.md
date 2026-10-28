# Boréal Marché — Apriori Performance, Memory & Coverage Benchmark

**Course:** 420-975-VA — Unsupervised Learning  
**Dataset:** 1,000,000 baskets · 8,379,837 transaction rows · 98 products (`avg = 8.38` items/basket)  
**Algorithm:** `mlxtend.frequent_patterns.apriori` & `association_rules` (`metric='confidence'`, `min_threshold=0.25`)

---

## 1 · Wide-Format Transformation Benchmark (Phase 1)

Before calling Apriori, we must transform the raw long-format transaction table (`8,379,837` rows of `TransactionID, Item`) into a boolean matrix of shape `(1,000,000, 98)`.

| Transformation Approach | Execution Time | Peak RAM Allocated | Final Table Size in RAM | Engineering Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Naive `pd.crosstab()` / `.unstack()`** | ~42.0 – 85.0 s | ~1,850 – 3,100 MB | 747.7 MB (`int64`) | Builds massive intermediate hash tables and 8-byte integer cells before casting to `bool`. |
| **Pre-allocated NumPy `bool` matrix via Categorical codes (Chosen)** | **3.8 – 11.1 s** *(incl. ZIP decompression)* | **161.4 MiB** | **93.5 MiB** (`98.0 MB` decimal) | Maps `TransactionID` to `0..999,999` and `Item` to `0..97`, setting `matrix[row_idx, col_idx] = True` in one pass (exact `1 byte` per boolean cell). |

---

## 2 · Apriori Threshold Benchmark Table (Phase 2 & Phase 6)

Below are the real measurements collected on our `1,000,000 × 98` boolean table using `mlxtend.frequent_patterns.apriori(..., use_colnames=True, low_memory=True)` and `association_rules(..., metric='confidence', min_threshold=0.25)`:

| `min_support` | Itemsets | Rules (`conf >= 0.25` / Actionable 1-consequent) | Time | Peak memory | Outcome |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`0.05`** *(50,000 baskets)* | `191` | `282` raw / **`118` actionable** | **5.2 s** | **148.2 MB** | **Too restrictive.** Only captures top-level summer BBQ & basic household pairs; misses winter raclette/fondue rules (`~2.7%–3.4%` annual support) and almost all 3-item antecedents. |
| **`0.02`** *(20,000 baskets)* | `658` | `2,987` raw / **`1,080` actionable** | **12.3 s** | **166.5 MB** | **Good but incomplete.** Captures core meal clusters, but misses 3-item rules such as `{Steak de boeuf, Pain à hamburger, Charbon de bois} -> Relish` (`support = 2.25%`) or smaller seasonal combinations (`1%–2%` support). |
| **`0.01`** *(10,000 baskets)* | `2,267` | `8,770` raw / **`2,856` actionable** | **26.9 s** | **192.8 MB** | **Selected production threshold.** Balances rich multi-item context (`up to 5-itemsets`) with fast offline mining (`~27s`). Covers all **57** recommendable products in the store. |
| **`0.005`** *(5,000 baskets)* | `7,106` | `26,136` raw / **`9,184` actionable** | **53.2 s** | **246.4 MB** | **Diminishing returns & combinatorial explosion.** Doubles execution time and triples rule volume (`26,136` raw rules), yet adds **zero** new recommendable products (still `57 / 98`, because the 5 luxury specialties sit down at `~0.08%` support). |

> **Note on "Raw" vs. "Actionable" Rules at `min_support = 0.01`:**  
> At `min_support = 0.01` and `min_confidence = 0.25`, `association_rules()` produces `8,770` total rules (including multi-item consequents). Restricting to single-item consequents (`len(consequents) == 1`) with `lift > 1.0` yields `3,442` rules. Of those, `586` rules simply recommend **`Sac réutilisable` (Reusable bag)** with a trivial lift between `1.0000` and `1.0088` because 90% of all baskets contain a reusable bag. Filtering out those trivial reusable-bag consequents (`lift > 1.05`) leaves **exactly `2,856` actionable association rules** (`lift` from `2.14` to `10.17`), which are saved to `models/rules.json`.

---

## 3 · Why Default Apriori (`low_memory=False`) Hits a `MemoryError` — And How We Fixed It

### What Happens Under the Hood with `low_memory=False` (Default)
When you call `apriori(basket_df, min_support=0.01, use_colnames=True)` with its default parameter `low_memory=False`, `mlxtend` vectorizes support counting across all candidate itemsets of size $k$ at once using 3D NumPy fancy indexing:

```python
# Inside mlxtend.frequent_patterns.apriori when low_memory=False:
_bools = np.all(X[:, combin], axis=2)
```

Here:
- `X` is our boolean basket matrix of shape **`(1,000,000, 98)`**.
- At pass $k = 2$, `93` out of `98` products survive `min_support >= 0.01` (all except the 5 rare specialty products).
- Therefore, `generate_new_combinations()` builds $\binom{93}{2} = 4,278$ candidate pairs, so `combin` has shape **`(4278, 2)`**.
- Evaluating `X[:, combin]` forces NumPy to allocate a **single contiguous 3D boolean tensor** of shape:
  $$\text{Shape} = (1,000,000 \times 4,278 \times 2) = 8,556,000,000 \text{ bytes} \approx \mathbf{8.56\text{ GB of contiguous RAM!}}$$
- Even worse, at pass $k = 3$ (especially at `min_support = 0.005` where thousands of candidate triplets are tested), `X[:, combin]` demands **15 GB to 35+ GB of contiguous RAM**, immediately triggering `numpy._core._exceptions._ArrayMemoryError` (`MemoryError`) or freezing the operating system in swap thrashing.

### How `low_memory=True` Solves It
Setting **`low_memory=True`** in `apriori()` replaces bulk 3D tensor slicing with `generate_new_combinations_low_memory()`, a Python generator that evaluates candidate itemsets **one combination at a time** over the `1,000,000` rows:
- Instead of allocating `(1,000,000, 4278, 2)` all at once (`8.56 GB`), it only ever holds a single candidate slice of `(1,000,000, 2)` (`2.0 MB`) in memory at any instant.
- As a result, peak RAM stays below **250 MB** even at `min_support = 0.005`, completing all four benchmark runs cleanly in 5 to 53 seconds.

---

## 4 · Catalogue Coverage Analysis (Phase 6)

Out of the **98 products** in Boréal Marché's catalogue:
- **57 products (`58.16%`)** can be recommended by our system (they appear as the consequent in at least one of the 2,856 rules).
- **72 products (`73.47%`)** appear as an antecedent in at least one rule (they can trigger recommendations when combined in a basket, including 15 everyday staples that only fire when paired with `Sac réutilisable`).
- **41 products (`41.84%`)** can **never** be recommended as a consequent. They fall into three distinct groups:
  1. **5 Ultra-Rare Luxury Specialties (`< 0.1%` support):** `Safran` (`818` baskets / `0.082%`), `Huile de truffe` (`817` / `0.082%`), `Caviar` (`810` / `0.081%`), `Vinaigre balsamique vieilli` (`779` / `0.078%`), and `Foie gras` (`756` / `0.076%`). Because their frequency is more than 12× below `min_support = 0.01` (`10,000` baskets), Apriori prunes them in Pass 1.
  2. **1 Ubiquitous Checkout Item (`90.0%` support):** `Sac réutilisable` (`900,026` baskets). It appears in almost every basket regardless of what else is bought (`lift ~ 1.00`), so recommending it adds zero value.
  3. **35 Independent Everyday Staples (`3.1% – 16.1%` support):** Items like `Biscuits` (`16.1%`), `Noix mélangées` (`15.5%`), `Beurre d'arachide` (`15.4%`), `Savon à vaisselle` (`15.0%`), `Fromage cheddar` (`14.3%`), `Papier hygiénique` (`14.0%`), `Céréales` (`13.9%`), `Lait 2%` (`11.9%`), `Baguette` (`6.9%`), and `Carottes` (`3.1%`). Customers buy these general household staples across all types of trips rather than as part of a specific recipe cluster, so no antecedent boosts their purchase probability above our `25%` confidence threshold with `lift > 1.05`.

---

## 5 · Two Core Limitations of This Approach — And How to Fix Each One

### Limitation 1: Blindness to the Long Tail ("The Support Cliff") and Reinforcement Bias
- **What Apriori cannot do:** Apriori enforces a single uniform `min_support` cutoff across the entire store. Because `min_support = 0.01` requires at least 10,000 co-occurrences, the 5 luxury products in *Spécialités* (`Caviar`, `Foie gras`, `Huile de truffe`, `Safran`, `Vinaigre balsamique vieilli`) are completely invisible to the algorithm—even though our notebook proved that shoppers who buy `Safran` also buy `Huile de truffe` **38.6% of the time (`lift = 472.8×`)** and `Caviar` **37.8% of the time (`lift = 466.4×`)**! Worse, because 41 products are never recommended, the website creates a **reinforcement bias loop**: popular recipe items get recommended more, while niche and newly launched catalogue items never get exposure.
- **What we would add to fix it:** Implement **Multiple Minimum Supports (MSApriori)**—assigning each product a support threshold proportional to its category frequency (e.g., `0.02%` for *Spécialités* vs. `1.0%` for *Épicerie*)—or pair Apriori with an **Item2Vec / Cosine-Similarity Content & Co-occurrence Fallback** that triggers whenever a rare item (`< 1%` support) is placed in the basket.

### Limitation 2: Zero Customer Personalization and Ignorance of Purchase History / Sequential Timing
- **What Apriori cannot do:** Our transaction dataset tracks `32,000` unique `CustomerID`s and exact `Date`s, yet Apriori discards both columns and treats every basket as an anonymous, static boolean vector. A vegetarian customer, a household with a baby, and a lactose-intolerant shopper all receive the exact same suggestions when they put an item in their basket. It also ignores purchase quantities and repeat replenishment cycles (e.g., buying coffee or diapers every two weeks).
- **What we would add to fix it:** Build a **two-stage hybrid recommender**: use Apriori rules as the fast, explainable candidate generator for the current basket, and re-rank those candidates using a personalized **Collaborative Filtering model (ALS Matrix Factorization / LightFM)** trained on each `CustomerID`'s historical purchase vector, plus a **time-aware seasonal weighting** (filtering rules on a rolling 60–90 day window so winter shoppers never get summer charcoal rules).
