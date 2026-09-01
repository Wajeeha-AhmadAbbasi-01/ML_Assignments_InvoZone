# Observations — Credit Card Customer Segmentation

## Model Selection Rationale

### Why I Used KMeans and DBSCAN

I chose to run two fundamentally different clustering algorithms on the credit card dataset (~8,950 customers across 8 behavioral features) because each serves a distinct purpose:

**KMeans** was my primary segmentation tool. It's fast, interpretable, and assigns every customer to a cluster—which is essential for operational deployment (e.g., targeting campaigns, setting credit limits, assigning service tiers). The algorithm iteratively refines centroids until convergence, producing compact, roughly spherical clusters. For business segmentation, this means I get a clean, usable 4-way split of the customer base.

**DBSCAN** was my secondary validation and outlier-detection tool. Unlike KMeans, DBSCAN doesn't assume cluster shapes or require every point to be assigned. It finds regions of high density separated by low-density regions, and explicitly marks points that don't fit anywhere as "noise." This is critical for credit card data, where I suspected there might be a small but significant group of high-value, high-risk outliers that KMeans would force into an otherwise "normal" cluster.

---

## Why KMeans Performed Like This

### How Scaling Changed the Result

**Without standardization**, the raw feature ranges would have completely dominated the distance calculations:

- `CREDIT_LIMIT` ranges from ~$1,000 to ~$20,000+
- `PURCHASES_FREQUENCY` is bounded 0–1
- `BALANCE` varies from $0 to ~$16,000

If I'd used unscaled data, credit limit alone would have been the de facto primary driver of similarity. Two customers with similar credit limits but wildly different spending behaviors would look "close" purely because of the units, not actual behavioral resemblance.

**I used `StandardScaler`** to center every feature at mean 0 with standard deviation 1. This put all eight features on equal footing. Now the algorithm judges similarity based on behavioral *pattern* (how balanced, how frequent, how much they buy) rather than the accident of which features were originally measured in large vs. small units.

**The impact was decisive.** With scaling, I got behaviorally distinct clusters. Without it, I would have gotten clusters that were mostly just "high credit limit vs. low credit limit" with behavioral differences as secondary noise.

---

### How the Final K Was Selected

I used **two complementary methods** instead of trusting either one alone:

**The elbow plot** showed inertia dropping steeply from K=2 through K=5, then flattening significantly after K=5. The "bend" is visible around K=5, where the marginal reduction in inertia begins to diminish notably.

**The silhouette scores** provided a different perspective:

- K=2: 0.42 (high)
- K=3: 0.39 (high)
- K=4: 0.26 (lower)
- K=5: 0.28 (slightly higher than K=4)
- K=6: ~0.28–0.29 (stable)

The very high silhouette at K=2 and K=3 is a **red flag** in this context—it usually means I'm over-simplifying. Two clusters would just be "active vs. inactive," which loses the nuance between frequent small-transaction users and high-value/high-cash-advance users.

**My decision:** I chose **K=5** as the sweet spot where the elbow bend is clearest *and* I capture enough behavioral distinction to be actionable, even though the silhouette score (0.28) is lower than K=2 or K=3. The moderate silhouette score is honest—it acknowledges that these clusters aren't perfectly separated, which matches reality: customer behavior is messy and doesn't fit into clean silos.

---

## KMeans Cluster Interpretations

The five KMeans clusters from my results are:

### Cluster 0 — High-Value, High-Activity Customers (296 customers, 3.3%)

- **Average balance:** $4,051.68
- **Average purchases:** $8,647.28 (highest)
- **Average cash advances:** $812.72
- **Average payments:** $8,676.84 (highest)
- **Credit limit:** $10,275.84 (highest)
- **Purchase frequency:** 0.94 (very high)
- **Average purchase transactions:** 103.62 (highest)

This is a small but extremely high-value segment. They have the highest balances, make the most purchases, have the highest credit limits, and pay back the most. Their purchase frequency of 0.94 means they're active nearly every cycle. This is the bank's premium revenue segment.

### Cluster 1 — Inactive/Low-Engagement Customers (1,222 customers, 13.7%)

- **Average balance:** $105.04 (lowest)
- **Average purchases:** $363.01
- **Average cash advances:** $298.69
- **Average payments:** $1,093.47
- **Credit limit:** $3,799.63
- **Purchase frequency:** 0.29

This segment has very low balances and moderate activity. They're not completely dormant, but their engagement is minimal compared to other segments. They may be occasional users or newer customers still establishing behavior patterns.

### Cluster 2 — Frequent, Moderate-Value Purchasers (2,946 customers, 32.9%)

- **Average balance:** $945.11
- **Average purchases:** $1,483.38
- **Average cash advances:** $181.60 (lowest)
- **Average payments:** $1,539.56
- **Credit limit:** $4,598.99
- **Purchase frequency:** 0.90 (very high)
- **Average purchase transactions:** 25.01

This is the largest and most frequent purchasing group. They use their cards regularly (0.90 frequency) but keep transactions moderate. Their low cash advance usage suggests they primarily use the card as a credit tool rather than a cash access tool.

### Cluster 3 — Cash-Advance-Oriented Customers (944 customers, 10.6%)

- **Average balance:** $5,321.88 (second highest)
- **Average purchases:** $616.14 (lowest)
- **Average cash advances:** $5,040.60 (highest)
- **Average payments:** $4,109.08
- **Credit limit:** $8,769.60 (second highest)
- **Purchase frequency:** 0.33
- **Cash advance frequency:** 0.46 (highest)

This is a highly distinctive segment. Their cash advances ($5,040.60) far exceed their purchases ($616.14), indicating they're using the card primarily as a cash access tool rather than for purchasing. They carry high balances and have moderate payment activity. This is a **credit-risk-sensitive** segment.

### Cluster 4 — Low-Engagement, Low-Balance Customers (2,868 customers, 32.0%)

- **Average balance:** $1,494.75
- **Average purchases:** $255.05 (very low)
- **Average cash advances:** $749.05
- **Average payments:** $974.06
- **Credit limit:** $3,157.03 (lowest)
- **Purchase frequency:** 0.18 (lowest)
- **Tenure:** 11.88 (longest)

This is a large segment of low-activity customers. Despite the longest average tenure (11.88 months), they have the lowest purchase frequency and very low purchase amounts. They hold relatively low credit limits and make small payments. This looks like a dormant or near-dormant segment.

---

## Why DBSCAN Performed Like This

### Sensitivity to eps and min_samples

The grid search results show DBSCAN was **extremely sensitive** to parameter tuning, with a sharp trade-off between finding too many tiny clusters versus collapsing everything into one mega-cluster:

**At eps=0.5 (tight neighborhood radius):**

- Found 5–27 tiny clusters
- 53.7–68.4% of customers marked as noise
- Silhouette scores negative or near-zero (−0.42 to 0.20)

The radius is too tight; clusters fragment into slivers, and most points end up as noise. This is unhelpful for segmentation.

**At eps=0.8:**

- Found 4–21 clusters
- 26.6–37.4% noise
- Silhouette scores range from −0.29 to 0.29

A slight improvement but still problematic. Too much noise, and the clusters that form aren't well-separated.

**At eps=1.0:**

- Found 1–9 clusters
- 15.7–23.2% noise
- Best silhouette: 0.37 (eps=1.0, min_samples=15) with 2 clusters

This starts to become usable, but with 1–3 clusters, it's too coarse for meaningful segmentation.

**At eps=1.5:**

- Found 1–2 clusters
- 5.5–8.0% noise
- Best silhouette: 0.30 (eps=1.5, min_samples=5) with 2 clusters

The clusters are better separated, but with only 2 clusters, I'm back to "active vs. inactive," which loses nuance.

**At eps=2.0:**

- Found 1 cluster across all min_samples values
- 2.4–3.2% noise
- Silhouette undefined for 1 cluster

At this point, DBSCAN collapses everything into one blob—completely unhelpful.

---

### What I Chose: eps=1.5, min_samples=5

I selected **eps=1.5, min_samples=5** as the final DBSCAN configuration. This produced:

- **2 clusters** (rather than 1 giant one)
- **489 noise points** (5.5% of the dataset)
- **Silhouette score: 0.30**

I chose this over eps=1.0, min_samples=15 (which had a slightly higher silhouette of 0.37 but also 2 clusters with 23.2% noise) because:

1. The 5.5% noise rate is more interpretable—these are truly anomalous customers, not just borderline cases.
2. The noise points (489 customers) are genuinely high-value outliers worth investigating.
3. eps=1.5 with min_samples=5 provides a better balance between capturing structure and not over-fragmenting.

---

### Which Points Were Considered Noise and Why That's Useful

The 489 DBSCAN noise points (5.5% of customers) were not randomly scattered outliers—they were **systematically high-activity, high-value customers**. From my cluster profile results:

| Metric | Cluster 0 (in DBSCAN, not noise) | Noise points |
|---|---|---|
| **Balance** | ~$4,052 | Very high (similar pattern) |
| **Purchases** | $8,647 | Very high |
| **Cash advances** | $813 | Very high |
| **Payments** | $8,677 | Very high |
| **Credit limit** | $10,276 | Very high |

These are the **highest-value, highest-risk customers**—people who don't behave like "typical" credit card users. They make large purchases, carry substantial balances, and engage in unusual cash behavior.

**Why this is useful:** KMeans would have force-assigned these 489 customers to whichever centroid was nearest (likely Cluster 0), hiding them inside an otherwise "normal" segment summary. DBSCAN explicitly surfaces them as "this is unusual." For credit risk, fraud detection, or premium service targeting, knowing which customers deviate sharply from the norm can be as valuable as knowing the norm itself.

---

## PCA Variance Explained

From my results, the first two principal components explain:

- **PC1: ~32% of variance** (likely dominated by balance, purchases, payments, and credit limit)
- **PC2: ~20% of variance** (likely dominated by purchase frequency and cash advance behavior)
- **Together: ~52% of the original 8-dimensional variance**

This means **48% of the structure in the data is not visible in 2D scatter plots.** The PCA visualizations are useful for spotting cluster *shapes* and rough separation, but they don't capture the full behavioral distance between customers. Two points that look far apart in PC1–PC2 space might actually be quite close in PC3 or PC4, or vice versa.

This is why I didn't rely solely on visual inspection—I grounded my K choice in the elbow and silhouette metrics, which operate in the full 8D space.

---

## Which Clustering Algorithm Was More Useful

### KMeans Strengths

- **Assigns every customer to a cluster**, making results immediately deployable (marketing campaigns, credit risk tiers, service level assignments).
- **Five interpretable segments** with clear behavioral profiles (high-value, inactive, frequent, cash-advance, low-engagement).
- **Stable results** across the dataset—no need to tune two interacting parameters.
- **Silhouette score of 0.28** is honest but workable; the clusters are real, just not perfectly separated.

### KMeans Limitations

- **Forced to assign outliers** to whichever cluster they're "least far from," obscuring their true distinctiveness.
- **Assumes roughly convex, spherical clusters** around centroids (not always true).
- **Silent about who doesn't fit well**—can hide problems.

### DBSCAN Strengths

- **Explicitly identified 489 high-activity outliers** without forcing them into a pigeonhole.
- **Doesn't assume cluster shape**—can follow genuine density patterns.
- **Useful for anomaly detection** in high-risk customer segments.

### DBSCAN Limitations

- **Extremely sensitive to parameter tuning**; eps and min_samples interact in non-obvious ways.
- **My chosen configuration (eps=1.5, min_samples=5) found only 2 clusters**, reducing actionability.
- **Noise points aren't assigned to any segment**, so operationally, "what do we do with these 489 people?" still requires a second decision.
- **Less stable for datasets with varying density**—dense regions vs. sparse outliers are treated very differently.

### Verdict

**For business segmentation, KMeans is more actionable.** It provides a clean, interpretable five-way split of the customer base. The silhouette score acknowledges real fuzziness in the data, and that's honest.

**But DBSCAN adds value by flagging the ~5.5% of high-value outliers** for specialized treatment (premium service, enhanced monitoring, custom credit terms). KMeans would quietly include them in Cluster 0, but DBSCAN makes it explicit: "These 489 are fundamentally different—they don't play by the rules of the other clusters."

---

## What I Would Conclude

The credit card customer base is genuinely diverse, but not chaotically so. The KMeans clusters capture meaningful behavioral distinctions, and DBSCAN reveals the presence of a small but important outlier segment.

1. **Cluster 0 (high-value, high-activity, 3.3%)** is the smallest but most valuable segment—likely driving a disproportionate share of revenue. These should be the target of premium retention efforts.

2. **Cluster 3 (cash-advance-oriented, 10.6%)** is a distinctive credit-risk segment. Their cash advance usage (5x their purchase volume) signals they're using the card for cash access, not purchasing. This warrants enhanced monitoring and possibly different credit terms.

3. **Clusters 1 and 4 (inactive, ~46% combined)** represent a large group of underutilized accounts. This is either an untapped growth opportunity or a signal that credit was extended to people unlikely to be active users. Activation campaigns could be valuable here.

4. **Cluster 2 (frequent purchasers, 32.9%)** is the largest active segment. They use their cards regularly, keep transactions moderate, and rarely use cash advances. They're the "everyday user"—high frequency, moderate value, low risk.

5. **DBSCAN's 5.5% noise segment** is compelling. These are real customers with real behavior, just unusual enough that they don't cluster with anyone else. Treating them as a separate, premium risk profile (vs. forcing them into KMeans Cluster 0) could improve credit decisions and fraud detection.

6. **Scaling was non-negotiable.** Without standardization, credit limit alone would have driven the entire analysis, and I would have missed the purchasing frequency, cash advance, and payment behavior distinctions that actually separate the clusters.

---

## Final Takeaway

**KMeans with K=6** is a solid, deployable segmentation—it's stable, interpretable, and grounded in both the elbow plot and silhouette metrics. But it's not gospel. **DBSCAN's noise segment** is a reminder that the "average" customer profile hides real outliers who behave in fundamentally different ways.

The most honest  conclusion is that the credit card user base has at least **six distinct behavioral profiles**: Clusters 0–4 (which KMeans surfaces), plus the 5.5% of outliers (which DBSCAN surfaces). A robust strategy would acknowledge all six. For operational deployment, KMeans provides the clean framework; for risk management and premium targeting, DBSCAN's outliers offer additional signal