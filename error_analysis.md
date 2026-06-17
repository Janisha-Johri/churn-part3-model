# Error Analysis: Churn Prediction Model

**Model:** XGBoost classifier | **Threshold:** 0.40 | **Evaluated on:** Test set (336 customers,
held out, used only once for this final evaluation)

**Confusion Matrix (Test Set):**

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| **Actual: No Churn** | TN = 124 | FP = 44 |
| **Actual: Churn** | FN = 23 | TP = 145 |

Accuracy 80.1%, Precision 76.7%, Recall 86.3%, F1 0.812, ROC-AUC 0.868.

---

## Part A: False Positives (Predicted Churn, Actually Retained) - 44 total

These customers were flagged as high-risk but ended up purchasing again in the 60-day window.  
**Business risk:** Wasted retention spend (an unnecessary discount, outreach, or campaign touch on
a customer who didn't need it). Per Part 2's budget logic, this cost is bounded since lower-cost
channels are used for less-certain segments, but at scale, false positives still represent
real, avoidable marketing spend.

### 1. CUST01246: predicted probability 0.96, but retained
**Profile:** Recency 262 days (very high), Frequency 0, Monetary ₹0, last visit 60 days ago.  
**Why the model flagged this:** Recency is the dominant feature (see feature importance), and 262
days of silence combined with zero recent orders is an extreme value the model has learned to
associate strongly with churn.  
**Why it was wrong:** This customer apparently made a surprise return purchase despite extreme
dormancy, possibly a seasonal or gift-occasion buyer whose purchase cadence doesn't follow the
180-day window logic at all.  
**Business takeaway:** A small number of customers have naturally long, infrequent purchase
cycles (e.g., annual gift sets) that look identical to "churning" in a 180-day feature window.

### 2. CUST00437: predicted probability 0.95, but retained
**Profile:** Recency 151 days, Frequency 1, Monetary ₹729, last visit 33 days ago.  
**Why flagged:** High recency + low frequency, classic disengagement profile.  
**Why wrong:** Came back anyway, possibly prompted by an external factor not captured in our
features (e.g., a personal need, a friend's recommendation, a one-off promotional email outside
our `campaign_clicks_30d` window).  
**Business takeaway:** The model can't see causes outside the dataset; some retention is driven
by factors we simply don't have data on.

### 3. CUST01017: predicted probability 0.94, but retained
**Profile:** Recency 133 days, Frequency 2, Monetary ₹1,167, return rate 50%, last visit 13 days
ago.  
**Why flagged:** High recency combined with a high return rate (50% of orders returned), a
"high value but unhappy"-style profile from Part 2's segmentation.  
**Why wrong:** Despite the return history, the customer's web activity was recent (last visit only
13 days ago), a signal the model under-weighted relative to recency.  
**Business takeaway:** This particular customer might be better served by Part 2's "High-Value
but Unhappy" service-recovery strategy rather than treating the model's churn flag as final.

### 4. CUST01325: predicted probability 0.94, but retained
**Profile:** Recency 186 days, Frequency 0, Monetary ₹0, last visit 43 days ago.  
**Why flagged:** Same extreme dormancy pattern as Case 1.  
**Why wrong:** Likely the same "long natural cycle" explanation.  
**Business takeaway:** Reinforces that a meaningful minority of "dormant-looking" customers are
not actually lost (Dormant + Platinum tier, retained despite 226 days of silence).

### 5. CUST01370: predicted probability 0.94, but retained
**Profile:** Recency 161 days, Frequency 2, Monetary ₹1,246, last visit 35 days ago.  
**Why flagged:** High recency despite reasonable historical frequency/monetary.  
**Why wrong:** Came back despite the gap, again, web activity (35 days) wasn't extreme enough to
fully justify the dormancy-based flag.  
**Business takeaway:** Indicates that strong past engagement can outweigh recent inactivity, making some apparently high-risk customers more likely to return than the model predicts.

**Pattern across all 5 false positives:** All have very high `recency_days` (133-262 days) and the
model leans heavily on this single feature. The false positives suggest recency alone
over-predicts churn for a subset of customers whose true purchase cadence is longer than the
180-day modeling window assumes.

---

## Part B: False Negatives (Predicted Safe, Actually Churned) - 23 total

These customers looked low-risk by the model but did not purchase again.  
**Business risk:** This is the more costly error type, these customers receive NO retention intervention at all and are silently lost, directly costing future revenue.

### 6. CUST00184: predicted probability 0.02 (very confident "safe"), but churned
**Profile:** Recency 14 days, Frequency 3, Monetary ₹2,457, last visit 6 days ago, looks like an
excellent, highly engaged customer.  
**Why the model missed this:** Every visible signal (recency, frequency, monetary, recent web
visit) looks strong. There is no evidence in our feature set of an impending churn.  
**Business takeaway:** This is the most concerning type of miss, a seemingly "great" customer who
vanishes with zero warning signs in the data. This may reflect an external cause (competitor
switch, personal circumstance) entirely outside our feature set.

### 7. CUST01990: predicted probability 0.05, but churned
**Profile:** Recency 59 days, Frequency 4, Monetary ₹3,878 (one of the highest spenders in the
test set), last visit 7 days ago.  
**Why missed:** Strong historical engagement across every metric.  
**Business takeaway:** Losing a customer of this value with no warning is a real business risk. This case argues for monitoring even "safe" high-value customers with lightweight, low-cost check-ins (e.g., satisfaction surveys) rather than only acting on model flags.

### 8. CUST01655: predicted probability 0.07, but churned
**Profile:** Recency 13 days, Frequency 2, Monetary ₹1,359, last visit 7 days ago.  
**Why missed:** Again, all near-term signals look healthy.  
**Business takeaway:** Reinforces that some churn is essentially unpredictable from this feature
set alone which is sudden, not gradual.

### 9. CUST00838: predicted probability 0.09, but churned
**Profile:** Recency 9 days, Frequency 1, Monetary ₹403, negative-ticket rate 100% (had a
negative support interaction), last visit 12 days ago.  
**Why missed:** The negative-ticket signal was present but this customer's low ticket *count*
(implying it may have been a single, recent negative ticket) combined with otherwise-strong
recency wasn't enough to push the probability up.  
**Business takeaway:** This is the one false negative with an actual visible warning sign in our
data (the negative ticket) that the model under-weighted. Worth flagging to the support team
directly: a single very negative interaction shortly before snapshot may deserve more proactive
follow-up regardless of model score.

### 10. CUST01303: predicted probability 0.09, but churned
**Profile:** Recency 20 days, Frequency 1, Monetary ₹845, last visit 0 days ago (visited the site
on the snapshot date itself).  
**Why missed:** Very recent web visit and order, looks like an active, recently-engaged customer.  
**Business takeaway:** A site visit doesn't guarantee a future purchase; this case shows the model
slightly over-trusts very recent web activity as a safety signal.  

**Pattern across all 5 false negatives:** Every one of these customers had strong, recent
engagement signals. There was essentially no advance warning visible in the available features.
This means a portion of churn in this dataset is **not predictable from behavioral history alone**, likely driven by external factors (competitor offers, life changes, one-off bad experiences not yet reflected in ticket volume) that aren't captured by any of our data sources.

---

## Overall Business Implications

1. **False positives cluster around customers with naturally long purchase cycles:** the model
   may be overly aggressive flagging anyone with `recency_days` > 150, even when other signals are
   fine. A refinement could segment by historical purchase frequency before applying recency-based
   risk.
2. **False negatives are mostly "sudden" churners with no visible warning:** this is a structural
   limitation, not a model tuning problem; no feature set built only from internal behavioral data
   can fully predict externally-driven churn.
3. **The model should be combined with, not replace, human judgment:** particularly for high-value
   customers like CUST01990 (₹3,878 spend) who slipped through undetected. A secondary, lightweight
   "high value customer health check" process independent of the model score is recommended for
   the top monetary decile regardless of predicted risk.
