# Model Card: D2C Customer Churn Prediction Model

## 1. Model Details

- **Model type:** XGBoost gradient-boosted tree classifier, wrapped in a scikit-learn `Pipeline`
  (median imputation + standard scaling for numeric features, constant-fill + one-hot encoding
  for categorical features).
- **Version:** v1.0
- **Trained on:** `rfm_modeling_snapshot.csv` (1,728 training customers from the pre-assigned
  `split` column)
- **Saved as:** `model.pkl` (joblib-serialized full pipeline, including preprocessing, the
  evaluator/API does not need to repeat feature engineering)

## 2. Intended Use

**Primary use case:** Score existing customers (as of a snapshot date) with a probability of
churning (no purchase) in the next 60 days, to help the CRM and retention teams prioritize which
customers receive proactive outreach.

**Intended users:** Internal CRM tooling, retention/marketing team analysts, customer support
team.

**Out of scope / NOT intended for:**
- Determining individual customer value or worth as a person, this model predicts a behavioral
  outcome only.
- Automated decisions with no human review (e.g., automatically denying support to "high churn
  risk" customers), predictions should inform retention outreach, not punitive action.
- Use on customer populations meaningfully different from this D2C personal-care brand's customer
  base (e.g., a different industry, country, or business model) without retraining.
- Real-time scoring during checkout or browsing, this model is snapshot-based (uses 30/90/180-day
  trailing windows) and is not designed for moment-to-moment prediction.

## 3. Data Used

- **Source:** `rfm_modeling_snapshot.csv`, a feature-engineered table built from customer
  profile, order history (180-day window, excluding post-snapshot orders), support tickets
  (90-day window), and web/app activity (30-day window), all as of snapshot date 2025-09-30.
- **Target:** `churn_next_60d`, if the customer made no purchase in the 60 days following the
  snapshot (2025-10-01 to 2025-11-29), else 0.
- **Train/validation/test split:** Pre-assigned (1,728 / 336 / 336 customers) and used as-is for
  reproducibility and comparability across runs.
- **Leakage prevention:** All features are computed using only data available on or before the
  snapshot date. We explicitly verified that `churn_next_60d` and `split` are excluded from the
  feature set, and that all underlying raw data (orders, tickets, web events) used in
  constructing this snapshot respects the snapshot-date cutoff.

## 4. Model Approach

- **Baseline:** Logistic Regression (ROC-AUC ≈ 0.88 on validation)- included for interpretability
  and as a sanity-check lower bound.
- **Final model:** XGBoost (200 estimators, max_depth=4, learning_rate=0.05)- chosen for its
  ability to capture non-linear feature interactions (e.g., recency combined with low web activity)
  without manual feature engineering, and for providing feature importance.
- **Threshold:** 0.40 (below the default 0.5), chosen because the business cost of missing a
  churner (lost future revenue) is judged higher than the cost of an unnecessary low-cost retention
  touch on a customer who would have stayed.

## 5. Performance (Test Set, 336 Customers)

| Metric | Value |
|---|---:|
| ROC-AUC | 0.868 |
| PR-AUC | 0.840 |
| Accuracy | 0.801 |
| Precision | 0.767 |
| Recall | 0.863 |
| F1-score | 0.812 |

**Confusion matrix:** TN=124, FP=44, FN=23, TP=145 (see `metrics.json` for the full export).

**Top features driving predictions:** `recency_days` (dominant), `negative_ticket_rate_90d`,
`frequency_180d`, `monetary_180d`, `return_rate_180d`, and web-engagement signals
(`wishlist_adds_30d`, `cart_adds_30d`, `campaign_clicks_30d`).

## 6. Limitations

- **Recency-dominant behavior:** The model leans heavily on `recency_days`, which causes
  false-positive flags on customers with naturally long, infrequent purchase cycles. It does not distinguish "churning" from "long natural repurchase cycle."
- **Cannot predict externally-driven churn:** All 5 false-negative examples analyzed had strong,
  healthy engagement signals right up to the snapshot date. The model has no visibility into
  external factors (competitor switching, personal circumstances, one-off bad experiences not yet
  reflected in support data).
- **Static snapshot model:** Trained on a single snapshot date; customer behavior patterns,
  seasonality, or business changes after this snapshot are not reflected and may degrade
  performance over time (see `monitoring_plan.md` in Part 4 for drift-detection guidance).
- **Small support-ticket signal for most customers:** ~48% of customers have zero support
  tickets, so ticket-based features are zero/uninformative for nearly half the customer base;
  the model relies more heavily on order and web-activity features for these customers.

## 7. Ethical Risks

- **Risk of over-targeting low-value, high-risk customers with costly interventions:** The model
  alone doesn't account for customer value; it must be used alongside Part 2's segment-based
  budget tiers (e.g., Dormant customers get only cheap channels regardless of churn probability)
  to avoid wasting high-cost retention spend on customers with low historical value.
- **Risk of customers feeling surveilled or singled out:** If retention outreach explicitly
  references "we noticed you haven't visited in X days," some customers may find this
  uncomfortable. Outreach messaging should be framed positively (e.g., new product recommendations)
  rather than exposing the underlying behavioral tracking.
- **Risk of self-fulfilling discount dependency:** Repeatedly targeting at-risk customers with
  discounts could train customers to delay purchases until they receive an offer, inflating churn
  risk artificially over time. This connects to the Discount-Sensitive segment risk flagged in
  Part 2.
- **No protected-class features used:** The model does not use gender or any explicitly protected
  demographic attribute beyond `age_group` (a broad bracket) and `city_tier` (market
  classification, not a protected class), both included only because they were part of the
  provided feature set and showed legitimate behavioral correlation, not because they are intended
  as proxies for protected characteristics. If retraining, consider testing for disparate impact
  across `age_group` and `acquisition_channel` segments before deployment.

## 8. Monitoring Needs

In summary: track prediction-score distribution drift, feature distribution drift (especially `recency_days`), actual vs. predicted churn rate by segment, and retrain when validation performance on fresh snapshots degrades meaningfully (e.g., ROC-AUC drop > 0.05 from this baseline of 0.868).

## 9. When This Model Should NOT Be Used

- Do not use this model's score as the sole basis for denying a customer service, discounts, or
  account access, it predicts a behavioral outcome only, with the limitations above.
- Do not use this model on a different product line, market, or business (e.g., B2B, a different
  country, or a different industry) without retraining and revalidating on representative data.
- Do not treat a single customer's score as ground truth, according to the error analysis, individual predictions can be wrong in both directions; the model is intended for **prioritization at scale**, not individual certainty.
