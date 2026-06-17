# Part 3: Churn Prediction Model & Model Card

## D2C Customer Churn Intelligence Capstone Project (Part 3 of 4)

This repository builds, evaluates, and documents a churn-prediction model that identifies
customers likely to churn in the next 60 days.

---

## Contents

| File | Description |
|---|---|
| `churn_model.ipynb` | Full modeling workflow: leakage checks, train/val/test split, baseline (Logistic Regression) + stronger model (XGBoost), evaluation, threshold selection, error analysis exploration, feature importance, model saving |
| `model.pkl` | Final trained model: a scikit-learn `Pipeline` (preprocessing + XGBoost classifier), saved with `joblib` |
| `metrics.json` | Final test-set metrics (ROC-AUC, PR-AUC, accuracy, precision, recall, F1, confusion matrix, threshold) |
| `error_analysis.md` | 10 specific customer examples (5 false positives, 5 false negatives) with business-risk interpretation |
| `model_card.md` | Structured model card: intended use, data, approach, performance, limitations, ethical risks, monitoring needs |
| `charts/` | 4 saved chart images |
| `data/` | Raw dataset CSVs |
| `requirements.txt` | Python dependencies |

---

## How to Run

1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and run all cells:
   ```bash
   jupyter notebook churn_model.ipynb
   ```
   All paths are relative, so it runs from the repo root unmodified. Running the notebook
   regenerates `model.pkl`, `metrics.json`, and all charts from scratch.

---

## Loading the Saved Model

```python
import joblib
model = joblib.load('model.pkl')

# model is a full sklearn Pipeline (preprocessing + XGBoost) 

import pandas as pd
sample = pd.DataFrame([{
    'recency_days': 45, 'frequency_180d': 2, 'monetary_180d': 1200.0,
    'return_rate_180d': 0.1, 'avg_discount_pct_180d': 0.2, 'avg_rating_180d': 4.2,
    'category_diversity_180d': 2, 'ticket_count_90d': 0, 'negative_ticket_rate_90d': 0.0,
    'avg_resolution_hours_90d': 0.0, 'days_since_signup': 200, 'sessions_30d': 5,
    'product_views_30d': 10, 'cart_adds_30d': 2, 'wishlist_adds_30d': 1,
    'abandoned_carts_30d': 0, 'email_opens_30d': 3, 'campaign_clicks_30d': 1,
    'last_visit_days_ago': 10, 'city_tier': 'Tier 1', 'age_group': '25-34',
    'acquisition_channel': 'Instagram', 'loyalty_tier': 'Gold',
    'preferred_category': 'Skin Care', 'marketing_consent': 'Yes'
}])
proba = model.predict_proba(sample)[:,1]
print(proba)
```

---

## Methodology Summary

- **Data source:** `rfm_modeling_snapshot.csv`, pre-built to exclude any post-snapshot
  (2025-10-01 onward) information -- verified explicitly in the notebook's leakage-check section.
- **Split:** Pre-assigned `train` (1,728) / `validation` (336) / `test` (336), used as-is.
- **Models:** Logistic Regression (baseline) and XGBoost (final model).
- **Final test performance:** ROC-AUC 0.868, Precision 0.767, Recall 0.863, F1 0.812 at a
  business-justified decision threshold of 0.40.
- **Top features:** `recency_days`, `negative_ticket_rate_90d`, `frequency_180d`,
  `monetary_180d`, `return_rate_180d`.

See `model_card.md` for full details, limitations, and ethical considerations, and
`error_analysis.md` for specific misclassified-customer breakdowns.
