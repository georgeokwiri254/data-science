# Hotel Revenue Time Series Analysis - Notebook Suite

## Created Notebooks

### 1. Hotel_Revenue_Daily_TSA.ipynb ✅
- **Data:** Daily granularity (combined_occupancy_chronological.csv)
- **Period:** 2022-2025 (post-COVID)
- **Records:** ~1,200 daily observations
- **Features:** 25+ (includes DOW, Is_Weekend, Month dummies, etc.)
- **Use Case:** Short-term daily forecasting, operational planning

### 2. Hotel_Revenue_Monthly_TSA.ipynb ⚠️ (Requires Manual Edits)
- **Data:** Monthly granularity (monthly_revenue_2009_2025_all.csv)
- **Period:** 2009-2025 (16+ years of history)
- **Records:** ~200 monthly observations
- **Features:** **7 OPTIMIZED** (no redundancy)
- **Use Case:** Strategic planning, long-term trends, annual forecasting

---

## Quick Start

### For Daily Analysis:
1. Open `Hotel_Revenue_Daily_TSA.ipynb` in Google Colab
2. Upload `combined_occupancy_chronological.csv`
3. Run all cells (no modifications needed)

### For Monthly Analysis:
1. Open `Hotel_Revenue_Monthly_TSA.ipynb` in Google Colab
2. **Follow edit instructions in `MONTHLY_TSA_IMPLEMENTATION_GUIDE.md`**
3. Upload `monthly_revenue_2009_2025_all.csv`
4. Run all cells after making edits

---

## Key Differences: Daily vs Monthly

| Aspect | Daily TSA | Monthly TSA |
|--------|-----------|-------------|
| **Granularity** | Days | Months |
| **File** | combined_occupancy_chronological.csv | monthly_revenue_2009_2025_all.csv |
| **History** | 2022-2025 (3 years) | 2009-2025 (16 years) |
| **Records** | ~1,200 | ~200 |
| **Features** | 25+ | **7** ✓ |
| **Sample:Feature Ratio** | 48:1 | **28:1** ✓ |
| **Month Dummies** | Yes (12) | **No** (use High/Low Season) |
| **Day-of-Week** | Yes (7+) | **No** (not applicable) |
| **Moving Averages** | Optional | **No** (overfitting risk) |
| **Lag Periods** | 1 day | 1 month |
| **Decomposition** | period=365 | **period=12** |
| **ACF/PACF Lags** | 60 | **24** |
| **SARIMA/X** | SARIMA s=7/30 | **SARIMAX s=12 + Dubai exog** |
| **XGBoost Depth** | 5 | **4** |
| **LSTM Timesteps** | 7 days | **6 months** |

---

## Monthly TSA: Optimized 7-Feature Strategy

### Why Only 7 Features?

**Problem:** ~200 monthly records with 25+ features = **overfitting risk**
**Solution:** Minimal, non-redundant features = **better generalization**

### Final Feature List (7 Total):

1. **Quarter** (1) - Quarterly seasonality
2. **High_Season** (1) - Dubai Oct-Apr tourism peak
3. **Low_Season** (1) - Dubai May-Sep off-season
4. **RevPar_lag_1** (1) - Previous month RevPar
5. **ADR_lag_1** (1) - Previous month ADR
6. **Revenue_lag_1** (1) - Previous month Revenue
7. **Occupancy_Pct_lag_1** (1) - Previous month Occupancy

### What We Removed & Why:

❌ **Month Dummies (12 features)** → Redundant with High_Season/Low_Season
❌ **Moving Averages (12 features)** → Multicollinearity + overfitting risk
❌ **Rolling Std Dev (12 features)** → Same issues as MA
❌ **Day-of-Week Features (7+ features)** → Not applicable for monthly
❌ **Year Column** → Chronological order captures time progression
❌ **Additional Lags (lag_3, lag_6, etc.)** → Limited data, overfitting risk

**Result:** From 50+ potential features → **7 optimal features**

---

## SARIMAX with Dubai Seasonality (NEW!)

### What is SARIMAX?

**SARIMA:** Seasonal AutoRegressive Integrated Moving Average
**SARIMAX:** SARIMA with eXogenous variables (external predictors)

### Why Use Exogenous Variables?

Traditional SARIMA only uses the target variable's history. SARIMAX adds external factors:

```python
SARIMAX(
    RevPar,  # Target variable
    exog=[High_Season, Low_Season],  # External Dubai seasonality!
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
)
```

**Benefit:** Explicitly models Dubai's tourism pattern (Oct-Apr high, May-Sep low)

### Dubai Tourism Seasons:

- **High Season (Oct-Apr):** Cooler weather, international tourists, higher prices
- **Low Season (May-Sep):** Hot summer, domestic focus, lower rates

**Implementation:**
```python
High_Season = 1 if Month in [10,11,12,1,2,3,4] else 0
Low_Season = 1 if Month in [5,6,7,8,9] else 0
```

---

## Model Configurations

### 1. SARIMAX (Statistical - Original Scale)
```python
SARIMAX(
    train_original['RevPar'],
    exog=train_original[['High_Season', 'Low_Season']],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
```
- **Data:** ORIGINAL (unscaled)
- **Exog:** High_Season, Low_Season
- **Output:** AED (direct)

### 2. XGBoost (Tree-Based - Scaled)
```python
XGBRegressor(
    n_estimators=100,  # Monthly: 100 (Daily: 200)
    max_depth=4,       # Monthly: 4 (Daily: 5)
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3
)
```
- **Data:** SCALED
- **Output:** Scaled → Inverse transform to AED

### 3. LSTM (Neural Network - Scaled)
```python
Sequential([
    LSTM(24, input_shape=(6, 7)),  # Monthly: 6 timesteps, 7 features
    Dropout(0.3),
    Dense(12),
    Dense(4)
])
```
- **Data:** SCALED
- **Timesteps:** 6 months (Daily: 7 days)
- **Output:** Scaled → Inverse transform to AED

---

## Data Scaling Strategy

### Why Different Scales?

Different model types have different mathematical requirements:

### SARIMAX → Original Scale
```python
train_sarimax = train_data_original['RevPar']  # Unscaled!
```
**Reason:** Statistical models assume stationarity in natural units

### XGBoost & LSTM → Scaled
```python
scaler = StandardScaler()
scaler.fit(train_data[cols_to_scale])
train_data_scaled = scaler.transform(train_data[cols_to_scale])
```
**Reason:** Regularization + gradient descent work better with normalized data

### Columns to Scale (8):
- RevPar, ADR, Revenue, Occupancy_Pct (targets)
- RevPar_lag_1, ADR_lag_1, Revenue_lag_1, Occupancy_Pct_lag_1 (lags)

### Columns NOT to Scale (5):
- Date, Month, Quarter, High_Season, Low_Season (binary/categorical)

---

## Files Reference

| File | Purpose |
|------|---------|
| `Hotel_Revenue_Daily_TSA.ipynb` | Daily analysis notebook (ready to use) |
| `Hotel_Revenue_Monthly_TSA.ipynb` | Monthly analysis template (needs edits) |
| `MONTHLY_TSA_IMPLEMENTATION_GUIDE.md` | **Step-by-step edit instructions** |
| `README_NOTEBOOKS.md` | This file (overview) |

---

## Next Steps

1. **Read:** `MONTHLY_TSA_IMPLEMENTATION_GUIDE.md` for detailed edit instructions
2. **Edit:** `Hotel_Revenue_Monthly_TSA.ipynb` following the guide
3. **Upload:** `monthly_revenue_2009_2025_all.csv` to Google Colab
4. **Run:** All cells and compare SARIMAX vs XGBoost vs LSTM
5. **Forecast:** Use best model for Sept-Dec 2025 predictions

---

## Expected Results

### Validation Metrics (Example):
- **SARIMAX:** RMSE ~50-80 AED, R² ~0.7-0.85
- **XGBoost:** RMSE ~40-70 AED, R² ~0.75-0.90
- **LSTM:** RMSE ~50-80 AED, R² ~0.70-0.85

*Actual results depend on data quality and patterns*

### Forecast Output:
- Monthly predictions for Sept-Dec 2025
- RevPar, ADR, Revenue, Occupancy_Pct
- Confidence intervals (from SARIMAX)
- Feature importance (from XGBoost)

---

## Troubleshooting

### "Too many features" error:
- Check that Month dummies were removed (Cell 35)
- Verify feature_cols has exactly 7 features (Cell 47)

### SARIMAX convergence issues:
- Ensure using ORIGINAL (unscaled) data
- Check exog variables are binary (0/1)
- Try simpler order: (1,1,1) instead of (2,1,2)

### XGBoost overfitting:
- Reduce max_depth to 3
- Reduce n_estimators to 75
- Increase min_child_weight to 5

### LSTM not learning:
- Check data is SCALED
- Reduce timesteps to 3
- Increase patience in EarlyStopping to 15

---

## Questions?

Refer to `MONTHLY_TSA_IMPLEMENTATION_GUIDE.md` for detailed cell-by-cell instructions.

Key concept: **7 optimized features > 25+ redundant features** for small monthly datasets!

---

**Created:** December 7, 2025
**Version:** 1.0
**Status:** Ready for implementation
