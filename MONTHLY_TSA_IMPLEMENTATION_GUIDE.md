# Hotel Revenue MONTHLY Time Series Analysis - Implementation Guide

## Files Created

1. **Hotel_Revenue_Daily_TSA.ipynb** - Exact copy of original (daily granularity)
2. **Hotel_Revenue_Monthly_TSA.ipynb** - Template for monthly analysis (needs manual edits below)

---

## Manual Edits Required for Monthly TSA Notebook

### Critical Changes Summary

**File to Upload:** `monthly_revenue_2009_2025_all.csv` (instead of combined_occupancy_chronological.csv)

**Final Feature Count:** **7 features** (optimized for ~200 monthly records)
- Quarter (1)
- High_Season, Low_Season (2) - Dubai tourism seasons
- RevPar_lag_1, ADR_lag_1, Revenue_lag_1, Occupancy_Pct_lag_1 (4)

**NO Month Dummies** - Redundant with High/Low Season (saves 12 features!)
**NO Moving Averages** - Prevents overfitting with small monthly dataset
**NO Day-of-Week Features** - Not applicable for monthly data

---

## Cell-by-Cell Edit Instructions

### Cell 0 (Title) - EDIT

```markdown
# Hotel Revenue MONTHLY Time Series Analysis - EDA & Preprocessing
## Multi-Year Forecasting for 2025 Annual Close

**Project Goal:** Predict hotel revenue metrics for September - December 2025 using historical **MONTHLY data from 2009-2025**.

**Current Date:** August 2025

**Data Splits (MONTHLY):**
- Training: 2009-01 to 2025-05 (16+ years!)
- Validation: 2025-06 to 2025-08 (3 months)
- Test/Forecast: 2025-09 to 2025-12 (4 months)
```

### Cell 4 (Configuration) - EDIT

```python
CURRENT_DATE = '2025-08'  # MONTHLY format
TARGET_VARIABLES = ['RevPar', 'ADR', 'Revenue', 'Occupancy_Pct']

# Data split dates (MONTHLY)
TRAIN_END = '2025-05'
VALIDATION_START = '2025-06'
VALIDATION_END = '2025-08'
FORECAST_START = '2025-09'
FORECAST_END = '2025-12'

COLORS = sns.color_palette('husl', 8)

print(f"  - Training Period: 2009-01 to {TRAIN_END}")  # Change this line
```

### Cell 5-6 (Data Upload) - EDIT

```python
print("Please upload your monthly_revenue_2009_2025_all.csv file:")  # Change filename
```

### Cell 18 (Day of Week Boxplots) - DELETE ENTIRE CELL

Day-of-week analysis not applicable for monthly data.

### Cell 21 (Decomposition) - EDIT

```python
decomposition = seasonal_decompose(train_eda['RevPar'], model='additive', period=12)  # Change to 12!
```

### Cell 23 (ACF/PACF) - EDIT

```python
plot_acf(train_eda['RevPar'].dropna(), lags=24, ax=axes[0])  # Change to 24
plot_pacf(train_eda['RevPar'].dropna(), lags=24, ax=axes[1])  # Change to 24
```

### Cell 29 (Temporal Features) - **CRITICAL EDIT**

```python
print("\n" + "="*80)
print("TEMPORAL FEATURES (MONTHLY - OPTIMIZED)")
print("="*80)

# MINIMAL temporal features
df_clean['Month'] = df_clean['Date'].dt.month  # For reference only
df_clean['Quarter'] = df_clean['Date'].dt.quarter

# DUBAI SEASONALITY (CRITICAL!)
df_clean['High_Season'] = df_clean['Month'].isin([10,11,12,1,2,3,4]).astype(int)  # Oct-Apr
df_clean['Low_Season'] = df_clean['Month'].isin([5,6,7,8,9]).astype(int)  # May-Sep

print("✓ Created 4 temporal features:")
print("  - Month (reference), Quarter")
print("  - High_Season (Oct-Apr), Low_Season (May-Sep)")
```

### Cell 31 (Lag Features) - EDIT

```python
lag_periods = [1]  # 1-MONTH lag (not 1-day!)

print(f"  - Lags: {lag_periods} MONTH")
```

### Cell 33 (Moving Averages) - EDIT

```python
print("✓ Skipping ALL rolling features (MA, Std Dev)")
print("  - Reason: Avoid overfitting with ~200 monthly records")
print("  - XGBoost captures patterns from lag features")
```

### Cell 34-35 (One-Hot Encoding) - **CRITICAL EDIT**

```markdown
### 3.5 Dubai Seasonality (No Month Dummies)
```

```python
print("\n" + "="*80)
print("SEASONAL ENCODING - DUBAI PATTERN")
print("="*80)

# High_Season and Low_Season already created in Cell 29
# NO month dummies - redundant!
# NO day-of-week encoding - not applicable for monthly!

print("✓ Using Dubai Seasonality (High_Season, Low_Season)")
print("✓ SKIPPED: Month dummies (redundant)")
print("✓ SKIPPED: Day-of-week encoding (not applicable)")

# Convert boolean to int
bool_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'bool']
for col in bool_cols:
    df_clean[col] = df_clean[col].astype(int)
```

### Cell 37 (Data Splitting) - EDIT

```python
# Use FULL history (2009-2025) - NO filtering!
print("✅ Using FULL historical data (2009-2025)")

# Remove this line: df_clean = df_clean[df_clean['Date'] >= '2022-01-01'].copy()

print(f"  Records: {len(train_data)} months")  # Change "records" to "months"
```

### Cell 39 (Scaling) - **CRITICAL EDIT**

```python
# Columns NOT to scale
cols_not_to_scale = ['Date', 'Month', 'Quarter', 'High_Season', 'Low_Season']

# Save ORIGINAL data for SARIMAX
train_data_original = train_data.copy()
validation_data_original = validation_data.copy()
test_data_original = test_data.copy()

# Rest of scaling code...

print("✓ ORIGINAL data saved for SARIMAX")
```

### Cell 47 (Feature Columns) - **CRITICAL EDIT**

```python
# Define feature columns (7 total)
exclude_cols = ['Date', 'Month', 'Dataset'] + TARGET_VARIABLES
feature_cols = [col for col in ml_ready_data.columns if col not in exclude_cols]

print(f"\nFeature columns: {len(feature_cols)}")
print(f"  {feature_cols}")  # Should show 7 features
```

### Cell 50 (SARIMAX) - **CRITICAL EDIT - Replace Entire Cell**

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*80)
print("BASELINE MODEL: SARIMAX (WITH DUBAI SEASONALITY)")
print("="*80)

# SARIMAX uses ORIGINAL (unscaled) data + exogenous variables
train_sarimax = train_data_original.set_index('Date')['RevPar']
train_exog = train_data_original[['High_Season', 'Low_Season']]

val_sarimax = validation_data_original.set_index('Date')['RevPar']
val_exog = validation_data_original[['High_Season', 'Low_Season']]

print(f"\nSARIMAX Training Data (Original Scale):")
print(f"  Records: {len(train_sarimax)} months")
print(f"  RevPar range: AED {train_sarimax.min():.2f} - {train_sarimax.max():.2f}")
print(f"  Exogenous vars: High_Season, Low_Season (Dubai pattern)")

# Train SARIMAX model with Dubai seasonality
sarimax_model = SARIMAX(
    train_sarimax,
    exog=train_exog,  # EXOGENOUS VARIABLES!
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),  # s=12 for monthly!
    enforce_stationarity=False,
    enforce_invertibility=False
)

print("\nTraining SARIMAX with Dubai seasonality exogenous variables...")
sarimax_fit = sarimax_model.fit(disp=False, maxiter=200)
print("✓ SARIMAX model trained")

# Predict with exogenous variables
sarimax_val_pred = sarimax_fit.forecast(steps=len(val_sarimax), exog=val_exog)

# Calculate metrics
sarimax_rmse = np.sqrt(mean_squared_error(val_sarimax, sarimax_val_pred))
sarimax_mae = mean_absolute_error(val_sarimax, sarimax_val_pred)
sarimax_r2 = r2_score(val_sarimax, sarimax_val_pred)
sarimax_mape = np.mean(np.abs((val_sarimax - sarimax_val_pred) / val_sarimax)) * 100

print(f"\nSARIMAX Validation Performance:")
print(f"  RMSE: AED {sarimax_rmse:.2f}")
print(f"  MAE: AED {sarimax_mae:.2f}")
print(f"  R²: {sarimax_r2:.4f}")
print(f"  MAPE: {sarimax_mape:.2f}%")

results = {
    'SARIMAX': {'RMSE': sarimax_rmse, 'MAE': sarimax_mae, 'R2': sarimax_r2, 'MAPE': sarimax_mape}
}
```

### Cell 51 - DELETE (duplicate SARIMA cell)

### Cell 53 (XGBoost) - EDIT

```python
# MONTHLY: Reduce complexity
xgb_model = xgb.XGBRegressor(
    n_estimators=100,  # REDUCED from 200
    max_depth=4,  # REDUCED from 5
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)
```

### Cell 57 (LSTM) - EDIT

```python
timesteps = 6  # CHANGE to 6 months (not 7 days!)

# MONTHLY: Reduce complexity
lstm_model = Sequential([
    LSTM(24, activation='tanh', return_sequences=False,  # REDUCED from 32
         input_shape=(timesteps, X_train.shape[1])),
    Dropout(0.3),
    Dense(12, activation='relu'),  # REDUCED from 16
    Dense(len(TARGET_VARIABLES))
])
```

---

## Feature Summary Table

| Feature Category | Count | Features | Usage |
|-----------------|-------|----------|-------|
| **Temporal** | 1 | Quarter | XGBoost, LSTM |
| **Dubai Seasonality** | 2 | High_Season, Low_Season | All models (SARIMAX exog!) |
| **Lag Features** | 4 | RevPar_lag_1, ADR_lag_1, Revenue_lag_1, Occupancy_lag_1 | XGBoost, LSTM |
| **Total** | **7** | - | **Optimal for ~200 monthly records** |

**Excluded:**
- ❌ Month dummies (12) - Redundant with High/Low Season
- ❌ Moving Averages (12) - Overfitting risk
- ❌ Day-of-week features (7+) - Not applicable monthly
- ❌ Year - Redundant with chronological order

**Sample-to-Feature Ratio:** ~200 months / 7 features = **28:1** ✓ Excellent!

---

## Data Scaling Strategy

### SARIMAX:
- **Input:** ORIGINAL (unscaled) data
- **Exog:** High_Season, Low_Season (original scale, binary 0/1)
- **Output:** AED (no transformation needed)

### XGBoost & LSTM:
- **Input:** SCALED data (StandardScaler)
- **Output:** SCALED → Inverse transform to AED

### Columns to Scale (8):
```python
cols_to_scale = [
    'RevPar', 'ADR', 'Revenue', 'Occupancy_Pct',  # Targets
    'RevPar_lag_1', 'ADR_lag_1', 'Revenue_lag_1', 'Occupancy_Pct_lag_1'  # Lags
]
```

### Columns NOT to Scale (5):
```python
cols_not_to_scale = [
    'Date', 'Month', 'Quarter', 'High_Season', 'Low_Season'  # Binary/categorical
]
```

---

## Model Configurations (Monthly-Optimized)

### 1. SARIMAX
```python
SARIMAX(
    train_original['RevPar'],
    exog=train_original[['High_Season', 'Low_Season']],  # Dubai seasonality!
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),  # s=12 for monthly
    enforce_stationarity=False,
    enforce_invertibility=False
)
```

**Key Points:**
- Uses ORIGINAL (unscaled) data
- Exogenous variables: Dubai High/Low seasons
- Seasonal period: 12 (monthly)

### 2. XGBoost (Reduced Complexity)
```python
XGBRegressor(
    n_estimators=100,  # ← Reduced from 200
    max_depth=4,       # ← Reduced from 5
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    random_state=42
)
```

**Rationale:** Monthly data has fewer records → lower complexity to prevent overfitting

### 3. LSTM (Adapted for Monthly)
```python
Sequential([
    LSTM(24, activation='tanh', input_shape=(6, 7)),  # 6 months history, 7 features
    Dropout(0.3),
    Dense(12, activation='relu'),
    Dense(4)  # 4 targets
])

timesteps = 6  # Use 6 months of history
```

**Rationale:** Monthly patterns evolve slower → 6-month lookback sufficient

---

## Quick Reference: What Changed from Daily to Monthly

| Aspect | Daily | Monthly |
|--------|-------|---------|
| **Data File** | combined_occupancy_chronological.csv | monthly_revenue_2009_2025_all.csv |
| **Date Range** | 2022-2025 (3 years) | 2009-2025 (16 years) |
| **Records** | ~1,200 days | ~200 months |
| **Features** | 25+ | **7** |
| **Decomposition Period** | 365 | **12** |
| **ACF/PACF Lags** | 60 | **24** |
| **Lag Periods** | [1] day | [1] **month** |
| **Moving Averages** | Removed | **Removed** |
| **Month Dummies** | 12 | **0** (use High/Low Season) |
| **DOW Features** | 7+ | **0** (not applicable) |
| **SARIMA** | s=7 or s=30 | **SARIMAX s=12 + exog** |
| **XGBoost n_estimators** | 200 | **100** |
| **XGBoost max_depth** | 5 | **4** |
| **LSTM timesteps** | 7 days | **6 months** |
| **LSTM units** | 32 | **24** |

---

## Final Checklist Before Running

- [ ] Cell 0: Updated title to "MONTHLY"
- [ ] Cell 4: Changed dates to monthly format (2009-01 to 2025-05, etc.)
- [ ] Cell 5-6: Changed filename to `monthly_revenue_2009_2025_all.csv`
- [ ] Cell 18: Deleted Day-of-Week boxplots
- [ ] Cell 21: Changed decomposition period to 12
- [ ] Cell 23: Changed ACF/PACF lags to 24
- [ ] Cell 29: Created High_Season and Low_Season (Dubai pattern)
- [ ] Cell 31: Changed lag_periods to [1] month
- [ ] Cell 35: Removed month dummies, removed DOW encoding
- [ ] Cell 37: Removed post-COVID filter (use full 2009-2025 history)
- [ ] Cell 39: Saved train_data_original, etc. for SARIMAX
- [ ] Cell 47: Verified feature_cols has 7 features
- [ ] Cell 50: Replaced with SARIMAX + exog (High_Season, Low_Season)
- [ ] Cell 51: Deleted duplicate SARIMA
- [ ] Cell 53: Reduced XGBoost complexity (n_estimators=100, max_depth=4)
- [ ] Cell 57: Updated LSTM (timesteps=6, units=24)

---

## Expected Output

**Feature Count:** 7 features
**Training Records:** ~190 months (2009-01 to 2025-05)
**Validation Records:** 3 months (2025-06 to 2025-08)
**Forecast:** 4 months (2025-09 to 2025-12)

**Sample-to-Feature Ratio:** 190/7 = 27:1 ✓ Excellent!

---

## Next Steps

1. Upload `monthly_revenue_2009_2025_all.csv` to Google Colab
2. Make the edits listed above to `Hotel_Revenue_Monthly_TSA.ipynb`
3. Run all cells sequentially
4. Compare SARIMAX (with Dubai seasonality) vs XGBoost vs LSTM
5. Use best model for final Sept-Dec 2025 forecast

---

## Questions & Answers

**Q: Why no month dummies?**
A: High_Season/Low_Season already captures monthly patterns (2 features vs 12), less overfitting risk.

**Q: Why full 2009-2025 history instead of just 2022+?**
A: Monthly data is smoother/less noisy than daily → longer history helps capture annual patterns.

**Q: Why only 1-month lag?**
A: With ~200 monthly records, adding more lags (3, 6, 12 months) increases overfitting risk. Keep it minimal.

**Q: Why remove Moving Averages?**
A: MA features are redundant with lag features + cause multicollinearity. XGBoost/LSTM learn these patterns automatically.

**Q: What's special about SARIMAX?**
A: It uses High_Season/Low_Season as exogenous variables → explicitly models Dubai tourism pattern!

---

**Created:** December 7, 2025
**Author:** Claude Code Assistant
**Notebook:** Hotel_Revenue_Monthly_TSA.ipynb
