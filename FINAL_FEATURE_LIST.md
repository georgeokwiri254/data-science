# Monthly TSA - Final Feature List (6 Features)

## ✅ CONFIRMED: 6 Features (Quarter Removed)

### Feature Breakdown:

#### 1. Dubai Seasonality (2 features)
```python
High_Season = 1 if Month in [10,11,12,1,2,3,4] else 0  # Oct-Apr
Low_Season = 1 if Month in [5,6,7,8,9] else 0          # May-Sep
```

**Why these matter:**
- Oct-Apr: Cooler weather, international tourists, peak rates
- May-Sep: Hot summer, domestic focus, lower rates
- Directly captures hotel revenue pattern in Dubai

#### 2. Lag Features - 1 Month (4 features)
```python
RevPar_lag_1 = RevPar from previous month
ADR_lag_1 = ADR from previous month
Revenue_lag_1 = Revenue from previous month
Occupancy_Pct_lag_1 = Occupancy from previous month
```

**Why 1-month lag only:**
- ~200 monthly records → keep features minimal
- Immediate previous month most predictive
- Adding lag_3, lag_6, lag_12 → overfitting risk

---

## ❌ Features Explicitly REMOVED:

### Quarter - REMOVED (Your Correction!)
**Reason:** Redundant with High_Season/Low_Season

**Problem with Quarter:**
```
Q1 (Jan-Mar) = 100% High Season
Q2 (Apr-Jun) = 33% High, 67% Low ← SPLIT!
Q3 (Jul-Sep) = 100% Low Season
Q4 (Oct-Dec) = 100% High Season
```

Q2 splits the Dubai seasons → doesn't capture pattern cleanly!

High/Low Season is superior:
- Clean separation (no split months)
- Directly models Dubai tourism cycle
- 2 features vs 4 quarter dummies

### Month Dummies (12 features) - REMOVED
**Reason:** Redundant with High_Season/Low_Season
- Would add 12 features for same information
- High/Low captures monthly pattern with just 2 features

### Moving Averages (MA_3, MA_6, MA_12) - REMOVED
**Reason:**
- Multicollinearity with lag features
- XGBoost/LSTM learn these patterns automatically
- Overfitting risk with ~200 records

### Rolling Std Dev - REMOVED
**Reason:** Same as MA (redundant, multicollinear)

### Day-of-Week Features - REMOVED
**Reason:** Not applicable for monthly granularity

### Year - REMOVED
**Reason:** Chronological order captures time progression

### Additional Lags (lag_3, lag_6, lag_12) - REMOVED
**Reason:** Limited data (~200 months) → overfitting risk

---

## Complete Feature List for Models:

### Features Used in Training (6):
1. `High_Season` - Binary (0/1)
2. `Low_Season` - Binary (0/1)
3. `RevPar_lag_1` - Continuous (AED)
4. `ADR_lag_1` - Continuous (AED)
5. `Revenue_lag_1` - Continuous (AED)
6. `Occupancy_Pct_lag_1` - Continuous (%)

### Features NOT Used in Training (2):
- `Date` - Index/reference only
- `Month` - Reference for visualization only (not in feature_cols)

### Target Variables (4):
- `RevPar`
- `ADR`
- `Revenue`
- `Occupancy_Pct`

---

## Data Scaling:

### Columns to Scale (8):
```python
cols_to_scale = [
    'RevPar', 'ADR', 'Revenue', 'Occupancy_Pct',           # Targets (4)
    'RevPar_lag_1', 'ADR_lag_1', 'Revenue_lag_1', 'Occupancy_Pct_lag_1'  # Lags (4)
]
```

### Columns NOT to Scale (4):
```python
cols_not_to_scale = [
    'Date',          # DateTime index
    'Month',         # Reference only
    'High_Season',   # Binary (0/1) - already normalized
    'Low_Season'     # Binary (0/1) - already normalized
]
```

---

## Sample-to-Feature Ratio:

**Training Records:** ~190 months (2009-01 to 2025-05)
**Features:** 6
**Ratio:** 190 / 6 = **31.6:1** ✅

**Industry Standard:** 10:1 minimum
**Our Ratio:** 31:1 → **Excellent!** Low overfitting risk.

---

## Model Configurations:

### 1. SARIMAX
```python
SARIMAX(
    train_original['RevPar'],  # ORIGINAL scale
    exog=train_original[['High_Season', 'Low_Season']],  # 2 exog vars
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
```

### 2. XGBoost
```python
X_train = train_scaled[[
    'High_Season', 'Low_Season',
    'RevPar_lag_1', 'ADR_lag_1', 'Revenue_lag_1', 'Occupancy_Pct_lag_1'
]]  # 6 features, SCALED

XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3
)
```

### 3. LSTM
```python
input_shape = (6, 6)  # 6 timesteps (months), 6 features

Sequential([
    LSTM(24, input_shape=(6, 6)),  # SCALED data
    Dropout(0.3),
    Dense(12),
    Dense(4)
])
```

---

## Code Updates Required:

### Cell 29 - Create Features:
```python
# Create ONLY essential features
df_clean['Month'] = df_clean['Date'].dt.month  # Reference only

# Dubai Seasonality
df_clean['High_Season'] = df_clean['Month'].isin([10,11,12,1,2,3,4]).astype(int)
df_clean['Low_Season'] = df_clean['Month'].isin([5,6,7,8,9]).astype(int)

# DO NOT create Quarter!
```

### Cell 39 - Scaling:
```python
cols_not_to_scale = ['Date', 'Month', 'High_Season', 'Low_Season']
# Quarter removed from this list!
```

### Cell 47 - Feature Columns:
```python
exclude_cols = ['Date', 'Month', 'Dataset'] + TARGET_VARIABLES
feature_cols = [col for col in ml_ready_data.columns if col not in exclude_cols]

# Should output 6 features:
# ['High_Season', 'Low_Season', 'RevPar_lag_1', 'ADR_lag_1', 'Revenue_lag_1', 'Occupancy_Pct_lag_1']
```

---

## Verification Checklist:

- [ ] Cell 29: Quarter NOT created
- [ ] Cell 29: High_Season and Low_Season created
- [ ] Cell 31: Only 1-month lags created (4 total)
- [ ] Cell 33: NO moving averages
- [ ] Cell 35: NO month dummies
- [ ] Cell 39: cols_not_to_scale does NOT include Quarter
- [ ] Cell 47: feature_cols has exactly 6 features
- [ ] Cell 50: SARIMAX exog has ['High_Season', 'Low_Season']
- [ ] Cell 53: XGBoost uses 6 features
- [ ] Cell 57: LSTM input_shape = (6, 6)

---

## Expected Output When Running:

```
Feature columns: 6
  ['High_Season', 'Low_Season', 'RevPar_lag_1', 'ADR_lag_1', 'Revenue_lag_1', 'Occupancy_Pct_lag_1']

Training set after removing NaN: ~190 months
Validation set after removing NaN: 3 months

Sample-to-feature ratio: 190:6 = 31:1 ✓
```

---

## Why This is Optimal:

1. **Minimal Features:** 6 vs 25+ → Less overfitting
2. **Domain Knowledge:** High/Low Season captures Dubai pattern
3. **No Redundancy:** Each feature adds unique information
4. **Strong Ratio:** 31:1 sample-to-feature ratio
5. **Clean Separation:** Dubai seasons don't split across quarters

**Bottom Line:** Simple, interpretable, effective! 🎯

---

**Last Updated:** December 7, 2025 (Quarter removed per user feedback)
**Status:** Final - Ready for implementation
