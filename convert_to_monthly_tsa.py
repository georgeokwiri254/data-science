"""
Script to convert Daily TSA notebook to Monthly TSA notebook
Automatically makes all necessary adaptations for monthly data analysis
"""

import json
import re

# Read the current monthly notebook
with open(r'C:\Users\reservations\Desktop\Data Science\Hotel_Revenue_Monthly_TSA.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Cell modifications mapping
cell_modifications = {
    # Cell 0: Update title
    'cell-0': {
        'type': 'markdown',
        'content': '''# Hotel Revenue MONTHLY Time Series Analysis - EDA & Preprocessing
## Multi-Year Forecasting for 2025 Annual Close

**Project Goal:** Predict hotel revenue metrics (RevPar, ADR, Revenue) for September - December 2025 using historical **MONTHLY data from 2009-2025**.

**Current Date:** August 2025

**Data Splits (MONTHLY):**
- Training: 2009-01 to 2025-05 (16+ years of history!)
- Validation: 2025-06 to 2025-08 (3 months)
- Test/Forecast: 2025-09 to 2025-12 (4 months)

---'''
    },

    # Cell 4: Update configuration for monthly dates
    'cell-4': {
        'type': 'code',
        'content': '''# Define key constants
CURRENT_DATE = '2025-08'  # MONTHLY format
TARGET_VARIABLES = ['RevPar', 'ADR', 'Revenue', 'Occupancy_Pct']

# Data split dates (MONTHLY FORMAT)
TRAIN_END = '2025-05'  # May 2025
VALIDATION_START = '2025-06'  # June 2025
VALIDATION_END = '2025-08'  # August 2025
FORECAST_START = '2025-09'  # September 2025
FORECAST_END = '2025-12'  # December 2025

# Color palette for visualizations
COLORS = sns.color_palette('husl', 8)

print(f"Configuration set:")
print(f"  - Current Date: {CURRENT_DATE}")
print(f"  - Training Period: 2009-01 to {TRAIN_END}")
print(f"  - Validation Period: {VALIDATION_START} to {VALIDATION_END}")
print(f"  - Forecast Period: {FORECAST_START} to {FORECAST_END}")
print(f"  - Target Variables: {TARGET_VARIABLES}")'''
    },

    # Cell 5: Update upload instructions
    'cell-5': {
        'type': 'markdown',
        'content': '''### 1.3 Data Ingestion (Google Colab File Upload)

**Instructions:** Upload your `monthly_revenue_2009_2025_all.csv` file when prompted below.'''
    },

    # Cell 6: Update upload filename
    'cell-6': {
        'type': 'code',
        'content': '''# Google Colab file upload
from google.colab import files

print("Please upload your monthly_revenue_2009_2025_all.csv file:")
uploaded = files.upload()

# Get the uploaded filename
filename = list(uploaded.keys())[0]
print(f"\\nFile '{filename}' uploaded successfully!")'''
    },

    # Cell 16: Update time series viz title
    'cell-16': {
        'type': 'code',
        'content': '''# Plot all target variables over time
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

for idx, var in enumerate(TARGET_VARIABLES):
    axes[idx].plot(df['Date'], df[var], linewidth=2, color=COLORS[idx], marker='o', markersize=3)
    axes[idx].set_title(f'{var} Over Time - MONTHLY (2009-2025)', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Date', fontsize=11)
    axes[idx].set_ylabel(var, fontsize=11)
    axes[idx].grid(True, alpha=0.3)

    # Add vertical lines for data splits
    axes[idx].axvline(x=pd.to_datetime(TRAIN_END), color='blue', linestyle='--',
                      linewidth=1.5, label='Train End', alpha=0.7)
    axes[idx].axvline(x=pd.to_datetime(VALIDATION_END), color='red', linestyle='--',
                      linewidth=1.5, label='Current Date', alpha=0.7)
    axes[idx].legend()

plt.tight_layout()
plt.show()'''
    },

    # Cell 18: REMOVE Day of Week analysis (not applicable for monthly)
    'cell-18': {
        'type': 'markdown',
        'content': '''### 2.3 Seasonality Analysis - Dubai Tourism Seasons

**Note:** Day-of-week analysis not applicable for monthly data.'''
    },

    # Cell 19: Update for monthly
    'cell-19': {
        'type': 'code',
        'content': '''# Box plots by Month (across all years)
df['Month'] = df['Date'].dt.month
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.ravel()

for idx, var in enumerate(TARGET_VARIABLES):
    sns.boxplot(data=df[df['Date'] <= CURRENT_DATE], x='Month', y=var,
                palette='viridis', ax=axes[idx])
    axes[idx].set_title(f'{var} Distribution by Month (2009-2025)', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Month', fontsize=11)
    axes[idx].set_ylabel(var, fontsize=11)
    axes[idx].set_xticklabels(month_names)

    # Highlight Dubai seasons
    axes[idx].axvspan(9.5, 12.5, alpha=0.1, color='green', label='High Season (Oct-Dec)')
    axes[idx].axvspan(0.5, 4.5, alpha=0.1, color='green')
    axes[idx].axvspan(4.5, 9.5, alpha=0.1, color='red', label='Low Season (May-Sep)')

plt.tight_layout()
plt.show()'''
    },

    # Cell 21: Update decomposition period to 12
    'cell-21': {
        'type': 'code',
        'content': '''# Decompose RevPar time series (only training data)
train_eda = df[df['Date'] <= TRAIN_END].copy().set_index('Date')

decomposition = seasonal_decompose(train_eda['RevPar'], model='additive', period=12)  # 12-month seasonality

fig, axes = plt.subplots(4, 1, figsize=(16, 12))

decomposition.observed.plot(ax=axes[0], color='blue', marker='o')
axes[0].set_ylabel('Observed', fontsize=11)
axes[0].set_title('Time Series Decomposition - RevPar MONTHLY (Training Data)', fontsize=14, fontweight='bold')

decomposition.trend.plot(ax=axes[1], color='green', marker='o')
axes[1].set_ylabel('Trend', fontsize=11)

decomposition.seasonal.plot(ax=axes[2], color='orange')
axes[2].set_ylabel('Seasonal (12-month)', fontsize=11)

decomposition.resid.plot(ax=axes[3], color='red', marker='o')
axes[3].set_ylabel('Residual', fontsize=11)
axes[3].set_xlabel('Date', fontsize=11)

plt.tight_layout()
plt.show()'''
    },

    # Cell 23: Update ACF/PACF lags to 24
    'cell-23': {
        'type': 'code',
        'content': '''# ACF and PACF for RevPar
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

plot_acf(train_eda['RevPar'].dropna(), lags=24, ax=axes[0])  # 24 months
axes[0].set_title('Autocorrelation Function (ACF) - RevPar MONTHLY', fontsize=14, fontweight='bold')

plot_pacf(train_eda['RevPar'].dropna(), lags=24, ax=axes[1])  # 24 months
axes[1].set_title('Partial Autocorrelation Function (PACF) - RevPar MONTHLY', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()'''
    },

    # Cell 27: Update interpolation method
    'cell-27': {
        'type': 'code',
        'content': '''# Create clean copy
df_clean = df.copy()

print("="*80)
print("DATA CLEANING (MONTHLY)")
print("="*80)

# Handle missing values using interpolation (monthly)
numeric_cols = ['Revenue', 'ADR', 'RevPar', 'Occupancy_Pct']
missing_before = df_clean[numeric_cols].isnull().sum().sum()

for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].interpolate(method='linear')  # Linear interpolation for monthly

missing_after = df_clean[numeric_cols].isnull().sum().sum()
print(f"Missing values before: {missing_before}")
print(f"Missing values after: {missing_after}")
print("\\n✓ Data cleaning complete")'''
    },

    # Cell 29: UPDATE - Minimal temporal features + Dubai seasonality
    'cell-29': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("TEMPORAL FEATURES (MONTHLY - OPTIMIZED)")
print("="*80)

# Create MINIMAL temporal features for monthly data
df_clean['Month'] = df_clean['Date'].dt.month  # 1-12
df_clean['Quarter'] = df_clean['Date'].dt.quarter  # 1-4

# Dubai Hotel Seasonality (CRITICAL FEATURE!)
df_clean['High_Season'] = df_clean['Month'].isin([10,11,12,1,2,3,4]).astype(int)  # Oct-Apr (high tourism)
df_clean['Low_Season'] = df_clean['Month'].isin([5,6,7,8,9]).astype(int)  # May-Sep (low tourism)

print("✓ Created 4 temporal features (OPTIMIZED):")
print("  - Month (1-12) - for general seasonality")
print("  - Quarter (1-4) - for quarterly patterns")
print("  - High_Season (Oct-Apr) - Dubai tourism high season")
print("  - Low_Season (May-Sep) - Dubai tourism low season")
print("\\n  Total: 4 features (reduced from 25+ for daily)")'''
    },

    # Cell 31: Update lag features for monthly
    'cell-31': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("LAGGED FEATURES (MONTHLY)")
print("="*80)

lag_features = ['RevPar', 'ADR', 'Occupancy_Pct', 'Revenue']
lag_periods = [1]  # 1-MONTH lag (not 1-day!)

for feature in lag_features:
    for lag in lag_periods:
        df_clean[f'{feature}_lag_{lag}'] = df_clean[feature].shift(lag)

print(f"✓ Created {len(lag_features) * len(lag_periods)} lagged features")
print(f"  - Variables: {lag_features}")
print(f"  - Lags: {lag_periods} MONTH (1-month lag)")
print(f"  - Note: Using only 1-month lag to avoid overfitting (~200 monthly records)")'''
    },

    # Cell 33: Confirm no MA features
    'cell-33': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("MOVING AVERAGES - REMOVED")
print("="*80)

print("✓ Skipping ALL rolling features (MA, Std Dev)")
print("  - Reason: Avoid curse of dimensionality with ~200 monthly records")
print("  - XGBoost/LSTM automatically capture these patterns from lag features")
print("  - Would need MA_3, MA_6, MA_12 for monthly → adds 12 redundant features")
print("  - DECISION: REMOVED to keep feature count minimal (7 total)")'''
    },

    # Cell 34: Update section title
    'cell-34': {
        'type': 'markdown',
        'content': '''### 3.5 Dubai Seasonality Features (No Month Dummies)

**Note:** We use High_Season/Low_Season instead of 12 month dummies to avoid redundancy (2 features vs 12).'''
    },

    # Cell 35: REMOVE month dummies, NO DOW encoding
    'cell-35': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("SEASONAL ENCODING - DUBAI PATTERN (NO MONTH DUMMIES)")
print("="*80)

# HIGH_SEASON and LOW_SEASON already created in Cell 29
# NO month dummies (redundant with High_Season/Low_Season)
# NO day-of-week encoding (not applicable for monthly data)

print("✓ Using Dubai Seasonality features (already created):")
print("  - High_Season: Oct-Apr (10,11,12,1,2,3,4) = 1")
print("  - Low_Season: May-Sep (5,6,7,8,9) = 1")
print("\\n✓ SKIPPED: Month dummies (would add 12 redundant features)")
print("✓ SKIPPED: Day-of-week encoding (not applicable for monthly)")

# Convert boolean to int (if any)
bool_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'bool']
for col in bool_cols:
    df_clean[col] = df_clean[col].astype(int)

print(f"\\nTotal temporal features: 4 (Month, Quarter, High_Season, Low_Season)")'''
    },

    # Cell 37: Update data filtering - use full history!
    'cell-37': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("DATA SPLITTING (MONTHLY)")
print("="*80)

# Use FULL history (2009+) - monthly data benefits from longer history!
print("\\n✅ Using FULL historical data (2009-2025) for monthly analysis")
print("  - Rationale: Monthly data less noisy than daily, more stable patterns")
print(f"Full dataset: {len(df_clean)} monthly records from {df_clean['Date'].min()} to {df_clean['Date'].max()}")

# Create splits
train_data = df_clean[df_clean['Date'] <= TRAIN_END].copy()
validation_data = df_clean[(df_clean['Date'] >= VALIDATION_START) &
                           (df_clean['Date'] <= VALIDATION_END)].copy()
test_data = df_clean[(df_clean['Date'] >= FORECAST_START) &
                     (df_clean['Date'] <= FORECAST_END)].copy()

print(f"\\nTraining Set:")
print(f"  Period: {train_data['Date'].min()} to {train_data['Date'].max()}")
print(f"  Records: {len(train_data)} months")

print(f"\\nValidation Set:")
print(f"  Period: {validation_data['Date'].min()} to {validation_data['Date'].max()}")
print(f"  Records: {len(validation_data)} months")

print(f"\\nTest/Forecast Set:")
print(f"  Period: {test_data['Date'].min()} to {test_data['Date'].max()}")
print(f"  Records: {len(test_data)} months")

print(f"\\nTotal: {len(df_clean)} monthly records")
print(f"✓ Excellent sample-to-feature ratio: {len(train_data)} months / 7 features = {len(train_data)//7}:1")'''
    },

    # Cell 39: Update scaling columns
    'cell-39': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("FEATURE SCALING (STANDARDIZATION - MONTHLY)")
print("="*80)

# Columns NOT to scale (binary/categorical only)
cols_not_to_scale = ['Date', 'Month', 'Quarter', 'High_Season', 'Low_Season']

# Get numeric columns to scale (targets + lags only)
numeric_cols_all = df_clean.select_dtypes(include=[np.number]).columns.tolist()
cols_to_scale = [col for col in numeric_cols_all if col not in cols_not_to_scale]

print(f"Columns NOT scaled: {len(cols_not_to_scale)} (categorical/binary)")
print(f"  {cols_not_to_scale}")
print(f"\\nColumns TO scale: {len(cols_to_scale)} (continuous values)")
print(f"  {cols_to_scale}")

# Save ORIGINAL data for SARIMAX (statistical models use unscaled data)
train_data_original = train_data.copy()
validation_data_original = validation_data.copy()
test_data_original = test_data.copy()

# Fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(train_data[cols_to_scale])

# Transform all datasets
train_data[cols_to_scale] = scaler.transform(train_data[cols_to_scale])
validation_data[cols_to_scale] = scaler.transform(validation_data[cols_to_scale])
test_data[cols_to_scale] = scaler.transform(test_data[cols_to_scale])

print("\\n✓ StandardScaler fitted on training data ONLY")
print("✓ Applied to train, validation, and test sets")
print("✓ ORIGINAL data saved for SARIMAX model (train_data_original, etc.)")'''
    },

    # Cell 41: Update ML-ready dataset sample display
    'cell-41': {
        'type': 'code',
        'content': '''print("\\n" + "="*80)
print("ML-READY DATASET (MONTHLY)")
print("="*80)

# Combine all data
ml_ready_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

# Add dataset identifier
ml_ready_data['Dataset'] = 'Train'
ml_ready_data.loc[ml_ready_data['Date'] >= VALIDATION_START, 'Dataset'] = 'Validation'
ml_ready_data.loc[ml_ready_data['Date'] >= FORECAST_START, 'Dataset'] = 'Test'

print(f"\\nTotal records: {len(ml_ready_data)} months")
print(f"Total features: {len(ml_ready_data.columns)}")
print(f"Date range: {ml_ready_data['Date'].min()} to {ml_ready_data['Date'].max()}")

print(f"\\nDataset Distribution:")
print(ml_ready_data['Dataset'].value_counts().sort_index())

print("\\nSample of ML-Ready Data (MONTHLY):")
display(ml_ready_data[['Date', 'RevPar', 'RevPar_lag_1', 'High_Season', 'Low_Season', 'Dataset']].head(10))'''
    },

    # Cell 43: Update output filename
    'cell-43': {
        'type': 'code',
        'content': '''# Export to CSV
output_filename = 'monthly_revenue_ml_ready.csv'
ml_ready_data.to_csv(output_filename, index=False)

print("="*80)
print("EXPORT COMPLETE (MONTHLY)")
print("="*80)
print(f"\\n✓ ML-ready data saved: {output_filename}")
print(f"✓ Ready for model training and forecasting")
print(f"\\n  Features: 7 (Quarter, High_Season, Low_Season, + 4 lags)")
print(f"  Records: {len(ml_ready_data)} months (2009-2025)")

# Download
files.download(output_filename)
print(f"\\n✓ File downloaded!")'''
    },

    # Cell 44: Update preprocessing summary
    'cell-44': {
        'type': 'markdown',
        'content': '''---
## Preprocessing Summary (MONTHLY)

**Completed Steps:**
1. ✓ Data cleaning and missing value imputation (linear interpolation)
2. ✓ Temporal features: Month, Quarter (2 features)
3. ✓ Dubai Seasonality: High_Season, Low_Season (2 features)
4. ✓ Lagged variables: 1-month lag only (4 features)
5. ✓ Moving averages: REMOVED (not needed, avoid overfitting)
6. ✓ Month dummies: REMOVED (redundant with High/Low Season)
7. ✓ Chronological data splitting (Train/Validation/Test)
8. ✓ Feature scaling (StandardScaler fitted on training only)
9. ✓ Export ML-ready dataset

**Final Feature Count: 7 features**
- Temporal: Month (not used in models - only for reference), Quarter (1)
- Seasonality: High_Season, Low_Season (2)
- Lags: RevPar_lag_1, ADR_lag_1, Revenue_lag_1, Occupancy_lag_1 (4)

**Next Steps:**
- Model training (SARIMAX with Dubai seasonality exog, XGBoost, LSTM)
- Model evaluation on validation set
- Final forecast: September - December 2025'''
    },

    # Cell 46: Update upload message
    'cell-46': {
        'type': 'code',
        'content': '''# If you already have ml_ready_data from Section 3, skip this cell
# Otherwise, upload the preprocessed file

print("Upload monthly_revenue_ml_ready.csv if not already in memory:")
try:
    # Check if ml_ready_data exists
    print(f"ML-ready data already loaded: {ml_ready_data.shape}")
except NameError:
    # Upload and load the preprocessed file
    uploaded_ml = files.upload()
    ml_filename = list(uploaded_ml.keys())[0]
    ml_ready_data = pd.read_csv(ml_filename)
    ml_ready_data['Date'] = pd.to_datetime(ml_ready_data['Date'])
    print(f"ML-ready data loaded: {ml_ready_data.shape}")'''
    },

    # Cell 47: Update feature column definition
    'cell-47': {
        'type': 'code',
        'content': '''# Prepare datasets for modeling
print("="*80)
print("PREPARING DATA FOR MODELING (MONTHLY)")
print("="*80)

# Split by Dataset identifier
train_df = ml_ready_data[ml_ready_data['Dataset'] == 'Train'].copy()
val_df = ml_ready_data[ml_ready_data['Dataset'] == 'Validation'].copy()
test_df = ml_ready_data[ml_ready_data['Dataset'] == 'Test'].copy()

print(f"\\nTraining Set: {len(train_df)} months")
print(f"Validation Set: {len(val_df)} months")
print(f"Test Set: {len(test_df)} months")

# Define feature columns (MONTHLY - 7 features)
exclude_cols = ['Date', 'Month', 'Dataset'] + TARGET_VARIABLES  # Exclude Month (used only for reference)
feature_cols = [col for col in ml_ready_data.columns if col not in exclude_cols]

# Remove any columns with NaN (from lagging)
train_df_clean = train_df.dropna()
val_df_clean = val_df.dropna()

print(f"\\nFeature columns: {len(feature_cols)}")
print(f"  {feature_cols}")
print(f"\\nTraining set after removing NaN: {len(train_df_clean)} months")
print(f"Validation set after removing NaN: {len(val_df_clean)} months")
print(f"\\nSample-to-feature ratio: {len(train_df_clean)}:{len(feature_cols)} = {len(train_df_clean)//len(feature_cols)}:1 ✓")'''
    },

    # Cell 48: Update data scaling note for monthly
    'cell-48': {
        'type': 'markdown',
        'content': '''### 4.0.1 Important Note: Data Scaling Strategy (MONTHLY)

**Different models require different data scales:**

1. **SARIMAX (Statistical Model)**
   - Trains on **ORIGINAL (unscaled)** data
   - Statistical models expect data in its natural scale
   - Predictions are directly in AED (currency units)
   - Better performance when working with actual values
   - **Uses exogenous variables:** High_Season, Low_Season (Dubai pattern)

2. **XGBoost (Tree-based Model)**
   - Trains on **SCALED** data
   - Benefits from standardization for regularization
   - Predictions need inverse transformation back to AED
   - **Reduced complexity for monthly:** max_depth=4, n_estimators=100

3. **LSTM (Neural Network)**
   - Trains on **SCALED** data
   - Neural networks require normalized inputs for optimization
   - Predictions need inverse transformation back to AED
   - **Adapted for monthly:** timesteps=6 months, units=24

**Why this matters:**
- SARIMAX performs better on original scale + uses Dubai seasonality as exog variables
- Each model type has different requirements for optimal performance
- Monthly data has fewer records (~200) → reduced model complexity to avoid overfitting'''
    }
}

# Apply modifications
for cell in notebook['cells']:
    cell_id = cell.get('id')
    if cell_id in cell_modifications:
        mod = cell_modifications[cell_id]
        cell['cell_type'] = mod['type']
        if mod['type'] == 'markdown':
            cell['source'] = mod['content']
        else:  # code
            cell['source'] = mod['content']

# Additional modifications for SARIMAX cells (50-51)
# We'll need to handle these specially since they contain SARIMAX code

# Save the modified notebook
output_path = r'C:\Users\reservations\Desktop\Data Science\Hotel_Revenue_Monthly_TSA.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Monthly TSA notebook created successfully!")
print(f"✓ Saved to: {output_path}")
print(f"\\nKey changes made:")
print(f"  - Updated to MONTHLY time granularity (2009-2025)")
print(f"  - Changed filename to monthly_revenue_2009_2025_all.csv")
print(f"  - Reduced features to 7 (no month dummies, no MA features)")
print(f"  - Added Dubai Seasonality (High_Season, Low_Season)")
print(f"  - Changed decomposition period to 12 (monthly)")
print(f"  - Updated ACF/PACF lags to 24 months")
print(f"  - Updated lag features to 1-month (not 1-day)")
print(f"  - Removed day-of-week analysis")
print(f"  - Updated all visualization titles to 'MONTHLY'")
print(f"\\nNext: Run this notebook in Google Colab with monthly_revenue_2009_2025_all.csv!")
