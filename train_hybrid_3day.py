"""
HYBRID SEISMIC WARNING SYSTEM — 3-DAY WINDOW VERSION
====================================================
Uses sliding 3-day windows to learn multi-day patterns.
Predicts if a significant event will occur within 48 hours.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

seismic_threshold = 0.17


def create_3day_windows(daily_df, label_col):
    """
    Convert daily features into sliding 3-day windows.
    Example:
      X(t) = [features_t-2 , features_t-1 , features_t]
      y(t) = label at day t
    """
    X, y = [], []
    feature_cols = [c for c in daily_df.columns if c not in ['date', label_col]]

    for i in range(2, len(daily_df)):
        window = []
        # Concatenate features from day i-2, i-1, i
        for j in range(i - 2, i + 1):
            window.extend(daily_df.iloc[j][feature_cols].values)

        X.append(window)
        y.append(daily_df.iloc[i][label_col])

    return np.array(X), np.array(y), feature_cols


def train_hybrid_system_3day():
    print("\n" + "="*70)
    print(" HYBRID WARNING SYSTEM — 3-DAY WINDOW MODEL")
    print("="*70)

    os.makedirs('trained_models', exist_ok=True)

    # =====================================================
    # LOAD DATA
    # =====================================================
    print("\n Loading data...")
    seismic_df = pd.read_csv('../seismic_events_1.csv')
    operations_df = pd.read_csv('../operational_metrics.csv')

    operations_df['recorded_at'] = pd.to_datetime(operations_df['recorded_at'])
    seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'])
    operations_df = operations_df.sort_values('recorded_at').reset_index(drop=True)
    operations_df = operations_df.ffill().bfill()

    # =====================================================
    # REMOVE SHUTDOWN PERIODS (VERY IMPORTANT)
    # =====================================================
    print("\n  Removing shutdown periods...")

    shutdown_periods = [
        ('2019-07-01', '2021-04-30'),
        ('2022-12-01', '2023-03-31'),
        ('2023-10-01', '2023-11-30'),
        ('2025-07-01', '2025-09-19'),
    ]

    # Remove from operations and seismic
    for start, end in shutdown_periods:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        operations_df = operations_df[
            ~((operations_df['recorded_at'] >= start) &
            (operations_df['recorded_at'] <= end))
        ]

        seismic_df = seismic_df[
            ~((seismic_df['occurred_at'] >= start) &
            (seismic_df['occurred_at'] <= end))
        ]

    operations_df = operations_df.reset_index(drop=True)
    seismic_df = seismic_df.reset_index(drop=True)

    print(f"   Remaining operational days: {operations_df.shape[0]}")
    print(f"   Remaining seismic events:   {seismic_df.shape[0]}")

    # =====================================================
    # AGGREGATE TO DAILY LEVEL
    # =====================================================
    print("\n Aggregating to daily features...")

    operations_df['date'] = operations_df['recorded_at'].dt.date

    daily = operations_df.groupby('date').agg({
        'inj_flow': ['mean','max','std'],
        'inj_whp': ['mean','max','std'],
        'prod_flow': ['mean','max','std'],
        'prod_whp': ['mean','max','std'],
        'inj_temp': ['mean','max'],
        'prod_temp': ['mean','max'],
        'is_producing': 'mean'
    }).reset_index()

    daily.columns = ['date'] + [f"{a}_{b}" for a,b in daily.columns[1:]]
    daily['date'] = pd.to_datetime(daily['date'])

    # Derived features
    daily['pressure_diff'] = daily['inj_whp_mean'] - daily['prod_whp_mean']
    daily['flow_imbalance'] = daily['inj_flow_mean'] - daily['prod_flow_mean']
    daily['temp_diff'] = daily['prod_temp_mean'] - daily['inj_temp_mean']

    # Rolling 7-day stats
    for col in ['inj_whp_max','flow_imbalance','pressure_diff']:
        daily[f'{col}_7d_max'] = daily[col].rolling(7, min_periods=1).max()
        daily[f'{col}_7d_mean'] = daily[col].rolling(7, min_periods=1).mean()

    # Rate of change
    daily['inj_whp_change'] = daily['inj_whp_mean'].diff()
    daily['inj_flow_change'] = daily['inj_flow_mean'].diff()

    daily = daily.fillna(0)

    print(f" Daily records: {len(daily)}")

    # =====================================================
    # CREATE LABELS — EVENT IN NEXT 48 HOURS
    # =====================================================
    print("\n Creating labels...")

    seismic_df['date'] = seismic_df['occurred_at'].dt.date
    significant = seismic_df[seismic_df['magnitude'] > seismic_threshold]

    daily['has_event'] = 0

    for _, ev in significant.iterrows():
        ev_date = pd.Timestamp(ev['date'])

        for d in range(0, 3):   # event day, day-1, day-2
            target = ev_date - pd.Timedelta(days=d)
            daily.loc[daily['date'] == target, 'has_event'] = 1

    print(f" Positive days: {daily['has_event'].sum()}")

    # =====================================================
    # CREATE 3-DAY WINDOWS
    # =====================================================
    print("\n Creating 3-day sliding windows...")

    X, y, feature_cols = create_3day_windows(daily, 'has_event')
    print(f"Samples: {len(X)}, Features per sample: {X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =====================================================
    # CROSS-VALIDATION
    # =====================================================
    print("\n 5-Fold Cross-Validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []

    fold = 1
    for train_idx, test_idx in cv.split(X_scaled, y):
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled[train_idx], y[train_idx])

        y_proba = model.predict_proba(X_scaled[test_idx])[:,1]
        y_pred = (y_proba >= 0.3).astype(int)

        precision = precision_score(y[test_idx], y_pred)
        recall    = recall_score(y[test_idx], y_pred)
        f1        = f1_score(y[test_idx], y_pred)

        print(f" Fold {fold}:  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        fold += 1

        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)

    # Final metrics
    precision = precision_score(y_true_all, y_pred_all)
    recall    = recall_score(y_true_all, y_pred_all)
    f1        = f1_score(y_true_all, y_pred_all)

    print("\n" + "="*70)
    print(" 📊 FINAL CROSS-VALIDATION METRICS")
    print("="*70)
    print(f" Precision: {precision:.3f}")
    print(f" Recall:    {recall:.3f}")
    print(f" F1 Score:  {f1:.3f}")

    # =====================================================
    # TRAIN FINAL MODEL
    # =====================================================
    print("\nTraining final model...")

    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_scaled, y)

    # Save everything
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open('trained_models/hybrid_3day.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("\nSaved to trained_models/hybrid_3day.pkl")
    print("\nDONE")

    return model_data



if __name__ == "__main__":
    train_hybrid_system_3day()
