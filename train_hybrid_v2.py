"""
HYBRID SEISMIC WARNING SYSTEM - FIXED VERSION
==============================================
Uses cross-validation instead of time-series split
to handle limited event data better.
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


def train_hybrid_system():
    print("\n" + "="*70)
    print(" HYBRID SEISMIC WARNING SYSTEM - FIXED")
    print("="*70)
    
    os.makedirs('trained_models', exist_ok=True)
    
    # ==========================================
    # LOAD DATA
    # ==========================================
    print("\n📊 Loading data...")
    seismic_df = pd.read_csv('../seismic_events_1.csv')
    operations_df = pd.read_csv('../operational_metrics.csv')
    
    operations_df['recorded_at'] = pd.to_datetime(operations_df['recorded_at'])
    seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'])
    operations_df = operations_df.sort_values('recorded_at').reset_index(drop=True)
    operations_df = operations_df.ffill().bfill()
    
    print(f"✅ Loaded {len(operations_df)} operational records")
    print(f"✅ Loaded {len(seismic_df)} seismic events")
    
    # ==========================================
    # REMOVE SHUTDOWN PERIODS
    # ==========================================
    print("\n⚠️  Removing shutdown periods...")
    
    shutdown_periods = [
        ('2019-07-01', '2021-04-30'),
        ('2022-12-01', '2023-03-31'),
        ('2023-10-01', '2023-11-30'),
        ('2025-07-01', '2025-09-30'),
    ]
    
    for start, end in shutdown_periods:
        mask = ~((operations_df['recorded_at'] >= start) & 
                 (operations_df['recorded_at'] <= end))
        operations_df = operations_df[mask]
        
        mask = ~((seismic_df['occurred_at'] >= start) & 
                 (seismic_df['occurred_at'] <= end))
        seismic_df = seismic_df[mask]
    
    operations_df = operations_df.reset_index(drop=True)
    seismic_df = seismic_df.reset_index(drop=True)
    
    print(f"   Remaining: {len(operations_df)} operational records")
    print(f"   Remaining: {len(seismic_df)} events ({(seismic_df['magnitude'] > 0.5).sum()} significant)")
    
    # ==========================================
    # CREATE DAILY AGGREGATED DATA
    # ==========================================
    print("\n📊 Creating DAILY aggregated data...")
    
    operations_df['date'] = operations_df['recorded_at'].dt.date
    
    # Aggregate to daily level
    daily_ops = operations_df.groupby('date').agg({
        'inj_flow': ['mean', 'max', 'std'],
        'inj_whp': ['mean', 'max', 'std'],
        'prod_flow': ['mean', 'max', 'std'],
        'prod_whp': ['mean', 'max', 'std'],
        'inj_temp': ['mean', 'max'],
        'prod_temp': ['mean', 'max'],
        'is_producing': 'mean'
    }).reset_index()
    
    # Flatten column names
    daily_ops.columns = ['date'] + [f'{a}_{b}' for a, b in daily_ops.columns[1:]]
    daily_ops['date'] = pd.to_datetime(daily_ops['date'])
    
    # Add derived features
    daily_ops['pressure_diff_mean'] = daily_ops['inj_whp_mean'] - daily_ops['prod_whp_mean']
    daily_ops['pressure_diff_max'] = daily_ops['inj_whp_max'] - daily_ops['prod_whp_max']
    daily_ops['flow_imbalance_mean'] = daily_ops['inj_flow_mean'] - daily_ops['prod_flow_mean']
    daily_ops['temp_diff_mean'] = daily_ops['prod_temp_mean'] - daily_ops['inj_temp_mean']
    
    # Add rolling features (7-day window)
    for col in ['inj_whp_max', 'inj_flow_max', 'pressure_diff_max']:
        daily_ops[f'{col}_7d_max'] = daily_ops[col].rolling(7, min_periods=1).max()
        daily_ops[f'{col}_7d_mean'] = daily_ops[col].rolling(7, min_periods=1).mean()
    
    # Add rate of change
    daily_ops['inj_whp_change'] = daily_ops['inj_whp_mean'].diff()
    daily_ops['inj_flow_change'] = daily_ops['inj_flow_mean'].diff()
    
    daily_ops = daily_ops.fillna(0)
    
    print(f"   Created {len(daily_ops)} daily records")
    
    # ==========================================
    # CREATE DAILY LABELS
    # ==========================================
    print("\n🎯 Creating daily labels...")
    
    # Mark days with significant events (M > 0.5)
    seismic_df['date'] = seismic_df['occurred_at'].dt.date
    significant_events = seismic_df[seismic_df['magnitude'] > 0.5]
    
    # Create label: 1 if event within next 48 hours
    daily_ops['has_event'] = 0
    
    for _, event in significant_events.iterrows():
        event_date = pd.Timestamp(event['date'])
        
        # Mark 2 days BEFORE event (prediction window)
        for days_before in range(0, 3):  # 0, 1, 2 days before
            target_date = event_date - pd.Timedelta(days=days_before)
            mask = daily_ops['date'] == target_date
            daily_ops.loc[mask, 'has_event'] = 1
    
    n_positive = daily_ops['has_event'].sum()
    n_negative = len(daily_ops) - n_positive
    
    print(f"   Days with upcoming events: {n_positive}")
    print(f"   Days without events: {n_negative}")
    print(f"   Class ratio: {n_negative/n_positive:.1f}:1")
    
    # ==========================================
    # PREPARE FEATURES
    # ==========================================
    feature_cols = [c for c in daily_ops.columns if c not in ['date', 'has_event']]
    
    X = daily_ops[feature_cols].values
    y = daily_ops['has_event'].values
    
    print(f"\n📊 Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    print(f"   Positive: {y.sum()} ({y.mean()*100:.1f}%)")
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ==========================================
    # CROSS-VALIDATION
    # ==========================================
    print("\n🤖 Training with 5-Fold Cross-Validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    fold = 0
    for train_idx, test_idx in cv.split(X_scaled, y):
        fold += 1
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model with balanced weights
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.3).astype(int)  # Lower threshold
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
        # Fold metrics
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"   Fold {fold}: P={prec:.1%}, R={rec:.1%}, TP={tp}, FP={fp}, FN={fn}")
    
    # ==========================================
    # OVERALL METRICS
    # ==========================================
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    tp = ((all_y_pred == 1) & (all_y_true == 1)).sum()
    fp = ((all_y_pred == 1) & (all_y_true == 0)).sum()
    fn = ((all_y_pred == 0) & (all_y_true == 1)).sum()
    tn = ((all_y_pred == 0) & (all_y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Enrichment
    base_rate = all_y_true.mean()
    pred_rate = all_y_true[all_y_pred == 1].mean() if (all_y_pred == 1).sum() > 0 else 0
    enrichment = pred_rate / base_rate if base_rate > 0 else 0
    
    print("\n" + "="*70)
    print("📊 OVERALL CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"""
   ┌─────────────────────────────────────────┐
   │  MODEL: Random Forest Classifier        │
   │  METHOD: 5-Fold Cross-Validation        │
   │  THRESHOLD: 0.3                         │
   ├─────────────────────────────────────────┤
   │  PRECISION:  {precision:.1%}                       │
   │  RECALL:     {recall:.1%}                       │
   │  F1 SCORE:   {f1:.3f}                       │
   │  ENRICHMENT: {enrichment:.1f}x                        │
   ├─────────────────────────────────────────┤
   │  TP: {tp:5d}  (Correctly predicted events) │
   │  FP: {fp:5d}  (False alarms)               │
   │  FN: {fn:5d}  (Missed events)              │
   │  TN: {tn:5d}  (Correctly predicted safe)   │
   └─────────────────────────────────────────┘
    """)
    
    # ==========================================
    # TRAIN FINAL MODEL ON ALL DATA
    # ==========================================
    print("\n🤖 Training final model on all data...")
    
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_scaled, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 10 Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']:30s}: {row['importance']:.4f}")
    
    # ==========================================
    # CALCULATE RULE-BASED THRESHOLDS
    # ==========================================
    print("\n📋 Calculating rule-based thresholds...")
    
    # Get operational conditions during event days
    event_days = daily_ops[daily_ops['has_event'] == 1]
    
    thresholds = {
        'inj_whp_max': {
            'red': event_days['inj_whp_max'].quantile(0.75),
            'yellow': event_days['inj_whp_max'].quantile(0.50),
            'unit': 'MPa'
        },
        'inj_flow_max': {
            'red': event_days['inj_flow_max'].quantile(0.75),
            'yellow': event_days['inj_flow_max'].quantile(0.50),
            'unit': 'L/s'
        },
        'pressure_diff_max': {
            'red': event_days['pressure_diff_max'].quantile(0.75),
            'yellow': event_days['pressure_diff_max'].quantile(0.50),
            'unit': 'MPa'
        }
    }
    
    print("\n   Thresholds (from event days):")
    for param, vals in thresholds.items():
        print(f"   {param}:")
        print(f"      🔴 RED:    > {vals['red']:.2f} {vals['unit']}")
        print(f"      🟡 YELLOW: > {vals['yellow']:.2f} {vals['unit']}")
    
    # ==========================================
    # SAVE
    # ==========================================
    model_data = {
        'model': final_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'thresholds': thresholds,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'enrichment': enrichment,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        },
        'feature_importance': importance_df.to_dict(),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('trained_models/hybrid_system.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n💾 Saved to trained_models/hybrid_system.pkl")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    
    return model_data


if __name__ == "__main__":
    train_hybrid_system()