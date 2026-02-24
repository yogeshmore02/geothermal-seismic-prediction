import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler


def load_model(path='trained_models/hybrid_3day.pkl'):
    with open(path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def extract_daily_features(operations_df, debug=False):
    """Rebuild the exact same daily feature engineering used during training."""
    
    operations_df = operations_df.copy()
    
    if debug:
        print(f"  [DEBUG] Raw rows: {len(operations_df)}")
        print(f"  [DEBUG] First recorded_at: {operations_df['recorded_at'].iloc[0]}")
    
    # Robust date parsing - handles DD/MM/YYYY and other formats
    operations_df['recorded_at'] = pd.to_datetime(
        operations_df['recorded_at'], 
        format='mixed', 
        dayfirst=True
    )
    
    if debug:
        print(f"  [DEBUG] After parsing - Date range: {operations_df['recorded_at'].min()} to {operations_df['recorded_at'].max()}")
    
    operations_df = operations_df.sort_values('recorded_at').reset_index(drop=True)
    operations_df = operations_df.ffill().bfill()

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

    # Rolling 7-day statistics
    for col in ['inj_whp_max','flow_imbalance','pressure_diff']:
        daily[f'{col}_7d_max'] = daily[col].rolling(7, min_periods=1).max()
        daily[f'{col}_7d_mean'] = daily[col].rolling(7, min_periods=1).mean()

    # Rate-of-change features
    daily['inj_whp_change'] = daily['inj_whp_mean'].diff()
    daily['inj_flow_change'] = daily['inj_flow_mean'].diff()

    daily = daily.fillna(0)
    
    if debug:
        print(f"  [DEBUG] Daily rows: {len(daily)}")
        print(f"  [DEBUG] Daily date range: {daily['date'].min()} to {daily['date'].max()}")
    
    return daily


def build_3day_window(daily, feature_cols, target_date=None, debug=False):
    """
    Build a 3-day feature window ending on target_date.
    
    Returns:
        tuple: (X array, actual_date used)
    """
    # Reset index to ensure positional indexing works correctly
    daily = daily.copy().reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    # ----------------------------------------------------------
    # CASE 1 — Use the last 3 days (when no target_date specified)
    # ----------------------------------------------------------
    if target_date is None:
        if len(daily) < 3:
            raise ValueError("Need at least 3 daily records to build prediction window.")
        
        last3 = daily.iloc[-3:]
        window = []
        for i in range(len(last3)):
            window.extend(last3.iloc[i][feature_cols].values)
        
        actual_date = daily.iloc[-1]["date"]
        return np.array(window).reshape(1, -1), actual_date

    # ----------------------------------------------------------
    # CASE 2 — Use 3 days ending on target_date
    # ----------------------------------------------------------
    target_date = pd.to_datetime(target_date)
    
    if debug:
        print(f"  [DEBUG] Looking for target_date: {target_date} (type: {type(target_date)})")
    
    # Find matching date - compare date parts only
    daily_dates = daily["date"].dt.date
    target_date_only = target_date.date()
    
    if debug:
        print(f"  [DEBUG] target_date_only: {target_date_only}")
        print(f"  [DEBUG] Sample daily_dates: {list(daily_dates[:5])}")
        print(f"  [DEBUG] Target in daily_dates: {target_date_only in daily_dates.values}")
    
    mask = daily_dates == target_date_only
    
    if debug:
        print(f"  [DEBUG] Mask sum (matches found): {mask.sum()}")
    
    if not mask.any():
        available = sorted(daily_dates.unique())
        raise ValueError(
            f"Target date {target_date_only} not found in data. "
            f"Available dates: {available[0]} to {available[-1]}"
        )

    # Get positional index (not label) using np.where
    positions = np.where(mask)[0]
    pos = positions[0]
    
    if debug:
        print(f"  [DEBUG] Position found: {pos}")

    if pos < 2:
        raise ValueError(
            f"Not enough previous days before {target_date_only} to form a 3-day window. "
            f"Need at least 2 days before the target date."
        )

    # Get rows: day-2, day-1, day0 (target) using positional indexing
    rows = daily.iloc[pos-2 : pos+1]
    
    if debug:
        print(f"  [DEBUG] 3-day window dates: {rows['date'].tolist()}")

    window = []
    for i in range(len(rows)):
        window.extend(rows.iloc[i][feature_cols].values)
    
    X = np.array(window).reshape(1, -1)
    
    if debug:
        print(f"  [DEBUG] Window shape: {X.shape}")
        print(f"  [DEBUG] Window sum: {X.sum()}")
        print(f"  [DEBUG] Window has NaN: {np.isnan(X).any()}")
        print(f"  [DEBUG] Window min/max: {X.min():.4f} / {X.max():.4f}")

    return X, target_date


def predict_warning(operations_csv, target_date=None, model_path="trained_models/hybrid_3day.pkl", debug=False):
    """
    Predict seismic risk for an optional specified date.
    """
    try:
        # Load trained model
        model_data = load_model(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_cols']
        
        if debug:
            print(f"  [DEBUG] Model feature_cols: {len(feature_cols)} features")
            print(f"  [DEBUG] Expected total features: {len(feature_cols) * 3}")

        # Load new operational data
        operations_df = pd.read_csv(operations_csv)
        daily = extract_daily_features(operations_df, debug=debug)

        # Build 3-day window - now returns (X, actual_date)
        X, date_used = build_3day_window(daily, feature_cols, target_date=target_date, debug=debug)

        if debug:
            print(f"  [DEBUG] X shape before scaling: {X.shape}")

        # Scale and predict
        X_scaled = scaler.transform(X)
        
        if debug:
            print(f"  [DEBUG] X_scaled sum: {X_scaled.sum():.4f}")
        
        proba = model.predict_proba(X_scaled)
        
        if debug:
            print(f"  [DEBUG] predict_proba output: {proba}")
        
        event_proba = proba[0][1]
        print(event_proba)
        event_pred = 0
        if (event_proba >= 0.6):
            event_pred = 2
        elif (event_proba >= 0.3):
            event_pred = 1

        return {
            "date": date_used,
            "event_probability": float(event_proba),
            "warning_flag": event_pred,
            "error": None
        }
    
    except Exception as e:
        import traceback
        if debug:
            traceback.print_exc()
        return {
            "date": target_date,
            "event_probability": 0.0,
            "warning_flag": 0,
            "error": str(e)
        }


if __name__ == "__main__":
    print("=== Test with DEBUG for 2019-06-15 ===\n")
    result = predict_warning("operational_metrics.csv", target_date="2019-06-15", debug=True)
    print(f"\n🎯 RESULT: {result}")
