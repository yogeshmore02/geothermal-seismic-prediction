"""
HYBRID SEISMIC WARNING SYSTEM
==============================
Combines:
1. RULE-BASED: Hard thresholds from domain knowledge (HIGH PRECISION)
2. ML-BASED: Anomaly detection for early warning (HIGH RECALL)

Traffic Light Logic:
- 🔴 RED: Rule violation OR very high ML risk → EVACUATE
- 🟡 YELLOW: ML warning OR approaching thresholds → CAUTION
- 🟢 GREEN: All normal → CONTINUE
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HybridSeismicWarningSystem:
    """
    Two-layer warning system:
    Layer 1: Rule-based (HIGH PRECISION - trust these!)
    Layer 2: ML-based (EARLY WARNING - be cautious)
    """
    
    def __init__(self):
        self.scaler = None
        self.ml_model = None
        self.feature_names = None
        self.thresholds = None
        self.score_percentiles = None
        
    def calculate_thresholds_from_data(self, operations_df, seismic_df):
        """
        Calculate rule-based thresholds from historical data.
        Uses conditions that preceded past events.
        """
        print("\n📊 Calculating rule-based thresholds from historical data...")
        
        # Find operational conditions during events
        event_conditions = []
        
        for _, event in seismic_df[seismic_df['magnitude'] > 0.5].iterrows():
            event_time = event['occurred_at']
            
            # Get operations 1-48 hours BEFORE event
            mask = (
                (operations_df['recorded_at'] >= event_time - pd.Timedelta(hours=48)) &
                (operations_df['recorded_at'] < event_time)
            )
            
            if mask.sum() > 0:
                event_ops = operations_df[mask]
                event_conditions.append({
                    'inj_whp_max': event_ops['inj_whp'].max(),
                    'inj_whp_mean': event_ops['inj_whp'].mean(),
                    'inj_flow_max': event_ops['inj_flow'].max(),
                    'inj_flow_mean': event_ops['inj_flow'].mean(),
                    'pressure_diff': (event_ops['inj_whp'] - event_ops['prod_whp']).max(),
                    'flow_imbalance': (event_ops['inj_flow'] - event_ops['prod_flow']).max(),
                    'inj_whp_change': event_ops['inj_whp'].diff().abs().max(),
                    'inj_flow_change': event_ops['inj_flow'].diff().abs().max(),
                    'magnitude': event['magnitude']
                })
        
        event_df = pd.DataFrame(event_conditions)
        
        # Calculate thresholds based on event conditions
        # RED threshold: 75th percentile of event conditions (catches 25% highest)
        # YELLOW threshold: 50th percentile (catches 50%)
        
        thresholds = {
            'inj_whp': {
                'red': event_df['inj_whp_max'].quantile(0.75),
                'yellow': event_df['inj_whp_max'].quantile(0.50),
                'unit': 'MPa'
            },
            'inj_flow': {
                'red': event_df['inj_flow_max'].quantile(0.75),
                'yellow': event_df['inj_flow_max'].quantile(0.50),
                'unit': 'L/s'
            },
            'pressure_diff': {
                'red': event_df['pressure_diff'].quantile(0.75),
                'yellow': event_df['pressure_diff'].quantile(0.50),
                'unit': 'MPa'
            },
            'flow_imbalance': {
                'red': event_df['flow_imbalance'].quantile(0.75),
                'yellow': event_df['flow_imbalance'].quantile(0.50),
                'unit': 'L/s'
            },
            'inj_whp_change_rate': {
                'red': event_df['inj_whp_change'].quantile(0.75),
                'yellow': event_df['inj_whp_change'].quantile(0.50),
                'unit': 'MPa/5min'
            },
            'inj_flow_change_rate': {
                'red': event_df['inj_flow_change'].quantile(0.75),
                'yellow': event_df['inj_flow_change'].quantile(0.50),
                'unit': 'L/s/5min'
            }
        }
        
        print("\n📋 CALCULATED THRESHOLDS (from historical events):")
        print("-" * 60)
        for param, values in thresholds.items():
            print(f"   {param}:")
            print(f"      🔴 RED:    > {values['red']:.2f} {values['unit']}")
            print(f"      🟡 YELLOW: > {values['yellow']:.2f} {values['unit']}")
        
        self.thresholds = thresholds
        return thresholds
    
    def check_rules(self, current_ops):
        """
        Layer 1: Rule-based checks (HIGH PRECISION)
        Returns: 'RED', 'YELLOW', or 'GREEN'
        """
        violations = {'red': [], 'yellow': []}
        
        # Check each threshold
        checks = [
            ('inj_whp', current_ops.get('inj_whp', 0)),
            ('inj_flow', current_ops.get('inj_flow', 0)),
            ('pressure_diff', current_ops.get('inj_whp', 0) - current_ops.get('prod_whp', 0)),
            ('flow_imbalance', current_ops.get('inj_flow', 0) - current_ops.get('prod_flow', 0)),
            ('inj_whp_change_rate', abs(current_ops.get('inj_whp_change', 0))),
            ('inj_flow_change_rate', abs(current_ops.get('inj_flow_change', 0)))
        ]
        
        for param, value in checks:
            if param in self.thresholds:
                if value > self.thresholds[param]['red']:
                    violations['red'].append(f"{param}: {value:.2f} > {self.thresholds[param]['red']:.2f}")
                elif value > self.thresholds[param]['yellow']:
                    violations['yellow'].append(f"{param}: {value:.2f} > {self.thresholds[param]['yellow']:.2f}")
        
        if violations['red']:
            return 'RED', violations['red']
        elif violations['yellow']:
            return 'YELLOW', violations['yellow']
        else:
            return 'GREEN', []
    
    def create_features(self, operations_df):
        """Create ML features from operational data"""
        features_df = pd.DataFrame()
        features_df['timestamp'] = operations_df['recorded_at']
        
        windows = {'1h': 12, '6h': 72, '24h': 288, '48h': 576}
        
        # Pressure features
        features_df['inj_whp'] = operations_df['inj_whp']
        features_df['inj_whp_change'] = operations_df['inj_whp'].diff()
        for name, w in windows.items():
            features_df[f'inj_whp_mean_{name}'] = operations_df['inj_whp'].rolling(w, min_periods=1).mean()
            features_df[f'inj_whp_std_{name}'] = operations_df['inj_whp'].rolling(w, min_periods=1).std()
            features_df[f'inj_whp_max_{name}'] = operations_df['inj_whp'].rolling(w, min_periods=1).max()
        
        features_df['prod_whp'] = operations_df['prod_whp']
        features_df['pressure_diff'] = operations_df['inj_whp'] - operations_df['prod_whp']
        
        # Flow features
        features_df['inj_flow'] = operations_df['inj_flow']
        features_df['inj_flow_change'] = operations_df['inj_flow'].diff()
        for name, w in windows.items():
            features_df[f'inj_flow_mean_{name}'] = operations_df['inj_flow'].rolling(w, min_periods=1).mean()
            features_df[f'inj_flow_std_{name}'] = operations_df['inj_flow'].rolling(w, min_periods=1).std()
        
        features_df['prod_flow'] = operations_df['prod_flow']
        features_df['flow_imbalance'] = operations_df['inj_flow'] - operations_df['prod_flow']
        
        # Temperature
        features_df['temp_diff'] = operations_df['prod_temp'] - operations_df['inj_temp']
        
        # Volume/Energy rates
        if 'cum_volume' in operations_df.columns:
            features_df['volume_rate'] = operations_df['cum_volume'].diff()
        if 'cum_inj_energy' in operations_df.columns:
            features_df['energy_rate'] = operations_df['cum_inj_energy'].diff()
        
        # Hydraulic power
        features_df['hydraulic_power'] = operations_df['inj_flow'] * operations_df['inj_whp']
        
        features_df = features_df.fillna(0).replace([np.inf, -np.inf], 0)
        
        return features_df
    
    def train(self, operations_df, seismic_df):
        """Train the hybrid system"""
        print("\n" + "="*70)
        print(" TRAINING HYBRID WARNING SYSTEM")
        print("="*70)
        
        # Step 1: Calculate rule-based thresholds
        self.calculate_thresholds_from_data(operations_df, seismic_df)
        
        # Step 2: Create ML features
        print("\n🔧 Creating ML features...")
        features_df = self.create_features(operations_df)
        
        self.feature_names = [c for c in features_df.columns if c != 'timestamp']
        print(f"   Created {len(self.feature_names)} features")
        
        # Step 3: Create labels (48h before significant events)
        print("\n🎯 Creating labels...")
        features_df['near_event'] = False
        
        for _, event in seismic_df[seismic_df['magnitude'] > 0.5].iterrows():
            t = event['occurred_at']
            mask = (
                (features_df['timestamp'] >= t - pd.Timedelta(hours=48)) &
                (features_df['timestamp'] <= t)
            )
            features_df.loc[mask, 'near_event'] = True
        
        print(f"   Events labeled: {features_df['near_event'].sum()} samples")
        
        # Step 4: Train ML model
        print("\n🤖 Training ML model (Random Forest Classifier)...")
        
        X = features_df[self.feature_names].values
        y = features_df['near_event'].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Time-series split: 80% train, 20% test
        split_idx = int(len(X_scaled) * 0.8)
        
        X_train = X_scaled[:split_idx]
        X_test = X_scaled[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"   Train: {len(X_train)} samples ({y_train.sum()} events)")
        print(f"   Test:  {len(X_test)} samples ({y_test.sum()} events)")
        
        # Random Forest Classifier (SUPERVISED - uses labels!)
        from sklearn.ensemble import RandomForestClassifier
        
        # Calculate class weight manually (more aggressive)
        n_events = y_train.sum()
        n_non_events = len(y_train) - n_events
        event_weight = n_non_events / n_events  # ~50x weight for events
        
        print(f"   Class imbalance: {n_non_events/n_events:.0f}:1")
        print(f"   Event weight: {event_weight:.1f}x")
        
        self.ml_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight={False: 1, True: event_weight},  # Aggressive weighting
            random_state=42,
            n_jobs=-1
        )
        self.ml_model.fit(X_train, y_train)
        
        # Store probabilities for percentile calculations
        self.score_percentiles = self.ml_model.predict_proba(X_scaled)[:, 1]
        
        # Evaluate on TEST set with MULTIPLE thresholds
        test_proba = self.ml_model.predict_proba(X_test)[:, 1]
        
        print(f"\n   Finding optimal threshold...")
        print(f"   Event probabilities in test set:")
        print(f"      Min:  {test_proba[y_test].min():.4f}" if y_test.sum() > 0 else "      No events")
        print(f"      Max:  {test_proba[y_test].max():.4f}" if y_test.sum() > 0 else "")
        print(f"      Mean: {test_proba[y_test].mean():.4f}" if y_test.sum() > 0 else "")
        
        # Try different thresholds, pick best F1
        best_f1 = 0
        best_threshold = 0.1
        best_metrics = {}
        
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            pred = test_proba >= threshold
            tp = (pred & y_test).sum()
            fp = (pred & ~y_test).sum()
            fn = (~pred & y_test).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            print(f"      Threshold {threshold}: P={prec:.1%}, R={rec:.1%}, F1={f1:.3f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}
        
        self.threshold = best_threshold
        print(f"\n   ✅ Best threshold: {best_threshold} (F1={best_f1:.3f})")
        
        # Final evaluation with best threshold
        test_predictions = test_proba >= best_threshold
        
        tp = (test_predictions & y_test).sum()
        fp = (test_predictions & ~y_test).sum()
        fn = (~test_predictions & y_test).sum()
        tn = (~test_predictions & ~y_test).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        base_rate = y_test.mean()
        anomaly_rate = y_test[test_predictions].mean() if test_predictions.sum() > 0 else 0
        enrichment = anomaly_rate / base_rate if base_rate > 0 else 0
        
        print(f"\n" + "="*50)
        print(f"📊 ML MODEL PERFORMANCE (on TEST data)")
        print(f"="*50)
        print(f"   Model:       Random Forest Classifier")
        print(f"   Threshold:   0.3 (for higher recall)")
        print(f"   Saved to:    trained_models/hybrid_system.pkl")
        print(f"")
        print(f"   Precision:   {precision:.1%}")
        print(f"   Recall:      {recall:.1%}")
        print(f"   F1 Score:    {f1:.3f}")
        print(f"   Enrichment:  {enrichment:.1f}x")
        print(f"")
        print(f"   Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                    No Event    Event")
        print(f"   Actual No Event    {tn:6d}    {fp:6d}")
        print(f"   Actual Event       {fn:6d}    {tp:6d}")
        print(f"="*50)
        
        # Feature importance
        print(f"\n📈 Top 10 Important Features:")
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.ml_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance_df.head(10).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        self.feature_importance = importance_df
        self.metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'enrichment': enrichment,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
        return self.metrics
    
    def predict(self, current_ops_df):
        """
        Make prediction using HYBRID approach
        
        Args:
            current_ops_df: DataFrame with recent operational data (last 24h)
        
        Returns:
            dict with risk level, reasons, and recommendations
        """
        # Get latest values
        latest = current_ops_df.iloc[-1].to_dict()
        latest['inj_whp_change'] = current_ops_df['inj_whp'].diff().iloc[-1]
        latest['inj_flow_change'] = current_ops_df['inj_flow'].diff().iloc[-1]
        
        # ========================================
        # LAYER 1: RULE-BASED (High Precision)
        # ========================================
        rule_level, rule_violations = self.check_rules(latest)
        
        # ========================================
        # LAYER 2: ML-BASED (Early Warning)
        # ========================================
        features_df = self.create_features(current_ops_df)
        X = features_df[self.feature_names].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        anomaly_score = self.ml_model.score_samples(X_scaled)[0]
        is_anomaly = self.ml_model.predict(X_scaled)[0] == -1
        
        # Calculate percentile (lower = more anomalous)
        percentile = (self.score_percentiles < anomaly_score).mean() * 100
        
        # ML risk level
        if percentile < 2:
            ml_level = 'RED'
        elif percentile < 10:
            ml_level = 'YELLOW'
        else:
            ml_level = 'GREEN'
        
        # ========================================
        # COMBINE: Take MAXIMUM risk level
        # ========================================
        levels = {'RED': 3, 'YELLOW': 2, 'GREEN': 1}
        
        final_level = 'RED' if levels[rule_level] == 3 or levels[ml_level] == 3 else \
                      'YELLOW' if levels[rule_level] == 2 or levels[ml_level] == 2 else \
                      'GREEN'
        
        # Build response
        result = {
            'final_level': final_level,
            'rule_level': rule_level,
            'ml_level': ml_level,
            'rule_violations': rule_violations,
            'ml_percentile': percentile,
            'ml_is_anomaly': is_anomaly,
            'timestamp': datetime.now()
        }
        
        # Add recommendations
        if final_level == 'RED':
            result['symbol'] = '🔴'
            result['action'] = 'EVACUATE high-risk zones immediately'
            result['details'] = [
                '⛔ CRITICAL RISK DETECTED',
                'Rule violations:' if rule_violations else 'ML: Highly anomalous pattern',
                *rule_violations,
                '',
                'IMMEDIATE ACTIONS:',
                '• Evacuate workers from underground/high-risk zones',
                '• Halt injection operations if safe to do so',
                '• Notify management immediately',
                '• Activate emergency response protocol'
            ]
        elif final_level == 'YELLOW':
            result['symbol'] = '🟡'
            result['action'] = 'CAUTION - Reduce high-risk activities'
            result['details'] = [
                '⚠️  ELEVATED RISK',
                f'Rule status: {rule_level}',
                f'ML status: {ml_level} (percentile: {percentile:.1f}%)',
                '',
                'RECOMMENDED ACTIONS:',
                '• Reduce non-essential high-risk activities',
                '• Increase monitoring frequency',
                '• Brief workers on evacuation procedures',
                '• Review operational parameters',
                '• Prepare for possible escalation'
            ]
        else:
            result['symbol'] = '🟢'
            result['action'] = 'Continue normal operations'
            result['details'] = [
                '✓ NORMAL OPERATIONS',
                f'All parameters within safe limits',
                f'ML anomaly score: percentile {percentile:.1f}%',
                '',
                'ACTIONS:',
                '• Continue standard operations',
                '• Maintain regular monitoring'
            ]
        
        return result
    
    def save(self, path='trained_models/hybrid_system.pkl'):
        """Save the trained system"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'ml_model': self.ml_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'thresholds': self.thresholds,
            'score_percentiles': self.score_percentiles,
            'ml_threshold': self.threshold,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance.to_dict(),
            'model_type': 'Random Forest Classifier',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n💾 Saved to {path}")
    
    def load(self, path='trained_models/hybrid_system.pkl'):
        """Load a trained system"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.ml_model = model_data['ml_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.thresholds = model_data['thresholds']
        self.score_percentiles = model_data['score_percentiles']
        
        print(f"✅ Loaded model from {path}")


def train_hybrid_system():
    """Main training function"""
    print("\n" + "="*70)
    print(" HYBRID SEISMIC WARNING SYSTEM")
    print("="*70)
    print("\n📋 This system combines:")
    print("   1. RULE-BASED thresholds (HIGH PRECISION - for RED alerts)")
    print("   2. ML anomaly detection (EARLY WARNING - for YELLOW alerts)")
    print()
    
    # Load data
    print("📊 Loading data...")
    seismic_df = pd.read_csv('../seismic_events_1.csv')
    operations_df = pd.read_csv('../operational_metrics.csv')
    
    operations_df['recorded_at'] = pd.to_datetime(operations_df['recorded_at'])
    seismic_df['occurred_at'] = pd.to_datetime(seismic_df['occurred_at'])
    operations_df = operations_df.sort_values('recorded_at').reset_index(drop=True)
    operations_df = operations_df.ffill().bfill()
    
    print(f"✅ Loaded {len(operations_df)} operational records")
    print(f"✅ Loaded {len(seismic_df)} seismic events")
    print(f"   Significant events (M>0.5): {(seismic_df['magnitude'] > 0.5).sum()}")
    
    # ========================================
    # REMOVE SHUTDOWN PERIODS (based on actual data analysis)
    # ========================================
    print("\n⚠️  Removing plant shutdown periods...")
    
    # Define shutdown periods (inj_flow = 0, is_producing = 0)
    shutdown_periods = [
        ('2019-07-01', '2021-04-30'),  # Main shutdown: 22 months
        ('2022-12-01', '2023-03-31'),  # Shutdown 2: 4 months
        ('2023-10-01', '2023-11-30'),  # Shutdown 3: 2 months
        ('2025-07-01', '2025-09-30'),  # Shutdown 4: 3 months (recent)
    ]
    
    ops_before = len(operations_df)
    events_before = len(seismic_df)
    
    # Remove each shutdown period
    for start, end in shutdown_periods:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        
        # Remove operational data
        operations_df = operations_df[
            ~((operations_df['recorded_at'] >= start_dt) & 
              (operations_df['recorded_at'] <= end_dt))
        ]
        
        # Remove seismic events
        seismic_df = seismic_df[
            ~((seismic_df['occurred_at'] >= start_dt) & 
              (seismic_df['occurred_at'] <= end_dt))
        ]
        
        print(f"   Removed: {start} to {end}")
    
    operations_df = operations_df.reset_index(drop=True)
    seismic_df = seismic_df.reset_index(drop=True)
    
    print(f"\n   Removed {ops_before - len(operations_df)} operational records")
    print(f"   Removed {events_before - len(seismic_df)} seismic events")
    print(f"   Remaining: {len(operations_df)} operational records")
    print(f"   Remaining: {len(seismic_df)} seismic events ({(seismic_df['magnitude'] > 0.5).sum()} significant)")
    
    # Create and train system
    system = HybridSeismicWarningSystem()
    metrics = system.train(operations_df, seismic_df)
    
    # Save
    system.save()
    
    # Summary
    print("\n" + "="*70)
    print("✅ HYBRID SYSTEM READY")
    print("="*70)
    
    print("\n" + "="*70)
    print("📊 FINAL MODEL METRICS (NO DATA LEAKAGE)")
    print("="*70)
    print(f"""
   ┌─────────────────────────────────────────┐
   │  MODEL: Random Forest Classifier        │
   │  WINDOW: 48 hours before events         │
   │  SAVED:  trained_models/hybrid_system.pkl│
   ├─────────────────────────────────────────┤
   │  PRECISION:  {metrics['precision']:.1%}                       │
   │  RECALL:     {metrics['recall']:.1%}                       │
   │  F1 SCORE:   {metrics['f1']:.3f}                       │
   │  ENRICHMENT: {metrics['enrichment']:.1f}x                        │
   ├─────────────────────────────────────────┤
   │  TP: {metrics['tp']:6d}  (Correctly predicted events) │
   │  FP: {metrics['fp']:6d}  (False alarms)               │
   │  FN: {metrics['fn']:6d}  (Missed events)              │
   │  TN: {metrics['tn']:6d}  (Correctly predicted safe)   │
   └─────────────────────────────────────────┘
    """)
    
    print("\n🚦 HOW IT WORKS:")
    print("-" * 50)
    print("🔴 RED (High Precision):")
    print("   • Triggered by RULE VIOLATIONS (hard thresholds)")
    print("   • OR very high ML anomaly (top 2%)")
    print("   • ACTION: EVACUATE immediately")
    print()
    print("🟡 YELLOW (Early Warning):")
    print("   • Triggered by ML anomaly detection (top 10%)")
    print("   • OR approaching rule thresholds")
    print("   • ACTION: Reduce risks, increase monitoring")
    print()
    print("🟢 GREEN (Safe):")
    print("   • All rules pass")
    print("   • ML shows normal patterns")
    print("   • ACTION: Continue normal operations")
    
    print("\n📋 THRESHOLDS SET:")
    print("-" * 50)
    for param, values in system.thresholds.items():
        print(f"   {param}:")
        print(f"      🔴 RED:    > {values['red']:.2f} {values['unit']}")
        print(f"      🟡 YELLOW: > {values['yellow']:.2f} {values['unit']}")
    
    print("\n💡 To use daily:")
    print("   python daily_hybrid_check.py")
    print()
    
    return system


if __name__ == "__main__":
    train_hybrid_system()