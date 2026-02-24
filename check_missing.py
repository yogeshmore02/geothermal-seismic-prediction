import pandas as pd
import numpy as np

ops = pd.read_csv('../operational_metrics.csv')
ops['recorded_at'] = pd.to_datetime(ops['recorded_at'])

print(f"Total records: {len(ops)}")
print(f"Date range: {ops['recorded_at'].min()} to {ops['recorded_at'].max()}")

# Check key operational variables
key_vars = ['inj_flow', 'inj_whp', 'prod_flow', 'prod_whp']

print("\n" + "="*60)
print("CHECKING FOR SHUTDOWN PERIODS")
print("="*60)

# Create month column
ops['month'] = ops['recorded_at'].dt.to_period('M')

# For each month, check if there's actual activity
print("\nMonthly MEAN of inj_flow (0 or NaN = no injection = shutdown):")
print("-"*60)

monthly_activity = ops.groupby('month').agg({
    'inj_flow': 'mean',
    'inj_whp': 'mean',
    'prod_flow': 'mean',
    'is_producing': 'mean'
}).round(2)

print(monthly_activity.to_string())

# Find months with no activity
print("\n" + "="*60)
print("MONTHS WITH ZERO/LOW INJECTION FLOW (likely shutdown):")
print("="*60)

low_activity = monthly_activity[monthly_activity['inj_flow'] < 1]
print(low_activity)

# Find the gap
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)

# Check is_producing column
ops['is_active'] = (ops['inj_flow'] > 0) | (ops['prod_flow'] > 0)
monthly_active = ops.groupby('month')['is_active'].mean()

inactive_months = monthly_active[monthly_active < 0.1].index.tolist()
print(f"\nInactive months (less than 10% active): {len(inactive_months)}")
if inactive_months:
    print(f"From: {inactive_months[0]}")
    print(f"To:   {inactive_months[-1]}")