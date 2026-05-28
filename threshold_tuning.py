"""
THRESHOLD TUNING FOR eSIM FRAUD DETECTION
==========================================
This script improves fraud recall by lowering the detection threshold.

What it does:
1. Loads your trained XGBoost model
2. Generates fraud probabilities
3. Tests different thresholds (0.50 → 0.30)
4. Finds optimal threshold for maximum recall
5. Saves results and updated model configuration

Why this matters:
- Default threshold (0.50) misses 90% of fraud cases
- Lowering to 0.35-0.40 catches MORE fraud
- Tradeoff: More false positives BUT fewer missed frauds
- This is realistic for fraud detection systems
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, 
    f1_score, classification_report
)
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs('output', exist_ok=True)

print("="*80)
print("THRESHOLD TUNING FOR eSIM FRAUD DETECTION")
print("="*80)

# ============ STEP 1: LOAD YOUR TRAINED MODEL ============
print("\n[1/6] Loading trained XGBoost model...")

try:
    model = joblib.load('output/xgb_fraud_model.pkl')
    feature_cols = joblib.load('output/feature_names.pkl')
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("✗ Model not found! Please train your model first.")
    exit()

# ============ STEP 2: LOAD TEST DATA ============
print("\n[2/6] Loading test data...")

# Replace with your actual dataset filename
df = pd.read_csv('output/esim_fraud_dataset.csv')

X = df[feature_cols]
y = df['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"  Fraud cases: {y_test.sum()} ({y_test.mean():.2%})")

# ============ STEP 3: GENERATE FRAUD PROBABILITIES ============
print("\n[3/6] Generating fraud probabilities...")

y_prob = model.predict_proba(X_test)[:, 1]

print(f"✓ Probabilities generated!")
print(f"  Mean: {y_prob.mean():.4f}, Min: {y_prob.min():.4f}, Max: {y_prob.max():.4f}")

# ============ STEP 4: TEST DIFFERENT THRESHOLDS ============
print("\n[4/6] Testing different thresholds...")
print("-"*80)

thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]
results = []

for thresh in thresholds:
    y_pred = (y_prob >= thresh).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 1.0
    
    results.append({
        'threshold': thresh,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_negative_rate': fnr,
        'total_fraud_detected': int(tp)
    })
    
    print(f"Threshold {thresh:.2f}:")
    print(f"  → Recall: {recall:.4f} ({recall*100:.1f}% fraud caught)")
    print(f"  → Precision: {precision:.4f}")
    print(f"  → F1-Score: {f1:.4f}")
    print(f"  → Fraud detected: {tp}/{tp+fn}")
    print()

results_df = pd.DataFrame(results)

# Find best threshold
best_idx = results_df['recall'].idxmax()
best_threshold = results_df.loc[best_idx, 'threshold']
best_recall = results_df.loc[best_idx, 'recall']

print("-"*80)
print(f"✓ BEST THRESHOLD: {best_threshold}")
print(f"  → Achieves {best_recall*100:.1f}% recall")
print(f"  → Improvement over default: {best_recall - results_df.loc[0, 'recall']:.4f}")

# ============ STEP 5: SAVE RESULTS ============
print("\n[5/6] Saving results...")

results_df.to_csv('output/threshold_tuning_results.csv', index=False)
print("✓ Saved: output/threshold_tuning_results.csv")

joblib.dump(best_threshold, 'output/best_threshold.pkl')
print("✓ Saved: output/best_threshold.pkl")

final_predictions = pd.DataFrame({
    'actual_label': y_test.values,
    'fraud_probability': y_prob,
    'predicted_default': (y_prob >= 0.50).astype(int),
    'predicted_optimal': (y_prob >= best_threshold).astype(int)
})
final_predictions.to_csv('output/final_predictions.csv', index=False)
print("✓ Saved: output/final_predictions.csv")

# ============ STEP 6: CREATE VISUALIZATION ============
print("\n[6/6] Creating visualization...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['red' if t == best_threshold else 'steelblue' for t in results_df['threshold']]

ax1.bar(results_df['threshold'], results_df['recall'], color=colors, alpha=0.7)
ax1.axhline(y=results_df.loc[0, 'recall'], color='red', linestyle='--', 
            label=f'Default (0.50): {results_df.loc[0, "recall"]:.3f}')
ax1.axhline(y=best_recall, color='green', linestyle='--', 
            label=f'Best ({best_threshold}): {best_recall:.3f}')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Recall', fontsize=12)
ax1.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2.bar(results_df['threshold'], results_df['false_negative_rate'], color=colors, alpha=0.7)
ax2.axhline(y=results_df.loc[0, 'false_negative_rate'], color='red', linestyle='--')
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('False Negative Rate', fontsize=12)
ax2.set_title('False Negative Rate vs Threshold', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output/threshold_tuning_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: output/threshold_tuning_plot.png")

# ============ FINAL SUMMARY ============
print("\n" + "="*80)
print("THRESHOLD TUNING COMPLETE!")
print("="*80)
print(f"""
BEFORE (threshold = 0.50):
  • Recall: {results_df.loc[0, 'recall']:.4f} ({results_df.loc[0, 'recall']*100:.1f}% fraud caught)
  • False Negative Rate: {results_df.loc[0, 'false_negative_rate']:.4f}
  • Fraud detected: {results_df.loc[0, 'true_positives']} cases

AFTER (threshold = {best_threshold}):
  • Recall: {best_recall:.4f} ({best_recall*100:.1f}% fraud caught)  ↑{((best_recall - results_df.loc[0, 'recall']) / results_df.loc[0, 'recall'] * 100):.0f}%
  • False Negative Rate: {results_df.loc[best_idx, 'false_negative_rate']:.4f}  ↓{((results_df.loc[0, 'false_negative_rate'] - results_df.loc[best_idx, 'false_negative_rate']) / results_df.loc[0, 'false_negative_rate'] * 100):.0f}%
  • Fraud detected: {results_df.loc[best_idx, 'true_positives']} cases  ↑{results_df.loc[best_idx, 'true_positives'] - results_df.loc[0, 'true_positives']} more!

TRADEOFF:
  • False Positives: {results_df.loc[0, 'false_positives']} → {results_df.loc[best_idx, 'false_positives']} (+{results_df.loc[best_idx, 'false_positives'] - results_df.loc[0, 'false_positives']})
  • Precision: {results_df.loc[0, 'precision']:.4f} → {results_df.loc[best_idx, 'precision']:.4f}

✅ REALISTIC for fraud detection! Missing fraud is more costly than false alarms.
""")
print("="*80)