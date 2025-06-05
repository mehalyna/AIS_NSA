import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

class NegativeSelectionMahalanobis:
    def __init__(self, feature_dim, inv_cov_matrix, num_detectors=200, radius=0.3):
        """
        feature_dim: dimensionality of input vectors
        inv_cov_matrix: inverse covariance matrix for Mahalanobis distance
        num_detectors: number of detectors to generate
        radius: threshold for anomaly detection
        """
        self.feature_dim = feature_dim
        self.inv_cov = inv_cov_matrix
        self.num_detectors = num_detectors
        self.radius = radius
        self.detectors = None

    def generate_detectors(self, self_samples):
        """
        Generate detectors that do not match any self sample, and print intermediate results.
        """
        detectors = []
        attempts = 0
        print(f"[generate_detectors] Starting with target of {self.num_detectors} detectors and radius = {self.radius}")
        print(f"[generate_detectors] Number of self‐samples: {len(self_samples)}; feature dimension: {self.feature_dim}")
        
        while len(detectors) < self.num_detectors and attempts < self.num_detectors * 10:
            candidate = np.random.rand(self.feature_dim)
            # compute Mahalanobis distance to all self samples
            dists = [mahalanobis(candidate, s, self.inv_cov) for s in self_samples]
            min_dist = np.min(dists)
            if min_dist > self.radius:
                detectors.append(candidate)
            attempts += 1
            
            # Occasionally print progress every 1000 attempts
            if attempts % 1000 == 0:
                print(f"[generate_detectors] Attempts: {attempts}, Detectors so far: {len(detectors)}")
        
        self.detectors = np.array(detectors)
        print(f"[generate_detectors] Completed. Generated {len(self.detectors)} detectors after {attempts} attempts.")
        # Show first 5 detectors (if we have at least 5)
        if len(self.detectors) >= 5:
            print("[generate_detectors] First 5 detectors:\n", self.detectors[:5])
        return self.detectors

    def detect(self, samples):
        """
        Compute anomaly scores (min Mahalanobis distance to detectors) and print intermediate results.
        """
        if self.detectors is None or len(self.detectors) == 0:
            raise ValueError("Detectors have not been generated yet.")
        
        print(f"[detect] Scoring {len(samples)} samples against {len(self.detectors)} detectors")
        scores = []
        
        for idx, sample in enumerate(samples):
            # compute Mahalanobis distance to each detector
            dists = [mahalanobis(sample, det, self.inv_cov) for det in self.detectors]
            min_dist = np.min(dists)
            scores.append(min_dist)
            
            # Print the first few sample scores
            if idx < 5:
                print(f"[detect] Sample #{idx} min distance = {min_dist:.4f}")

            if min_dist <= 16.71:
                print(f"[detect] Sample #{idx} anomaly with min distance = {min_dist:.4f}")
        
        scores = np.array(scores)
        print(f"[detect] Completed scoring. Example scores (first 5): {scores[:5]}")
        return scores

# --------------------------------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------------------------------

# Step 1: Load dataset
print("[main] Loading dataset")
data = pd.read_csv('UNSW_NB15.csv')
label_col = 'label'
if label_col not in data.columns:
    raise ValueError(f"Column '{label_col}' not found in dataset")

# Step 2: Preprocess
print("[main] Preprocessing: Selecting numeric columns and filling NaNs")
data = data.select_dtypes(include=[np.number]).fillna(0)
labels = data[label_col].astype(int).values
features = data.drop(columns=[label_col]).values
print(f"[main] Feature matrix shape: {features.shape}, label vector length: {len(labels)}")

# Step 3: Prepare self (normal) samples and test set
def prepare_data(features, labels):
    self_samples = features[labels == 0]
    test_features = features
    test_labels = labels
    return self_samples, test_features, test_labels

self_samples, test_features, true_labels = prepare_data(features, labels)
print(f"[main] Number of self (normal) samples: {self_samples.shape[0]}")
print(f"[main] Number of total test samples: {test_features.shape[0]}")

# Step 4: Compute inverse covariance matrix
print("[main] Computing covariance matrix on self-samples")
cov = np.cov(self_samples, rowvar=False)
print(f"[main] Covariance matrix shape: {cov.shape}")
inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)  # regularization
print("[main] Inverted covariance matrix computed")

# Step 5: Run Negative Selection with Mahalanobis
print("[main] Initializing NegativeSelectionMahalanobis")
nsa = NegativeSelectionMahalanobis(
    feature_dim=features.shape[1],
    inv_cov_matrix=inv_cov,
    num_detectors=300,
    radius=2.5  # can be adjusted
)

# Generate detectors and display results
print("[main] Generating detectors")
detectors = nsa.generate_detectors(self_samples)

# Step 6: Detect anomalies and display results
print("[main] Computing anomaly scores")
scores = nsa.detect(test_features)

# Step 7: Evaluate performance
print("[main] Evaluating performance metrics")
fpr, tpr, roc_thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)
print(f"[main] ROC AUC = {roc_auc:.4f}")

precision, recall, pr_thresholds = precision_recall_curve(true_labels, scores)
avg_prec = average_precision_score(true_labels, scores)
print(f"[main] Average Precision (PR-AUC) = {avg_prec:.4f}")

# Step 8: Compute F1 for each precision-recall threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
# Align thresholds (pr_thresholds length = len(precision)-1)
f1_thresholds = pr_thresholds
f1_scores = f1_scores[:-1]

# Find optimal threshold
best_idx = np.argmax(f1_scores)
best_threshold = f1_thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_precision = precision[best_idx]
best_recall = recall[best_idx]

print(f"[main] Optimal threshold: {best_threshold:.4f}")
print(f"[main] Precision at optimal: {best_precision:.4f}")
print(f"[main] Recall at optimal: {best_recall:.4f}")
print(f"[main] F1-score at optimal: {best_f1:.4f}")

# Step 9: Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Step 10: Plot Precision-Recall curve with optimal point
plt.figure()
plt.plot(recall, precision, label='PR Curve')
plt.scatter(best_recall, best_precision, color='red',
            label=f'Optimal (F1={best_f1:.2f})')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
