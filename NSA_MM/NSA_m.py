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
        Generate detectors that do not match any self sample
        """
        detectors = []
        attempts = 0
        while len(detectors) < self.num_detectors and attempts < self.num_detectors * 10:
            candidate = np.random.rand(self.feature_dim)
            # compute Mahalanobis distance to all self samples
            dists = [mahalanobis(candidate, s, self.inv_cov) for s in self_samples]
            if all(dist > self.radius for dist in dists):
                detectors.append(candidate)
            attempts += 1
        self.detectors = np.array(detectors)
        return self.detectors

    def detect(self, samples):
        """
        Compute anomaly scores (min Mahalanobis distance to detectors)
        """
        # vectorized distance computation per sample
        scores = []
        for sample in samples:
            dists = [mahalanobis(sample, det, self.inv_cov) for det in self.detectors]
            scores.append(min(dists))
        return np.array(scores)

# Load dataset
data = pd.read_csv('UNSW_NB15.csv')
label_col = 'label'
if label_col not in data.columns:
    raise ValueError(f"Column '{label_col}' not found in dataset")

# Preprocess
data = data.select_dtypes(include=[np.number]).fillna(0)
labels = data[label_col].astype(int).values
features = data.drop(columns=[label_col]).values

# Prepare self (normal) samples and test set
def prepare_data(features, labels):
    self_samples = features[labels == 0]
    test_features = features
    test_labels = labels
    return self_samples, test_features, test_labels

self_samples, test_features, true_labels = prepare_data(features, labels)

# Compute inverse covariance matrix
cov = np.cov(self_samples, rowvar=False)
inv_cov = np.linalg.inv(cov + np.eye(cov.shape[0]) * 1e-6)  # regularization

# Run Negative Selection with Mahalanobis
nsa = NegativeSelectionMahalanobis(
    feature_dim=features.shape[1],
    inv_cov_matrix=inv_cov,
    num_detectors=300,
    radius=2.5  # adjust based on data distribution
)
nsa.generate_detectors(self_samples)
scores = nsa.detect(test_features)

# Evaluate
fpr, tpr, roc_thresholds = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)
precision, recall, pr_thresholds = precision_recall_curve(true_labels, scores)
avg_prec = average_precision_score(true_labels, scores)

# Compute F1 for each precision-recall threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
# Align thresholds (pr_thresholds has len = len(precision)-1)
f1_thresholds = pr_thresholds
f1_scores = f1_scores[:-1]

# Optimal threshold for max F1
best_idx = np.argmax(f1_scores)
best_threshold = f1_thresholds[best_idx]
best_f1 = f1_scores[best_idx]
best_precision = precision[best_idx]
best_recall = recall[best_idx]

print(f"Optimal threshold: {best_threshold:.4f}")
print(f"Precision at optimal: {best_precision:.2f}")
print(f"Recall at optimal: {best_recall:.2f}")
print(f"F1-score at optimal: {best_f1:.2f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr)
plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# Plot Precision-Recall curve with optimal point
plt.figure()
plt.plot(recall, precision, label='PR Curve')
plt.scatter(best_recall, best_precision, color='red', label=f'Optimal (F1={best_f1:.2f})')
plt.title(f'Precision-Recall Curve (AP = {avg_prec:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.show()
