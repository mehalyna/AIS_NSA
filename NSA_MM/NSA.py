import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

class NegativeSelection:
    def __init__(self, feature_dim, num_detectors=200, radius=0.3):
        """
        feature_dim: dimensionality of input vectors
        num_detectors: number of detectors to generate
        radius: fuzzy threshold for detector matching
        """
        self.feature_dim = feature_dim
        self.num_detectors = num_detectors
        self.radius = radius
        self.detectors = []

    def generate_detectors(self, self_samples):
        """
        Generate detectors that do not match any self sample
        """
        self.detectors = []
        attempts = 0
        while len(self.detectors) < self.num_detectors and attempts < self.num_detectors * 10:
            candidate = np.random.rand(self.feature_dim)
            # compute distances to all self samples
            dists = np.linalg.norm(self_samples - candidate, axis=1)
            if np.all(dists > self.radius):
                self.detectors.append(candidate)
            attempts += 1
        self.detectors = np.array(self.detectors)
        return self.detectors

    def detect(self, samples):
        """
        For each sample, return distance to nearest detector
        """
        # Efficient computation using vectorized operations
        dists = []
        for d in self.detectors:
            # compute vector of distances from all samples to detector d
            dists.append(np.linalg.norm(samples - d, axis=1))
        # stack distances: shape (num_detectors, num_samples)
        dists = np.stack(dists, axis=0)
        # minimal distance across detectors for each sample
        min_dists = np.min(dists, axis=0)
        return min_dists

# Load UNSW-NB15 dataset (assumes CSV with numeric features and 'label' column)
data = pd.read_csv('UNSW_NB15.csv')

# Preprocess: keep only numeric features
label_col = 'label'
if label_col not in data.columns:
    raise ValueError(f"Column '{label_col}' not found in dataset")

# Separate labels
labels = data[label_col].astype(int).values
# Select numeric columns excluding label
def is_numeric(col):
    return pd.api.types.is_numeric_dtype(data[col])
numeric_cols = [c for c in data.columns if is_numeric(c) and c != label_col]
features = data[numeric_cols].fillna(0).values

# Split self samples (normal traffic) and tests
def prepare_data(features, labels):
    self_samples = features[labels == 0]
    test_features = features
    test_labels = labels
    return self_samples, test_features, test_labels

self_samples, test_features, true_labels = prepare_data(features, labels)

# Run NSA and get anomaly scores
def run_nsa(feature_dim, self_samples, test_features, num_detectors=300, radius=0.25):
    nsa = NegativeSelection(feature_dim=feature_dim, num_detectors=num_detectors, radius=radius)
    nsa.generate_detectors(self_samples)
    scores = nsa.detect(test_features)
    return scores

scores = run_nsa(features.shape[1], self_samples, test_features)

# Compute ROC curve and AUC
events = true_labels
fpr, tpr, _ = roc_curve(events, scores)
roc_auc = auc(fpr, tpr)

# Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(events, scores)
avg_prec = average_precision_score(events, scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr)
plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# Plot Precision-Recall curve
plt.figure()
plt.plot(recall, precision)
plt.title(f'Precision-Recall Curve (AP = {avg_prec:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()