import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Define self‐samples (cluster around origin)
self_samples = np.random.randn(10, 2) * 0.5

# Define negative selection radius (for “self” regions)
r = 0.5

# Sample candidate detectors uniformly in a region [-3, 3]^2
candidates = np.random.uniform(-3, 3, (1000, 2))

# Retain detectors that are at least distance r from all self‐samples
def is_valid_detector(candidate, self_samples, radius):
    distances = np.linalg.norm(self_samples - candidate, axis=1)
    return np.all(distances > radius)

all_detectors = np.array([c for c in candidates if is_valid_detector(c, self_samples, r)])

# Randomly choose a subset of detectors for plotting (e.g., 200)
plot_detectors = all_detectors[np.random.choice(len(all_detectors), size=200, replace=False)]

# Choose an unknown sample well outside the detector cloud
x = np.array([2.0, 2.0])

# Compute anomaly score: distance from x to each detector
distances_to_detectors = np.linalg.norm(plot_detectors - x, axis=1)
closest_idx = np.argmin(distances_to_detectors)
closest_detector = plot_detectors[closest_idx]

# Plot everything
fig, ax = plt.subplots(figsize=(8, 8))

# Plot each self-sample with a smaller, highly transparent circle of radius r
for s in self_samples:
    circle = plt.Circle(
        (s[0], s[1]),
        r,
        color='blue',
        alpha=0.15,  # more transparent
        linewidth=0  # no border line
    )
    ax.add_patch(circle)
ax.scatter(self_samples[:, 0], self_samples[:, 1], color='blue', s=50, label='Self Samples')

# Plot only the sampled detectors
ax.scatter(plot_detectors[:, 0], plot_detectors[:, 1],
           color='green', s=20, alpha=0.6, label='Detectors (subset)')

# Plot the unknown sample and its closest detector
ax.scatter(x[0], x[1], color='red', s=100, marker='X', label='Unknown Sample x')
ax.scatter(closest_detector[0], closest_detector[1], color='black',
           s=100, marker='*', label='Closest Detector')

# Draw a dashed line between x and its closest detector
ax.plot(
    [x[0], closest_detector[0]],
    [x[1], closest_detector[1]],
    color='red',
    linestyle='--',
    linewidth=1
)

# Annotate the points
ax.text(x[0] + 0.1, x[1] + 0.1, 'x (unknown)', color='red')
ax.text(closest_detector[0] + 0.1, closest_detector[1] + 0.1, 'd_closest', color='black')

# Plot settings
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('NSA Concept: Self‐Samples, Detectors, and Anomaly Scoring\n(cleaner visualization)')
ax.legend(loc='lower left')
ax.set_aspect('equal', 'box')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.grid(True)

plt.show()

