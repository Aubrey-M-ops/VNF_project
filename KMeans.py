from sklearn.cluster import KMeans
import numpy as np


def apply_kmeans_clustering(locations, num_clusters):
    """
    Apply K-means clustering to a set of locations.
    """
    # Ensure locations is a numpy array
    locations = np.array(locations)

    # Handle edge case: if num_clusters is larger than number of points
    num_clusters = min(num_clusters, len(locations))

    # Initialize and fit K-means
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=42,  # For reproducibility
        n_init=10  # Number of times to run with different centroid seeds
    )

    # Perform clustering and get labels
    labels = kmeans.fit_predict(locations)

    return labels
