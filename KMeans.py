# K-Means Clustering on GPS Coordinates to Define Zones

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def apply_kmeans_clustering(data, n_clusters=5):
    print(f'👉 Applying K-Means clustering with {n_clusters} clusters...')
    
    # Extract coordinates of VNF destinations
    coords = data[['end_x', 'end_y']].dropna()

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(coords)

    # Assign zone labels to the original dataset
    data = data.copy()
    data.loc[coords.index, 'end_zone'] = cluster_labels

    print('👉 Clustering complete. Zones assigned as end_zone.')

    # Return data with zones and cluster centers (2D array)
    return data, kmeans.cluster_centers_