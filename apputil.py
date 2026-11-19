"""Basic k-means clustering and timing functions."""

from time import perf_counter

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


df_diamonds = sns.load_dataset("diamonds").select_dtypes(include="number")


def kmeans(X, k):
    """Return cluster centroids and one cluster label per observation."""
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(np.asarray(X, dtype=float))
    return model.cluster_centers_, labels


def kmeans_diamonds(n, k):
    """Cluster the first n rows of the seven numeric diamonds columns."""
    if (
        not isinstance(n, (int, np.integer))
        or isinstance(n, bool)
        or not 1 <= n <= len(df_diamonds)
    ):
        raise ValueError("n must be between 1 and the dataset row count.")
    return kmeans(df_diamonds.iloc[:n].to_numpy(), k)


def kmeans_timer(n, k, n_iter=5):
    """Return the mean runtime in seconds across exactly n_iter runs."""
    if (
        not isinstance(n_iter, (int, np.integer))
        or isinstance(n_iter, bool)
        or n_iter < 1
    ):
        raise ValueError("n_iter must be a positive integer.")

    runtimes = []
    for _ in range(n_iter):
        start = perf_counter()
        kmeans_diamonds(n, k)
        runtimes.append(perf_counter() - start)
    return float(np.mean(runtimes))
