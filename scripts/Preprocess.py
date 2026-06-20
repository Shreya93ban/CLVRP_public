import numpy as np
import math


#  ----------------------------------------------------------------------
# STEP 1 & 2 : Constrained K‑Means + nearest points (merged)
# ----------------------------------------------------------------------


def constrained_kmeans(points, k=3, cluster_size=4, max_iter=300, random_state=None):
    """Constrained k‑means: each cluster gets exactly cluster_size points."""
    if random_state is not None:
        np.random.seed(random_state)
    n_points = len(points)
    centroids = points[np.random.choice(n_points, k, replace=False)]

    for _ in range(max_iter):
        distances = np.array([[np.sqrt(np.sum((points[i] - centroids[j])**2))
                               for j in range(k)] for i in range(n_points)])

        assignments = np.zeros(n_points, dtype=int)
        cluster_counts = np.zeros(k, dtype=int)
        sorted_indices = np.argsort(distances, axis=1)
        for i in range(n_points):
            for j in sorted_indices[i]:
                if cluster_counts[j] < cluster_size:
                    assignments[i] = j
                    cluster_counts[j] += 1
                    break

        if np.all(cluster_counts == cluster_size):
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                cluster_points = points[assignments == j]
                new_centroids[j] = np.mean(cluster_points, axis=0)
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        else:
            # re‑initialize if capacity violated
            centroids = points[np.random.choice(n_points, k, replace=False)]

    return assignments, centroids





def find_entry_exit(cluster_points, all_points, cluster_id, centroids, depot):
    """
    For a given cluster, find the two best points (entry and exit)
    using the original scoring: distance to depot + distances to other centroids.
    Returns the indices (in the original customers array) of the two points.
    """
    other_clusters = [c for c in range(3) if c != cluster_id]
    scored = []
    for p_idx in cluster_points:
        point = all_points[p_idx]
        d_depot = math.hypot(point[0] - depot[0], point[1] - depot[1])
        d_other = [math.hypot(point[0] - centroids[oc][0], point[1] - centroids[oc][1])
                   for oc in other_clusters]
        score = d_depot + sum(d_other)
        scored.append((p_idx, score))
    scored.sort(key=lambda x: x[1])
    return scored[0][0], scored[1][0]