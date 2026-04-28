# code to obtain the centroids of the clusters of the free space, to be used as starting points for the
# voronoi partitioning instead of the drone starting positions, in case the initial position of the drones is
# fixed and not optimal for the partitioning.

import numpy as np
from shapely.geometry import Point


def _sample_free_space_points(free_space, num_samples, rng):
    """Sample random 2D points inside the free-space geometry."""
    minx, miny, maxx, maxy = free_space.bounds
    accepted_points = []
    attempts = 0
    max_attempts = max(num_samples * 50, 5000)

    while len(accepted_points) < num_samples and attempts < max_attempts:
        remaining = num_samples - len(accepted_points)
        batch_size = min(max(remaining * 2, 128), 2048)
        samples = rng.uniform([minx, miny], [maxx, maxy], size=(batch_size, 2))

        for sample in samples:
            attempts += 1
            if free_space.covers(Point(float(sample[0]), float(sample[1]))):
                accepted_points.append(sample)
                if len(accepted_points) >= num_samples:
                    break
            if attempts >= max_attempts:
                break

    if len(accepted_points) < num_samples:
        raise ValueError("Unable to sample enough points from the free space for k-means initialization.")

    return np.asarray(accepted_points, dtype=float)


def _initialize_centroids_kmeans_pp(points, num_clusters, rng):
    """Initialize centroids with a k-means++ style seeding."""
    first_index = int(rng.integers(0, len(points)))
    centroids = [points[first_index]]

    while len(centroids) < num_clusters:
        centroid_array = np.asarray(centroids, dtype=float)
        squared_distances = np.sum((points[:, None, :] - centroid_array[None, :, :]) ** 2, axis=2)
        min_squared_distances = np.min(squared_distances, axis=1)
        total_distance = float(np.sum(min_squared_distances))

        if total_distance <= 0.0:
            fallback_index = int(rng.integers(0, len(points)))
            centroids.append(points[fallback_index])
            continue

        probabilities = min_squared_distances / total_distance
        next_index = int(rng.choice(len(points), p=probabilities))
        centroids.append(points[next_index])

    return np.asarray(centroids, dtype=float)


def kmeans_clustering(free_space, num_drones, *, seed=None, num_samples=2000, max_iter=100, waypoints=None):
    """Find Voronoi seeds by clustering random samples drawn from the map free space."""
    if num_drones <= 0:
        raise ValueError("num_drones must be positive.")

    rng = np.random.default_rng(seed)
    
    if waypoints is not None:
        points = waypoints
    else:
        points = _sample_free_space_points(free_space, num_samples=num_samples, rng=rng)

    if len(points) < num_drones:
        raise ValueError("Not enough free-space samples to initialize all k-means centroids.")

    centroids = _initialize_centroids_kmeans_pp(points, num_drones, rng)
    labels = np.zeros(len(points), dtype=int)

    for _ in range(max_iter):
        squared_distances = np.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(squared_distances, axis=1)
        new_centroids = np.array(centroids, copy=True)

        for cluster_index in range(num_drones):
            cluster_points = points[new_labels == cluster_index]
            if len(cluster_points) == 0:
                farthest_point_index = int(np.argmax(np.min(squared_distances, axis=1)))
                new_centroids[cluster_index] = points[farthest_point_index]
                continue
            new_centroids[cluster_index] = np.mean(cluster_points, axis=0)
            if not free_space.covers(Point(*new_centroids[cluster_index])):
                nearest_index = int(
                    np.argmin(np.sum((cluster_points - new_centroids[cluster_index]) ** 2, axis=1))
                )
                new_centroids[cluster_index] = cluster_points[nearest_index]

        if np.allclose(new_centroids, centroids):
            labels = new_labels
            centroids = new_centroids
            break

        labels = new_labels
        centroids = new_centroids

    # Final snap to sampled free-space points guarantees valid seeds even on non-convex free regions.
    for cluster_index in range(num_drones):
        if free_space.covers(Point(*centroids[cluster_index])):
            continue

        cluster_points = points[labels == cluster_index]
        candidate_points = cluster_points if len(cluster_points) > 0 else points
        nearest_index = int(np.argmin(np.sum((candidate_points - centroids[cluster_index]) ** 2, axis=1)))
        centroids[cluster_index] = candidate_points[nearest_index]

    return centroids

def sanitize_waypoints(waypoints, obstacles, safety_margin=1.5):
    """
    Sposta i waypoint troppo vicini agli ostacoli.
    safety_margin: distanza minima dal centro dell'ostacolo (raggio ostacolo + buffer).
    """
    sanitized_wps = np.copy(waypoints)
    
    for i, wp in enumerate(sanitized_wps):
        for obs in obstacles:
            # Distanza dal centro dell'ostacolo
            dist = np.linalg.norm(wp[:2] - np.array([obs.x, obs.y]))
            
            # Limite minimo = raggio fisico + margine di manovra MPC
            min_dist = obs.radius + safety_margin
            
            if dist < min_dist:
                # Calcola il vettore unitario dal centro dell'ostacolo al waypoint
                push_dir = (wp[:2] - np.array([obs.x, obs.y]))
                push_dir_norm = np.linalg.norm(push_dir)
                
                if push_dir_norm < 1e-3: # Se il WP è esattamente al centro
                    push_dir = np.array([1.0, 0.0]) # Direzione arbitraria
                else:
                    push_dir /= push_dir_norm
                
                # Riposiziona il waypoint alla distanza minima di sicurezza + un piccolo offset
                sanitized_wps[i, :2] = np.array([obs.x, obs.y]) + push_dir * (min_dist + 0.2)
                
    return sanitized_wps


