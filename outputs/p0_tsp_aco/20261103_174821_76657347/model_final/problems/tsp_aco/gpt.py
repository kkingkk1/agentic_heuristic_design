import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)
    scores = np.zeros_like(distances, dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    scores[valid] = 1.0 / distances[valid]

    np.fill_diagonal(scores, 0.0)
    return scores
