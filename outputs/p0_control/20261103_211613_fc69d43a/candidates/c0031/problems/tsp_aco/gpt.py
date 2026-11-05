import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge scores based on relative-neighborhood exposure."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros_like(distances, dtype=float)

    if n <= 1:
        return scores

    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0):
        raise ValueError("distance_matrix must contain nonnegative distances")

    valid = finite & ~np.eye(n, dtype=bool)
    if not np.any(valid):
        return scores

    positive = distances[valid & (distances > 0)]
    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, distances, np.inf)

    # For each directed edge i -> j, count nodes k that block the edge:
    # d(i,k) < d(i,j) and d(k,j) < d(i,j).
    edge_length = safe[:, :, None]
    left = safe[:, None, :]
    right = safe[None, :, :]

    blockers = (left < edge_length) & (right < edge_length)
    blockers &= np.isfinite(left) & np.isfinite(right)
    blockers &= ~np.eye(n, dtype=bool)[None, :, :]
    blockers &= ~np.eye(n, dtype=bool)[:, None, :]

    blocker_count = blockers.sum(axis=2).astype(float)

    # Relative-neighborhood exposure, with a smooth tolerance for weak blockers.
    temperature = max(np.sqrt(float(max(n - 1, 1))), 1.0)
    exposure = np.exp(-blocker_count / temperature)

    # Use a robust global scale only to suppress unusually long exposed edges.
    length_penalty = 1.0 / (1.0 + safe / scale)

    # Reward edges whose reverse direction is also locally unblocked.
    reverse_exposure = exposure.T
    directional_agreement = 0.75 + 0.25 * np.sqrt(
        np.clip(exposure, 0.0, 1.0)
        * np.clip(reverse_exposure, 0.0, 1.0)
    )

    scores = exposure * length_penalty * directional_agreement
    scores[~valid] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    np.fill_diagonal(scores, 0.0)

    maximum = np.max(scores)
    if maximum > 0.0:
        scores /= maximum

    return scores
