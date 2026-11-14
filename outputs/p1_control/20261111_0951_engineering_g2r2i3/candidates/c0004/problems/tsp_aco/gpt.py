import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return topology-aware edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    positive = d[valid & (d > 0.0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(valid, d, scale)
    safe = np.maximum(safe, 0.0)
    np.fill_diagonal(safe, scale)

    # Robust global distance kernel.
    q25, q75 = np.percentile(safe[valid], [25.0, 75.0]) if np.any(valid) else (0.0, scale)
    spread = max(float(q75 - q25), np.finfo(float).eps)
    distance_kernel = np.exp(-np.maximum(safe - q25, 0.0) / spread)

    # Centered distance profiles describe each node's structural position.
    profiles = safe.copy()
    profiles[~valid] = scale
    np.fill_diagonal(profiles, scale)

    profiles -= np.mean(profiles, axis=1, keepdims=True)
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(float).eps)

    correlation = (profiles @ profiles.T) / (norms @ norms.T)
    profile_similarity = np.clip(0.5 * (correlation + 1.0), 0.0, 1.0)

    # Contrast against each endpoint's typical incident distance.
    typical = np.median(np.where(valid, d, np.nan), axis=1)
    typical = np.where(np.isfinite(typical), typical, scale)
    typical = np.maximum(typical, np.finfo(float).eps)

    endpoint_contrast = np.sqrt(
        np.exp(-safe / typical[:, None])
        * np.exp(-safe / typical[None, :])
    )

    score = (
        distance_kernel
        * (0.35 + 0.65 * profile_similarity)
        * (0.40 + 0.60 * endpoint_contrast)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    score = np.divide(
        score,
        row_max,
        out=np.zeros_like(score),
        where=row_max > 0.0,
    )

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
