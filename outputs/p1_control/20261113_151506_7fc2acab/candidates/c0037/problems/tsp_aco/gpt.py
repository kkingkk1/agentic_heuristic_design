import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return graph-diffusion and two-hop-shortcut priors for TSP edges."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")
    if not np.all(np.isfinite(d)):
        raise ValueError("distance_matrix must contain only finite values")
    if np.any(d < 0):
        raise ValueError("distance_matrix must be nonnegative")

    n = d.shape[0]
    if n <= 1:
        return np.zeros_like(d, dtype=float)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    finite_edges = np.isfinite(work)
    positive = work[finite_edges & (work > 0)]

    if positive.size:
        scale = float(np.percentile(positive, 60.0))
    else:
        scale = 1.0
    scale = max(scale, np.finfo(float).eps)

    safe = np.where(finite_edges, work, scale * 1e6)

    affinity = np.exp(-np.minimum(safe / scale, 60.0))
    np.fill_diagonal(affinity, 0.0)

    row_sum = affinity.sum(axis=1, keepdims=True)
    transition = np.divide(
        affinity,
        row_sum,
        out=np.zeros_like(affinity),
        where=row_sum > 0,
    )

    reverse_flow = transition * transition.T
    reverse_flow /= np.maximum(
        np.max(reverse_flow, axis=1, keepdims=True),
        np.finfo(float).eps,
    )

    two_hop = np.min(
        safe[:, :, None] + safe[None, :, :],
        axis=1,
    )
    finite_two_hop = np.isfinite(two_hop) & finite_edges

    fallback_hop = np.maximum(
        2.0 * scale,
        np.max(np.where(finite_edges, work, 0.0), axis=1, initial=0.0)[:, None],
    )
    two_hop = np.where(np.isfinite(two_hop), two_hop, fallback_hop)

    shortcut_gain = np.maximum(two_hop - safe, 0.0)
    shortcut_gain = np.minimum(shortcut_gain / scale, 30.0)
    shortcut_signal = np.exp(shortcut_gain)

    diffusion_support = transition @ transition
    diffusion_support = np.sqrt(
        np.maximum(diffusion_support, 0.0)
        * np.maximum(diffusion_support.T, 0.0)
    )
    diffusion_support /= np.maximum(
        np.max(diffusion_support, axis=1, keepdims=True),
        np.finfo(float).eps,
    )

    direct_signal = np.exp(-np.minimum(safe / scale, 60.0))
    score = (
        direct_signal ** 0.35
        * shortcut_signal ** 0.35
        * (0.25 + reverse_flow) ** 0.15
        * (0.25 + diffusion_support) ** 0.15
    )

    score = np.where(finite_edges, score, 0.0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    off_diagonal = ~np.eye(n, dtype=bool)
    mean_score = float(np.mean(score[off_diagonal]))

    if mean_score > 0.0 and np.isfinite(mean_score):
        score /= mean_score

    return score
