import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = np.nan_to_num(
        distance_matrix,
        nan=np.inf,
        posinf=np.inf,
        neginf=np.inf,
    ).copy()

    np.fill_diagonal(d, np.inf)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    positive = d[finite & (d > 0)]
    scale = np.median(positive) if positive.size else 1.0
    scale = max(float(scale), np.finfo(float).eps)

    safe_d = np.where(np.isfinite(d), np.maximum(d, 0.0), scale * 1e6)

    # Direct distance affinity.
    direct = 1.0 / (1.0 + safe_d / scale)

    # Row-wise rank affinity: short edges receive larger scores.
    order = np.argsort(safe_d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    ranks[np.arange(n)[:, None], order] = rank_values[None, :]
    rank_affinity = np.exp(-ranks / max(1.0, np.sqrt(n)))

    # Emphasize edges that are simultaneously competitive from both endpoints.
    mutual = np.sqrt(rank_affinity * rank_affinity.T)

    # Local scarcity: an edge is valuable when it is among the better choices
    # for both endpoints relative to their local distance distributions.
    row_mean = np.sum(np.where(np.isfinite(d), safe_d, 0.0), axis=1)
    row_count = np.sum(np.isfinite(d), axis=1).clip(min=1)
    local_mean = row_mean / row_count
    scarcity = 1.0 / (1.0 + safe_d / (local_mean[:, None] + scale * 1e-12))
    mutual_scarcity = np.sqrt(scarcity * scarcity.T)

    # Triangle consistency: prefer edges whose endpoints share short detours
    # through a third node, reducing disconnected or geometrically implausible
    # selections.
    two_hop = safe_d[:, :, None] + safe_d[None, :, :]
    best_two_hop = np.min(two_hop, axis=1)
    triangle_consistency = 1.0 / (
        1.0 + np.maximum(safe_d - best_two_hop, 0.0) / scale
    )

    score = (
        0.35 * direct
        + 0.25 * mutual
        + 0.25 * mutual_scarcity
        + 0.15 * triangle_consistency
    )

    # Preserve symmetry for symmetric TSP instances while remaining valid for
    # asymmetric inputs.
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    score[~np.isfinite(score)] = 0.0
    minimum = np.min(score)
    maximum = np.max(score)

    if maximum > minimum:
        score = (score - minimum) / (maximum - minimum)

    return score.astype(float, copy=False)
