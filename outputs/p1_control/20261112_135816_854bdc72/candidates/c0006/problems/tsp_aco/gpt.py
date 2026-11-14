import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Generate edge-selection priors for a stochastic TSP solver.

    Larger values indicate edges that are more promising to include.
    The returned array has the same shape as distance_matrix.
    """
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=float)

    positive = d[finite & (d > 0)]
    fallback = float(np.max(positive)) if positive.size else 1.0
    safe_d = np.where(finite, d, fallback)
    safe_d = np.maximum(safe_d, 0.0)
    np.fill_diagonal(safe_d, np.inf)

    finite_offdiag = safe_d[np.isfinite(safe_d)]
    if finite_offdiag.size == 0:
        return np.zeros_like(safe_d, dtype=float)

    low = float(np.min(finite_offdiag))
    high = float(np.percentile(finite_offdiag, 90.0))
    scale = max(high - low, np.finfo(float).eps)

    normalized = np.clip((safe_d - low) / scale, 0.0, 8.0)

    # Local attractiveness from each source node.
    row_logits = -normalized
    row_logits -= np.max(np.where(np.isfinite(row_logits), row_logits, -np.inf), axis=1, keepdims=True)
    row_exp = np.exp(np.where(np.isfinite(row_logits), row_logits, -np.inf))
    row_exp /= np.maximum(row_exp.sum(axis=1, keepdims=True), np.finfo(float).eps)

    # Local attractiveness when entering each destination node.
    col_logits = -normalized
    col_logits -= np.max(np.where(np.isfinite(col_logits), col_logits, -np.inf), axis=0, keepdims=True)
    col_exp = np.exp(np.where(np.isfinite(col_logits), col_logits, -np.inf))
    col_exp /= np.maximum(col_exp.sum(axis=0, keepdims=True), np.finfo(float).eps)

    np.fill_diagonal(row_exp, 0.0)
    np.fill_diagonal(col_exp, 0.0)

    # Mutual local preference: an edge is favored when both endpoints
    # consider the corresponding connection unusually attractive.
    mutual = np.sqrt(row_exp * col_exp.T)

    # Competition relief: favor edges that are relatively distinctive
    # among the alternatives available at both endpoints.
    row_competition = row_exp / np.maximum(row_exp.sum(axis=1, keepdims=True), np.finfo(float).eps)
    col_competition = col_exp / np.maximum(col_exp.sum(axis=0, keepdims=True), np.finfo(float).eps)
    distinctiveness = np.sqrt(row_competition * col_competition.T)

    # Directness advantage over the best two-edge detour i -> k -> j.
    finite_d = np.where(np.isfinite(safe_d), safe_d, fallback * 2.0)
    detours = finite_d[:, :, None] + finite_d[None, :, :]
    best_detour = np.min(detours, axis=1)
    direct_advantage = np.maximum(best_detour - finite_d, 0.0)
    direct_scale = np.percentile(direct_advantage[np.isfinite(direct_advantage)], 90.0)
    direct_scale = max(float(direct_scale), np.finfo(float).eps)
    directness = np.clip(direct_advantage / direct_scale, 0.0, 1.0)

    # Robust inverse-distance signal.
    inverse_distance = 1.0 / (1.0 + normalized)

    score = (
        0.35 * mutual
        + 0.25 * distinctiveness
        + 0.25 * directness
        + 0.15 * inverse_distance
    )

    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score
