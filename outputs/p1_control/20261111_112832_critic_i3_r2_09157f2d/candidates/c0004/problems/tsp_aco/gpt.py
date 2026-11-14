import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite_nonnegative = np.isfinite(d) & (d >= 0)
    valid = d[finite_nonnegative]

    if valid.size == 0:
        return np.zeros((n, n), dtype=float)

    replacement = float(np.max(valid))
    if not np.isfinite(replacement) or replacement <= 0:
        replacement = 1.0

    clean = np.where(finite_nonnegative, d, replacement)
    clean = np.maximum(clean, 0.0)
    np.fill_diagonal(clean, np.inf)

    finite_edges = clean[np.isfinite(clean)]
    positive_edges = finite_edges[finite_edges > 0]

    if positive_edges.size == 0:
        return np.zeros((n, n), dtype=float)

    global_scale = float(np.median(positive_edges))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = 1.0

    neighbor_count = min(max(2, int(np.sqrt(n))), n - 1)
    sorted_distances = np.sort(clean, axis=1)
    local_scale = sorted_distances[:, neighbor_count - 1]

    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, global_scale * 1e-6)

    pair_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], 1e-24)
    )

    direct = np.exp(-np.square(clean / pair_scale))
    direct[~np.isfinite(direct)] = 0.0
    np.fill_diagonal(direct, 0.0)

    direct = 0.5 * (direct + direct.T)

    row_mass = np.sum(direct, axis=1, keepdims=True)
    transition = np.divide(
        direct,
        row_mass,
        out=np.zeros_like(direct),
        where=row_mass > 0,
    )

    two_step = transition @ transition
    two_step = 0.5 * (two_step + two_step.T)
    np.fill_diagonal(two_step, 0.0)

    direct_scale = np.maximum(np.max(direct, axis=1, keepdims=True), 1e-12)
    diffusion_scale = np.maximum(np.max(two_step, axis=1, keepdims=True), 1e-12)

    direct_norm = direct / np.sqrt(direct_scale @ direct_scale.T)
    diffusion_norm = two_step / np.sqrt(diffusion_scale @ diffusion_scale.T)

    score = np.sqrt(np.maximum(direct_norm, 0.0) * np.maximum(diffusion_norm, 0.0))
    score += 0.25 * direct_norm
    score = 0.5 * (score + score.T)

    finite_score = score[np.isfinite(score)]
    if finite_score.size:
        upper = float(np.percentile(finite_score, 99))
        if np.isfinite(upper) and upper > 0:
            score = np.clip(score / upper, 0.0, 1.0)

    np.fill_diagonal(score, 0.0)
    return np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
