import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = d.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    d = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=np.inf)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid = np.isfinite(d) & (d > 0)
    valid &= valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    eps = np.finfo(float).eps
    positive = d[valid]
    global_scale = max(float(np.median(positive)), eps)

    safe_d = d.copy()
    safe_d[~valid] = global_scale * 10.0
    np.maximum(safe_d, eps, out=safe_d)

    masked_d = np.where(valid, d, np.inf)
    k = min(2, n - 2)

    local_scale = np.partition(masked_d, k, axis=1)[:, k]
    row_has_valid = np.any(valid, axis=1)
    local_scale = np.where(
        row_has_valid & np.isfinite(local_scale) & (local_scale > 0),
        local_scale,
        global_scale,
    )

    endpoint_scale = np.sqrt(
        np.maximum(local_scale[:, None] * local_scale[None, :], eps)
    )
    relative_length = safe_d / endpoint_scale

    local_density = np.mean(
        valid & (safe_d <= 1.5 * local_scale[:, None]),
        axis=1,
    )
    density_penalty = (
        local_density[:, None] + local_density[None, :]
    ) * np.minimum(relative_length, 2.0)

    log_score = relative_length
    row_relative = safe_d / np.maximum(local_scale[:, None], eps)
    log_score += 0.5 * (row_relative + row_relative.T)
    log_score *= -0.85

    log_score -= 0.35 * (safe_d / global_scale)
    log_score -= 0.16 * density_penalty

    log_local_scale = np.log(np.maximum(local_scale, eps))
    log_score -= 0.18 * np.abs(
        log_local_scale[:, None] - log_local_scale[None, :]
    )

    log_score[~valid] = -np.inf
    finite = np.isfinite(log_score)

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    shift = np.max(log_score[finite])

    score = np.zeros_like(d)
    score[finite] = np.exp(log_score[finite] - shift)

    score = 0.5 * (score + score.T)
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    peak = float(np.max(score))
    if peak > 0.0:
        score /= peak

    return score.astype(float, copy=False)
