import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Construct normalized edge-selection priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    diagonal = np.eye(n, dtype=bool)

    finite_values = raw[np.isfinite(raw)]
    if finite_values.size == 0:
        return np.zeros((n, n), dtype=float)

    positive_values = finite_values[finite_values > 0.0]
    scale = (
        float(np.median(positive_values))
        if positive_values.size
        else float(np.median(np.abs(finite_values)))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    finite_max = float(np.max(finite_values))
    if not np.isfinite(finite_max):
        finite_max = scale

    replacement = max(finite_max + 10.0 * scale, 10.0 * scale)
    clean = np.where(np.isfinite(raw), raw, replacement)
    clean = np.maximum(clean, 0.0)

    # Use a symmetric structural distance while retaining a conservative
    # penalty for invalid or missing edges.
    clean = 0.5 * (clean + clean.T)
    np.fill_diagonal(clean, replacement)

    offdiag = ~diagonal
    valid_offdiag = np.isfinite(raw) & np.isfinite(raw.T) & offdiag
    valid_distances = clean[valid_offdiag]

    if valid_distances.size == 0:
        return np.zeros((n, n), dtype=float)

    eps = max(np.finfo(float).eps * scale, 1e-12)

    # Robust global temperature and endpoint-specific local scales.
    global_temperature = max(
        0.10 * scale,
        float(np.median(np.abs(valid_distances - np.median(valid_distances)))),
        eps,
    )

    masked = np.where(offdiag, clean, np.inf)
    nearest = np.min(masked, axis=1)
    nearest = np.where(np.isfinite(nearest), nearest, scale)

    row_median = np.median(
        np.where(offdiag, clean, np.nan),
        axis=1
    )
    row_median = np.nan_to_num(row_median, nan=scale, posinf=scale, neginf=eps)
    row_median = np.maximum(row_median, eps)

    local_temperature = np.maximum(
        0.5 * (row_median + nearest),
        0.25 * scale,
    )
    pair_temperature = np.sqrt(
        local_temperature[:, None] * local_temperature[None, :]
    )
    pair_temperature = np.maximum(pair_temperature, global_temperature)

    # Absolute short-edge attraction.
    shortness = np.exp(-np.minimum(clean / pair_temperature, 60.0))
    shortness[~offdiag] = 0.0

    # Row-wise and column-wise competitive preference.
    row_logits = -clean / local_temperature[:, None]
    row_logits = np.clip(row_logits, -60.0, 0.0)
    row_preference = np.exp(row_logits)
    row_preference[~offdiag] = 0.0
    row_preference /= np.maximum(
        row_preference.sum(axis=1, keepdims=True),
        eps,
    )

    mutual_preference = np.sqrt(
        np.maximum(row_preference, 0.0)
        * np.maximum(row_preference.T, 0.0)
    )

    # Relative regret: an edge is valuable when its alternatives are poor.
    regret = np.maximum(clean - nearest[:, None], 0.0)
    regret /= np.maximum(row_median[:, None], eps)
    scarcity = 1.0 / (1.0 + regret)
    scarcity = np.sqrt(scarcity * scarcity.T)
    scarcity[~offdiag] = 0.0

    # Rank evidence is less sensitive to outliers than raw distances.
    rank_order = np.argsort(clean, axis=1, kind="stable")
    ranks = np.empty_like(rank_order, dtype=float)
    ranks[np.arange(n)[:, None], rank_order] = np.arange(n, dtype=float)
    rank_score = np.exp(-ranks / max(1.0, np.sqrt(float(n))))
    rank_score[~offdiag] = 0.0
    mutual_rank = np.sqrt(rank_score * rank_score.T)

    # Degree-balancing factor discourages excessive concentration at one node.
    concentration = np.sqrt(
        np.maximum(row_preference.sum(axis=1), eps)
        * np.maximum(row_preference.sum(axis=0), eps)
    )
    concentration = concentration / max(float(np.max(concentration)), eps)
    balance = np.sqrt(
        np.maximum(concentration[:, None], eps)
        * np.maximum(concentration[None, :], eps)
    )

    score = (
        0.38 * shortness
        + 0.27 * mutual_preference
        + 0.20 * scarcity
        + 0.15 * mutual_rank
    )
    score *= 0.85 + 0.15 * balance
    score = 0.5 * (score + score.T)

    score[~valid_offdiag] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
