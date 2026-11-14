import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a globally balanced, two-hop-supported TSP edge prior."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros_like(raw, dtype=float)

    finite = np.isfinite(raw) & (raw >= 0)
    finite &= finite.T
    np.fill_diagonal(finite, False)

    if not np.any(finite):
        return np.zeros((n, n), dtype=float)

    d = np.where(finite, 0.5 * (raw + raw.T), np.inf)

    positive = d[finite & (d > 0)]
    if positive.size:
        temperature = float(np.median(positive))
    else:
        temperature = 1.0

    temperature = max(
        temperature,
        np.finfo(float).eps,
    )

    finite_values = d[finite]
    shift = float(np.min(finite_values))

    scaled = np.maximum(d - shift, 0.0) / temperature
    scaled = np.minimum(scaled, 60.0)

    kernel = np.zeros((n, n), dtype=float)
    kernel[finite] = np.exp(-scaled[finite])
    np.fill_diagonal(kernel, 0.0)

    active_rows = np.sum(kernel > 0.0, axis=1) > 0
    if not np.any(active_rows):
        return np.zeros((n, n), dtype=float)

    balanced = kernel.copy()

    for _ in range(80):
        row_sum = balanced.sum(axis=1)
        row_scale = np.ones(n, dtype=float)
        row_scale[active_rows] = np.divide(
            1.0,
            row_sum[active_rows],
            out=np.zeros(np.count_nonzero(active_rows), dtype=float),
            where=row_sum[active_rows] > 0,
        )
        balanced *= row_scale[:, None]

        col_sum = balanced.sum(axis=0)
        col_scale = np.ones(n, dtype=float)
        active_cols = col_sum > 0
        col_scale[active_cols] = np.divide(
            1.0,
            col_sum[active_cols],
            out=np.zeros(np.count_nonzero(active_cols), dtype=float),
            where=col_sum[active_cols] > 0,
        )
        balanced *= col_scale[None, :]

    balanced = 0.5 * (balanced + balanced.T)
    balanced[~finite] = 0.0
    np.fill_diagonal(balanced, 0.0)

    two_hop = balanced @ balanced
    two_hop[~finite] = 0.0
    np.fill_diagonal(two_hop, 0.0)

    support_max = float(np.max(two_hop))
    if support_max > 0.0:
        two_hop /= support_max

    direct_max = float(np.max(balanced))
    if direct_max > 0.0:
        direct = balanced / direct_max
    else:
        direct = balanced

    score = direct * (0.35 + 0.65 * two_hop)
    score = 0.5 * (score + score.T)
    score[~finite] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
