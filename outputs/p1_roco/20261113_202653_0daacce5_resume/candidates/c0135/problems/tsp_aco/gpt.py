import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return conservative but diversified edge-selection priors for TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(distances)
    valid = finite & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    safe = np.where(valid, distances, np.inf)

    values = safe[valid]
    positive = values[values > 0.0]

    if positive.size:
        median = float(np.median(positive))
        mad = float(np.median(np.abs(positive - median)))
        scale = max(median, 1.4826 * mad, np.finfo(float).eps)
    else:
        scale = 1.0

    eps = np.finfo(float).eps * max(scale, 1.0)
    normalized = np.zeros((n, n), dtype=float)
    normalized[valid] = np.minimum(
        safe[valid] / (scale + eps),
        40.0,
    )

    thermal = np.exp(-normalized)
    reciprocal = 1.0 / (1.0 + normalized)
    attraction = 0.68 * thermal + 0.32 * reciprocal
    attraction[~valid] = 0.0

    row_best = np.min(safe, axis=1, keepdims=True)
    col_best = np.min(safe, axis=0, keepdims=True)

    row_gap = np.zeros((n, n), dtype=float)
    col_gap = np.zeros((n, n), dtype=float)
    row_gap[valid] = np.maximum(
        safe[valid] - row_best.repeat(n, axis=1)[valid],
        0.0,
    ) / (scale + eps)
    col_gap[valid] = np.maximum(
        safe[valid] - col_best.repeat(n, axis=0)[valid],
        0.0,
    ) / (scale + eps)

    competition = 1.0 / (
        1.0
        + 0.42 * np.minimum(row_gap, 8.0)
        + 0.42 * np.minimum(col_gap, 8.0)
    )
    competition[~valid] = 0.0

    scores = attraction * competition
    scores[~valid] = 0.0

    if n >= 3:
        row_sorted = np.sort(safe, axis=1)
        col_sorted = np.sort(safe, axis=0)

        row_second = row_sorted[:, 1][:, None]
        col_second = col_sorted[1, :][None, :]

        finite_second = np.isfinite(row_second) & np.isfinite(col_second)
        regret = np.zeros((n, n), dtype=float)
        regret[valid] = np.maximum(
            row_second.repeat(n, axis=1)[valid] - safe[valid],
            0.0,
        ) / (scale + eps)
        regret[valid] += np.maximum(
            col_second.repeat(n, axis=0)[valid] - safe[valid],
            0.0,
        ) / (scale + eps)

        regret = 1.0 + 0.10 * np.tanh(np.minimum(regret, 6.0))
        regret[~finite_second] = 1.0
        regret[~valid] = 1.0
        scores *= regret

    overlap = valid & valid.T
    if np.any(overlap):
        difference = np.abs(distances - distances.T)
        symmetric_error = float(np.median(difference[overlap]))
        if symmetric_error <= 0.05 * max(scale, eps):
            paired = np.sqrt(
                np.maximum(scores, 0.0) * np.maximum(scores.T, 0.0)
            )
            scores[overlap] = (
                0.80 * scores[overlap] + 0.20 * paired[overlap]
            )

    # Small exploratory signal: bounded two-hop bypass savings.
    if n >= 3:
        candidate_count = min(
            max(2, int(np.sqrt(n)) + 1),
            n - 1,
        )
        two_hop = np.full((n, n), np.inf, dtype=float)

        for i in range(n):
            neighbors = np.flatnonzero(valid[i])
            if neighbors.size == 0:
                continue

            order = neighbors[np.argsort(safe[i, neighbors])]
            for middle in order[:candidate_count]:
                via = safe[i, middle] + safe[middle]
                two_hop[i] = np.minimum(two_hop[i], via)

        np.fill_diagonal(two_hop, np.inf)

        savings = np.zeros((n, n), dtype=float)
        eligible = valid & np.isfinite(two_hop)
        savings[eligible] = np.maximum(
            two_hop[eligible] - safe[eligible],
            0.0,
        ) / (scale + eps)
        savings = np.minimum(savings, 8.0)

        # Keep this contribution deliberately subordinate to the exploitative core.
        scores *= 1.0 + 0.06 * np.tanh(savings)

    scores[~valid] = 0.0

    # Mild endpoint balancing, followed by a conservative mixture.
    balanced = scores.copy()
    for _ in range(2):
        row_sum = balanced.sum(axis=1, keepdims=True)
        row_factor = 1.0 / np.sqrt(np.maximum(row_sum, eps))
        active_rows = row_sum[:, 0] > 0.0

        if np.any(active_rows):
            reference = float(np.median(row_factor[active_rows]))
            row_factor = np.clip(
                row_factor / max(reference, eps),
                0.82,
                1.22,
            )
            balanced *= row_factor

        col_sum = balanced.sum(axis=0, keepdims=True)
        col_factor = 1.0 / np.sqrt(np.maximum(col_sum, eps))
        active_cols = col_sum[0] > 0.0

        if np.any(active_cols):
            reference = float(np.median(col_factor[0, active_cols]))
            col_factor = np.clip(
                col_factor / max(reference, eps),
                0.82,
                1.22,
            )
            balanced *= col_factor

        balanced[~valid] = 0.0

    scores = 0.84 * scores + 0.16 * balanced
    scores[~valid] = 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0 and np.isfinite(maximum):
        scores /= maximum

    np.fill_diagonal(scores, 0.0)
    return scores
