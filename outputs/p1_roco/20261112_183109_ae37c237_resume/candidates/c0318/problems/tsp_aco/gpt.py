import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized edge-priority indicators for stochastic TSP sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return result

    work = np.where(valid, d, np.inf)

    row_sorted = np.sort(work, axis=1)
    col_sorted = np.sort(work, axis=0)

    k = min(5, n - 1)

    row_neighbors = row_sorted[:, :k]
    col_neighbors = col_sorted[:k, :].T

    row_finite = np.isfinite(row_neighbors)
    col_finite = np.isfinite(col_neighbors)

    row_count = row_finite.sum(axis=1)
    col_count = col_finite.sum(axis=1)

    row_sum = np.where(row_finite, row_neighbors, 0.0).sum(axis=1)
    col_sum = np.where(col_finite, col_neighbors, 0.0).sum(axis=1)

    eps = np.finfo(float).eps

    row_scale = np.divide(
        row_sum,
        np.maximum(row_count, 1),
        out=np.ones(n, dtype=float),
        where=row_count > 0,
    )
    col_scale = np.divide(
        col_sum,
        np.maximum(col_count, 1),
        out=np.ones(n, dtype=float),
        where=col_count > 0,
    )

    row_scale = np.maximum(row_scale, eps)
    col_scale = np.maximum(col_scale, eps)

    best_out = row_sorted[:, 0]
    second_out = row_sorted[:, 1] if n > 2 else best_out.copy()

    best_in = col_sorted[0, :]
    second_in = col_sorted[1, :] if n > 2 else best_in.copy()

    safe_best_out = np.where(np.isfinite(best_out), best_out, row_scale)
    safe_second_out = np.where(
        np.isfinite(second_out),
        second_out,
        2.0 * safe_best_out,
    )

    safe_best_in = np.where(np.isfinite(best_in), best_in, col_scale)
    safe_second_in = np.where(
        np.isfinite(second_in),
        second_in,
        2.0 * safe_best_in,
    )

    out_regret = np.clip(
        (safe_second_out - safe_best_out)
        / np.maximum(safe_second_out, eps),
        0.0,
        1.0,
    )
    in_regret = np.clip(
        (safe_second_in - safe_best_in)
        / np.maximum(safe_second_in, eps),
        0.0,
        1.0,
    )

    urgency_out = 1.0 + 0.22 * out_regret
    urgency_in = 1.0 + 0.22 * in_regret

    normalized_distance = np.divide(
        d,
        np.sqrt(row_scale[:, None] * col_scale[None, :]),
        out=np.full_like(d, np.inf),
        where=valid,
    )

    clipped_distance = np.clip(normalized_distance, 0.0, 50.0)

    proximity = np.exp(-clipped_distance / 0.95)

    row_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                row_scale[:, None],
                out=np.full_like(d, np.inf),
                where=valid,
            ),
            0.0,
            50.0,
        )
        / 0.95
    )

    col_quality = np.exp(
        -np.clip(
            np.divide(
                d,
                col_scale[None, :],
                out=np.full_like(d, np.inf),
                where=valid,
            ),
            0.0,
            50.0,
        )
        / 0.95
    )

    mutual_quality = np.sqrt(row_quality * col_quality)

    row_order = np.argsort(work, axis=1, kind="stable")
    col_order = np.argsort(work, axis=0, kind="stable")

    row_rank = np.zeros_like(d, dtype=float)
    col_rank = np.zeros_like(d, dtype=float)

    for rank in range(k):
        row_idx = row_order[:, rank]
        row_rank[np.arange(n), row_idx] = rank + 1.0

        col_idx = col_order[rank, :]
        col_rank[col_idx, np.arange(n)] = rank + 1.0

    rank_quality = 1.0 / np.sqrt(
        (1.0 + 0.09 * row_rank) * (1.0 + 0.09 * col_rank)
    )

    row_competition = np.sum(
        valid & (d <= safe_second_out[:, None]),
        axis=1,
        dtype=float,
    )
    col_competition = np.sum(
        valid & (d <= safe_second_in[None, :]),
        axis=0,
        dtype=float,
    )

    endpoint_rarity = 1.0 / np.sqrt(
        (1.0 + row_competition[:, None])
        * (1.0 + col_competition[None, :])
    )

    row_neighborhood = np.zeros((n, n), dtype=float)
    col_neighborhood = np.zeros((n, n), dtype=float)

    for rank in range(k):
        row_idx = row_order[:, rank]
        col_idx = col_order[rank, :]

        row_neighborhood[np.arange(n), row_idx] = 1.0
        col_neighborhood[col_idx, np.arange(n)] = 1.0

    row_overlap = row_neighborhood @ row_neighborhood.T
    col_overlap = col_neighborhood @ col_neighborhood.T

    row_sizes = np.maximum(row_neighborhood.sum(axis=1), 1.0)
    col_sizes = np.maximum(col_neighborhood.sum(axis=1), 1.0)

    overlap = 0.5 * (
        row_overlap / np.maximum(row_sizes[:, None], 1.0)
        + col_overlap / np.maximum(col_sizes[None, :], 1.0)
    )

    bridge_gain = np.clip(1.0 - overlap, 0.0, 1.0)
    bridge_gain *= np.exp(-np.clip(normalized_distance, 0.0, 8.0) / 2.5)

    score = (
        proximity
        * mutual_quality
        * rank_quality
        * (1.0 + 0.08 * endpoint_rarity)
        * (1.0 + 0.06 * bridge_gain)
        * urgency_out[:, None]
        * urgency_in[None, :]
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    positive_scores = np.where(valid, score, 0.0)
    log_scores = np.full_like(d, -np.inf, dtype=float)
    log_scores[valid] = np.log(np.maximum(positive_scores[valid], eps))

    probabilities = np.zeros_like(d, dtype=float)

    for i in range(n):
        row_valid = valid[i]
        count = int(row_valid.sum())

        if count == 0:
            continue

        row_logits = log_scores[i, row_valid]
        centered = row_logits - np.mean(row_logits)
        dispersion = float(np.std(centered))

        temperature = 0.82 + 0.48 * np.tanh(dispersion)
        scaled = centered / max(temperature, 0.35)

        scaled -= np.max(scaled)
        exp_values = np.exp(np.clip(scaled, -50.0, 50.0))
        soft = exp_values / np.maximum(exp_values.sum(), eps)

        floor = 0.035
        mixed = (1.0 - floor) * soft + floor / count
        probabilities[i, row_valid] = mixed

    return probabilities
