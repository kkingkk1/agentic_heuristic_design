import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return row-normalized heuristic indicators for stochastic TSP sampling."""
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

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)
    sorted_rows = np.sort(work, axis=1)

    max_k = min(4, n - 1)
    nearest = sorted_rows[:, :max_k]
    finite_nearest = np.isfinite(nearest)
    counts = finite_nearest.sum(axis=1)

    first = nearest[:, 0]
    safe_first = np.where(np.isfinite(first), first, 1.0)

    if max_k >= 4:
        reference = nearest[:, 3]
        reference_valid = np.isfinite(reference)
        threshold = 0.75
        high_k = 4
    elif max_k == 3:
        reference = nearest[:, 2]
        reference_valid = np.isfinite(reference)
        threshold = 0.75
        high_k = 3
    else:
        reference = nearest[:, 0]
        reference_valid = np.isfinite(reference)
        threshold = 0.0
        high_k = 1

    spread = np.divide(
        reference - safe_first,
        np.maximum(safe_first, eps),
        out=np.full(n, np.inf, dtype=float),
        where=reference_valid,
    )

    adaptive_k = np.where(spread <= threshold, high_k, 2)
    adaptive_k = np.minimum(adaptive_k, np.maximum(counts, 1))
    adaptive_k = np.maximum(adaptive_k, 1)

    positions = np.arange(max_k)[None, :]
    selected = finite_nearest & (positions < adaptive_k[:, None])

    local_sum = np.where(selected, nearest, 0.0).sum(axis=1)
    selected_counts = selected.sum(axis=1)

    local_scale = np.divide(
        local_sum,
        np.maximum(selected_counts, 1),
        out=np.ones(n, dtype=float),
        where=selected_counts > 0,
    )
    local_scale = np.maximum(local_scale, eps)

    row_scale = local_scale[:, None]
    col_scale = local_scale[None, :]
    geometric_scale = np.sqrt(row_scale * col_scale)

    normalized_distance = np.divide(
        d,
        geometric_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    row_ratio = np.divide(
        d,
        row_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )
    col_ratio = np.divide(
        d,
        col_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance_quality = np.exp(
            -np.clip(normalized_distance, 0.0, 50.0) / 0.82
        )
        row_quality = np.exp(
            -np.clip(row_ratio, 0.0, 50.0) / 0.88
        )
        col_quality = np.exp(
            -np.clip(col_ratio, 0.0, 50.0) / 0.88
        )

        geometric_quality = np.sqrt(row_quality * col_quality)
        arithmetic_quality = 0.5 * (row_quality + col_quality)

        directional_quality = (
            0.87 * geometric_quality
            + 0.13 * arithmetic_quality
        )

        score = distance_quality * directional_quality
        score = np.where(valid, score, 0.0)

        # Bounded local scarcity correction.
        near_radius = 1.35 * local_scale
        near_count = np.sum(
            valid & (d <= near_radius[:, None]),
            axis=1,
            dtype=float,
        )

        scarcity = 1.0 / np.sqrt(np.maximum(near_count, 1.0))
        scarcity /= np.maximum(np.max(scarcity), 1.0)

        endpoint_scarcity = 0.5 * (
            scarcity[:, None] + scarcity[None, :]
        )

        local_competitiveness = np.exp(
            -np.clip(
                np.divide(
                    d,
                    np.maximum(local_scale[:, None], eps),
                ),
                0.0,
                50.0,
            )
            / 1.10
        )
        local_competitiveness = np.where(valid, local_competitiveness, 0.0)

        rank_gate = np.clip(
            local_competitiveness / 0.55,
            0.0,
            1.0,
        )

        score *= (
            1.0
            + 0.09 * endpoint_scarcity * rank_gate
        )

        row_max = score.max(axis=1, keepdims=True)
        row_sum = score.sum(axis=1, keepdims=True)
        concentration = np.divide(
            row_max,
            np.maximum(row_sum, eps),
        )

        exploration = np.clip(
            0.006 + 0.018 * concentration,
            0.006,
            0.026,
        )

        degree = valid.sum(axis=1).astype(float)
        safe_degree = np.maximum(degree, 1.0)
        degree_prior = valid.astype(float) / safe_degree[:, None]

        score = (
            (1.0 - exploration[:, None]) * score
            + exploration[:, None] * degree_prior
        )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    symmetric_input = (
        np.array_equal(valid, valid.T)
        and np.allclose(
            np.where(valid, d, 0.0),
            np.where(valid.T, d.T, 0.0),
            rtol=1e-10,
            atol=1e-12,
        )
    )

    if symmetric_input:
        score = np.sqrt(
            np.maximum(score, 0.0)
            * np.maximum(score.T, 0.0)
        )
        score = np.where(valid, score, 0.0)
        np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)

    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
