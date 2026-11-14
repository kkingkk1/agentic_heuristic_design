import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return normalized, diverse edge priorities for stochastic TSP sampling."""
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
    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif finite_values.size:
        global_scale = float(np.median(finite_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0

    eps = np.finfo(float).eps

    degree = valid.sum(axis=1).astype(float)
    safe_degree = np.maximum(degree, 1.0)

    sorted_rows = np.sort(work, axis=1)

    best = sorted_rows[:, 0]
    fallback = np.maximum(global_scale, 1.0)
    best = np.where(np.isfinite(best), best, fallback)

    # Adaptive neighborhood size: use a small neighborhood for clear
    # local structure and a larger one for dense or ambiguous structure.
    if n <= 3:
        k = 1
    else:
        density = np.clip(np.mean(valid, axis=1), 0.0, 1.0)
        ambiguity = np.clip(
            np.divide(
                np.where(
                    np.isfinite(sorted_rows[:, 1]),
                    sorted_rows[:, 1],
                    best,
                ),
                np.maximum(best, eps),
            )
            - 1.0,
            0.0,
            3.0,
        )
        adaptive = 2.0 + 2.0 * density + 1.5 * ambiguity
        k = int(np.clip(np.ceil(np.median(adaptive)), 2, min(6, n - 1)))

    neighbor_values = sorted_rows[:, :k]
    neighbor_values = np.where(
        np.isfinite(neighbor_values),
        neighbor_values,
        best[:, None],
    )

    # Arithmetic local scale avoids excessive concentration caused by
    # reciprocal or multiplicative distance transforms.
    local_scale = np.mean(neighbor_values, axis=1)
    local_scale = np.maximum(local_scale, eps)

    local_ratio = np.divide(d, local_scale[:, None])
    global_ratio = np.divide(d, global_scale)

    local_quality = np.exp(
        -np.clip(local_ratio, 0.0, 40.0) / 0.78
    )
    global_quality = np.exp(
        -np.clip(global_ratio, 0.0, 40.0) / 1.80
    )

    local_quality = np.where(valid, local_quality, 0.0)
    global_quality = np.where(valid, global_quality, 0.0)

    # Count near alternatives using an adaptive radius. This measures
    # whether an endpoint has genuine long-term substitution options.
    radius = np.maximum(
        1.20 * local_scale,
        0.55 * (local_scale + best),
    )

    near_count = np.sum(
        valid & (d <= radius[:, None]),
        axis=1,
        dtype=float,
    )

    # Low alternative count means an endpoint is structurally fragile.
    # The transformation is bounded and only used as a mild correction.
    scarcity = 1.0 / np.sqrt(np.maximum(near_count, 1.0))
    scarcity = scarcity / np.maximum(np.max(scarcity), 1.0)

    endpoint_scarcity = 0.5 * (
        scarcity[:, None] + scarcity[None, :]
    )

    # Rank gate: long-term potential should not rescue clearly poor edges.
    row_rank = np.sum(
        valid & (d <= radius[:, None]),
        axis=1,
        dtype=float,
    )
    rank_gate = np.where(
        valid,
        np.clip(
            np.divide(
                radius[:, None] - d,
                np.maximum(radius[:, None], eps),
            ),
            -1.0,
            1.0,
        ),
        0.0,
    )
    rank_gate = 0.5 + 0.5 * rank_gate

    # Front-loaded long-term potential: reward edges that are locally
    # competitive and connect endpoints with few alternatives.
    long_term = (
        0.58 * local_quality
        + 0.27 * global_quality
        + 0.15 * rank_gate
    )
    long_term *= 1.0 + 0.16 * endpoint_scarcity
    long_term = np.where(valid, long_term, 0.0)

    # A restrained arithmetic blend preserves medium-quality alternatives.
    score = (
        0.68 * local_quality
        + 0.16 * global_quality
        + 0.16 * long_term
    )

    # Diversity correction is active mainly for promising edges and is
    # capped so it cannot overturn the distance ordering globally.
    diversity_factor = (
        1.0
        + 0.14 * endpoint_scarcity * rank_gate
    )
    score *= diversity_factor

    # Adaptive exploration: concentrated neighborhoods receive more
    # uniform support, while already diverse rows receive less.
    concentration = np.divide(
        np.max(local_quality, axis=1),
        np.maximum(np.sum(local_quality, axis=1), eps),
    )
    exploration = np.clip(
        0.035 + 0.10 * concentration + 0.025 / np.sqrt(safe_degree),
        0.04,
        0.16,
    )

    degree_prior = np.divide(
        valid.astype(float),
        safe_degree[:, None],
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)

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

    score = (
        (1.0 - exploration[:, None]) * score
        + exploration[:, None] * degree_prior
    )

    score = np.where(valid, np.maximum(score, 0.0), 0.0)
    np.fill_diagonal(score, 0.0)

    totals = score.sum(axis=1, keepdims=True)
    return np.divide(
        score,
        totals,
        out=result,
        where=totals > 0.0,
    )
