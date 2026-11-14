import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term edge-priority indicators for TSP."""

    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    result = np.zeros_like(d, dtype=float)

    if n < 2:
        return result

    finite = np.isfinite(d)
    valid = finite & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    # Symmetric statistics are used for neighborhood calibration, while
    # the original directed edge values remain available for scoring.
    reciprocal = valid & valid.T
    symmetric = np.where(
        reciprocal,
        0.5 * (d + d.T),
        np.where(valid, d, np.where(valid.T, d.T, np.inf)),
    )
    symmetric[~(valid | valid.T)] = np.inf
    np.fill_diagonal(symmetric, np.inf)

    finite_sym = symmetric[np.isfinite(symmetric)]
    if finite_sym.size == 0:
        return result

    positive = finite_sym[finite_sym > eps]
    global_scale = (
        float(np.median(positive))
        if positive.size
        else max(float(np.median(finite_sym)), 1.0)
    )
    global_scale = max(global_scale, eps)

    ordered = np.sort(symmetric, axis=1)
    degree = max(1, n - 1)

    # Multi-scale local context.
    k = max(1, min(degree, int(np.ceil(np.sqrt(float(n))))))
    nearest = ordered[:, :k]
    local_scale = np.nanmedian(
        np.where(np.isfinite(nearest), nearest, np.nan),
        axis=1,
    )
    local_scale = np.where(
        np.isfinite(local_scale) & (local_scale > eps),
        local_scale,
        global_scale,
    )
    local_scale = np.maximum(local_scale, eps)

    pair_scale = (
        0.62 * np.sqrt(local_scale[:, None] * local_scale[None, :])
        + 0.38 * global_scale
    )
    pair_scale = np.maximum(pair_scale, eps)

    relative = np.divide(
        symmetric,
        pair_scale,
        out=np.full_like(symmetric, np.inf),
        where=np.isfinite(symmetric),
    )

    # Robust distance quality with a softer tail than a pure inverse distance.
    distance_quality = np.exp(-np.clip(relative, 0.0, 40.0))
    distance_quality[~valid] = 0.0

    # Stable rank features from both endpoints.
    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[np.arange(n)[:, None], row_order] = np.arange(
        n, dtype=float
    )[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(
        n, dtype=float
    )[:, None]

    rank_temperature = max(1.0, 0.85 * np.sqrt(float(n)))
    outgoing_rank = np.exp(-row_rank / rank_temperature)
    incoming_rank = np.exp(-col_rank / rank_temperature)
    outgoing_rank[~valid] = 0.0
    incoming_rank[~valid] = 0.0

    mutual_rank = np.sqrt(outgoing_rank * incoming_rank)

    reciprocal_k = max(
        1,
        min(degree, int(np.ceil(np.log2(float(n) + 1.0)))),
    )
    reciprocal_edge = (
        valid
        & (row_rank < reciprocal_k)
        & (col_rank < reciprocal_k)
    )
    reciprocal_bonus = 1.0 + 0.52 * reciprocal_edge.astype(float)

    local_score = distance_quality * (0.58 + 0.42 * mutual_rank)
    local_score[~valid] = 0.0

    # Long-term potential: use a soft top-neighborhood aggregate rather
    # than only the single best continuation.
    continuation_temperature = max(0.20, 0.48 / np.sqrt(float(n)))

    outgoing_logit = np.where(
        valid,
        np.log(np.maximum(local_score, eps))
        / continuation_temperature,
        -np.inf,
    )
    incoming_logit = np.where(
        valid,
        np.log(np.maximum(local_score, eps))
        / continuation_temperature,
        -np.inf,
    )

    def soft_top_mean(logits: np.ndarray, axis: int) -> np.ndarray:
        finite_logits = np.where(np.isfinite(logits), logits, -1e30)
        maximum = np.max(finite_logits, axis=axis, keepdims=True)
        shifted = np.exp(np.clip(finite_logits - maximum, -60.0, 0.0))
        shifted[~np.isfinite(logits)] = 0.0
        count = np.sum(shifted > 0.0, axis=axis)
        weighted = np.sum(shifted, axis=axis)
        value = np.squeeze(maximum, axis=axis)
        value = np.exp(np.clip(value, -60.0, 60.0))
        value *= np.maximum(weighted, eps) / np.maximum(count, 1.0)
        return np.sqrt(np.maximum(value, 0.0))

    outgoing_potential = soft_top_mean(outgoing_logit, axis=1)
    incoming_potential = soft_top_mean(incoming_logit, axis=0)

    # Edge i -> j should leave j with good outgoing options and i with
    # good incoming options.
    continuation = np.sqrt(
        outgoing_potential[None, :] * incoming_potential[:, None]
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    # Scarcity/regret terms protect structurally important edges.
    second_neighbor = ordered[:, min(1, degree - 1)]
    kth_neighbor = ordered[:, k - 1]

    second_neighbor = np.where(
        np.isfinite(second_neighbor),
        second_neighbor,
        global_scale,
    )
    kth_neighbor = np.where(
        np.isfinite(kth_neighbor),
        kth_neighbor,
        global_scale,
    )

    scarcity = np.clip(
        (kth_neighbor - local_scale)
        / np.maximum(kth_neighbor, eps),
        0.0,
        1.0,
    )
    regret = np.clip(
        (second_neighbor - local_scale)
        / np.maximum(second_neighbor, eps),
        0.0,
        1.0,
    )

    scarcity_bonus = 1.0 + 0.30 * (
        scarcity[:, None] + scarcity[None, :]
    )
    regret_bonus = 1.0 + 0.20 * (
        regret[:, None] + regret[None, :]
    )

    score = (
        local_score
        * reciprocal_bonus
        * scarcity_bonus
        * regret_bonus
        * (0.78 + 0.44 * continuation)
    )

    # Mildly penalize unusually long edges without eliminating them.
    score *= np.exp(-0.34 * np.maximum(relative - 1.0, 0.0))
    score[~valid] = 0.0
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.any(score > 0.0):
        valid_count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(valid_count, 1.0)
        return result

    # Global-diversity correction. The exponent is intentionally mild:
    # it compensates for endpoint collapse without erasing score gaps.
    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    row_support = np.sqrt(np.maximum(row_mass, eps))
    col_support = np.sqrt(np.maximum(col_mass, eps))

    diversity_factor = 1.0 / np.sqrt(
        row_support[:, None] * col_support[None, :]
    )
    diversity_factor /= max(float(np.mean(diversity_factor[valid])), eps)

    calibrated = score * (
        0.84 + 0.16 * np.clip(diversity_factor, 0.35, 2.5)
    )
    calibrated[~valid] = 0.0

    # Quality-aware exploration: retain a small amount of mass for edges
    # connecting weakly represented endpoints, rather than uniform noise.
    row_quality = calibrated / np.maximum(
        calibrated.sum(axis=1, keepdims=True),
        eps,
    )
    col_quality = calibrated / np.maximum(
        calibrated.sum(axis=0, keepdims=True),
        eps,
    )

    node_balanced = np.sqrt(
        np.maximum(row_quality, 0.0)
        * np.maximum(col_quality, 0.0)
    )
    node_balanced[~valid] = 0.0

    balanced_total = float(node_balanced.sum())
    if balanced_total > 0.0 and np.isfinite(balanced_total):
        node_balanced /= balanced_total
    else:
        node_balanced = np.zeros_like(calibrated)

    calibrated_total = float(calibrated.sum())
    if not np.isfinite(calibrated_total) or calibrated_total <= 0.0:
        valid_count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(valid_count, 1.0)
        return result

    calibrated /= calibrated_total

    exploration_weight = 0.025
    score = (
        (1.0 - exploration_weight) * calibrated
        + exploration_weight * node_balanced
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    total = float(score.sum())
    if not np.isfinite(total) or total <= 0.0:
        valid_count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(valid_count, 1.0)
        return result

    return score / total
