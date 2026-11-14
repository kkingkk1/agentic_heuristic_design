import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return entropy-balanced directed edge priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(raw) & (raw >= 0.0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    values = raw[valid]
    positive = values[values > 0.0]

    if positive.size:
        scale = float(np.median(positive))
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive)) if positive.size else 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    spread = float(q90 - q10)
    if not np.isfinite(spread) or spread <= 0.0:
        spread = safe_scale

    replacement = float(q90 + 8.0 * max(spread, safe_scale))
    if not np.isfinite(replacement) or replacement <= 0.0:
        replacement = safe_scale

    clean = np.where(valid, raw, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    finite_clean = clean[valid]
    centered = np.maximum(finite_clean - q50, 0.0)
    adaptive_factor = float(
        np.clip(0.55 + 0.35 * spread / max(abs(float(q50)), safe_scale), 0.45, 1.20)
    )

    kernel = np.zeros((n, n), dtype=float)

    for i in range(n):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue

        row_values = clean[i, row_valid]
        row_positive = row_values[row_values > 0.0]

        if row_positive.size:
            local_scale = float(np.median(row_positive))
        else:
            local_scale = safe_scale

        if not np.isfinite(local_scale) or local_scale <= 0.0:
            local_scale = safe_scale

        local_spread = float(np.quantile(row_values, 0.75) - np.quantile(row_values, 0.25))
        local_scale = max(
            local_scale,
            local_spread,
            safe_scale * 0.05,
        )

        temperature = max(
            local_scale * adaptive_factor,
            safe_scale * 0.08,
        )

        logits = -(clean[i] / temperature)
        logits -= np.min(logits[row_valid])

        row_kernel = np.zeros(n, dtype=float)
        row_kernel[row_valid] = np.exp(
            np.clip(logits[row_valid], -60.0, 0.0)
        )

        row_sum = float(np.sum(row_kernel))
        if np.isfinite(row_sum) and row_sum > 0.0:
            kernel[i] = row_kernel / row_sum

    kernel[~valid] = 0.0

    flow = kernel.copy()

    for _ in range(32):
        row_sum = flow.sum(axis=1, keepdims=True)
        row_sum = np.maximum(row_sum, eps)
        flow = flow / row_sum
        flow[~valid] = 0.0

        col_sum = flow.sum(axis=0, keepdims=True)
        col_sum = np.maximum(col_sum, eps)
        flow = flow / col_sum
        flow[~valid] = 0.0

    row_sum = flow.sum(axis=1, keepdims=True)
    transition = flow / np.maximum(row_sum, eps)
    transition[~valid] = 0.0

    stationary = np.full(n, 1.0 / n, dtype=float)

    for _ in range(48):
        updated = stationary @ transition
        total = float(np.sum(updated))

        if not np.isfinite(total) or total <= 0.0:
            break

        updated /= total
        if np.max(np.abs(updated - stationary)) < 1e-12:
            stationary = updated
            break

        stationary = updated

    incoming = stationary[:, None] * transition
    incoming_scale = np.maximum(
        incoming.sum(axis=0, keepdims=True),
        eps,
    )
    incoming_share = incoming / incoming_scale

    two_step = transition @ transition
    two_step = np.nan_to_num(
        two_step,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    closure = np.sqrt(
        np.maximum(
            transition * transition.T,
            0.0,
        )
    )

    row_entropy = -np.sum(
        transition * np.log(np.maximum(transition, eps)),
        axis=1,
    )
    row_entropy /= max(float(np.log(max(n - 1, 2))), 1.0)
    row_entropy = np.clip(row_entropy, 0.0, 1.0)

    source_importance = np.sqrt(
        np.maximum(stationary[:, None] * stationary[None, :], 0.0)
    )
    source_max = float(np.max(source_importance))
    if source_max > 0.0 and np.isfinite(source_max):
        source_importance /= source_max

    competition = np.zeros((n, n), dtype=float)
    for i in range(n):
        row = clean[i]
        valid_indices = np.flatnonzero(valid[i])

        if valid_indices.size == 0:
            continue

        local = row[valid_indices]
        order = np.argsort(local, kind="stable")
        ranks = np.empty(valid_indices.size, dtype=float)
        ranks[order] = np.arange(valid_indices.size, dtype=float)

        rank_temperature = max(
            1.0,
            0.5 * np.sqrt(float(valid_indices.size)),
        )

        competition[i, valid_indices] = np.exp(
            -ranks / rank_temperature
        )

    competition /= max(float(np.max(competition)), eps)

    score = (
        0.42 * flow
        + 0.20 * incoming_share
        + 0.16 * closure
        + 0.12 * two_step
        + 0.06 * source_importance
        + 0.04 * competition
    )

    entropy_factor = np.sqrt(
        np.maximum(
            (1.0 - 0.35 * row_entropy[:, None])
            * (1.0 - 0.35 * row_entropy[None, :]),
            0.0,
        )
    )
    score *= 0.75 + 0.25 * entropy_factor

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum
    else:
        score.fill(0.0)

    return np.nan_to_num(
        score,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
