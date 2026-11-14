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

    valid = np.isfinite(d) & (d >= 0.0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    work = np.where(valid, d, np.inf)

    positive = d[valid & (d > eps)]
    if positive.size:
        global_scale = float(np.median(positive))
    else:
        finite_values = d[valid]
        global_scale = (
            float(np.median(finite_values))
            if finite_values.size
            else 1.0
        )
    global_scale = max(global_scale, eps)

    degree = max(1, n - 1)
    local_k = max(1, min(degree, int(np.ceil(np.sqrt(float(n))))))

    outgoing_scale = np.full(n, global_scale, dtype=float)
    incoming_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        row_values = d[i, valid[i]]
        if row_values.size:
            row_values = np.sort(row_values)
            selected = row_values[:local_k]
            positive_selected = selected[selected > eps]
            if positive_selected.size:
                outgoing_scale[i] = float(np.median(positive_selected))
            else:
                outgoing_scale[i] = max(
                    float(np.median(selected)),
                    global_scale,
                )

        column_values = d[valid[:, i], i]
        if column_values.size:
            column_values = np.sort(column_values)
            selected = column_values[:local_k]
            positive_selected = selected[selected > eps]
            if positive_selected.size:
                incoming_scale[i] = float(np.median(positive_selected))
            else:
                incoming_scale[i] = max(
                    float(np.median(selected)),
                    global_scale,
                )

    outgoing_scale = np.maximum(outgoing_scale, eps)
    incoming_scale = np.maximum(incoming_scale, eps)

    pair_scale = (
        0.58 * np.sqrt(
            outgoing_scale[:, None] * incoming_scale[None, :]
        )
        + 0.42 * global_scale
    )
    pair_scale = np.maximum(pair_scale, eps)

    relative = np.divide(
        d,
        pair_scale,
        out=np.full_like(d, np.inf, dtype=float),
        where=valid,
    )

    distance_quality = np.exp(
        -np.clip(relative, 0.0, 40.0)
    )
    distance_quality[~valid] = 0.0

    row_order = np.argsort(work, axis=1, kind="stable")
    row_rank = np.full((n, n), np.inf, dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)[None, :]

    col_order = np.argsort(work, axis=0, kind="stable")
    col_rank = np.full((n, n), np.inf, dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)[:, None]

    rank_temperature = max(1.0, 0.85 * np.sqrt(float(n)))
    outgoing_rank = np.exp(
        -row_rank / rank_temperature
    )
    incoming_rank = np.exp(
        -col_rank / rank_temperature
    )
    outgoing_rank[~valid] = 0.0
    incoming_rank[~valid] = 0.0

    rank_quality = np.sqrt(
        outgoing_rank * incoming_rank
    )

    reciprocal = valid & valid.T
    mutual_near = (
        reciprocal
        & (row_rank < max(1, int(np.ceil(np.log2(n + 1.0)))))
        & (col_rank < max(1, int(np.ceil(np.log2(n + 1.0)))))
    )

    base_quality = distance_quality * (
        0.62 + 0.38 * rank_quality
    )
    base_quality *= 1.0 + 0.16 * mutual_near.astype(float)

    # Weak bridge retention: keep relatively long edges that are
    # nevertheless locally competitive at both endpoints.
    bridge_signal = np.sqrt(
        np.maximum(outgoing_rank, 0.0)
        * np.maximum(incoming_rank, 0.0)
    )
    bridge_signal = np.clip(bridge_signal, 0.0, 1.0)
    bridge_retention = 1.0 + 0.10 * bridge_signal * np.clip(
        relative - 0.8,
        0.0,
        1.5,
    ) / 1.5

    base_quality *= bridge_retention
    base_quality[~valid] = 0.0
    base_quality = np.nan_to_num(
        base_quality,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    candidate_k = max(
        1,
        min(degree, int(np.ceil(1.75 * np.sqrt(float(n))))),
    )
    candidate_temperature = max(
        0.04,
        0.18 / np.sqrt(float(n)),
    )

    def candidate_potential(values: np.ndarray) -> np.ndarray:
        potentials = np.zeros(n, dtype=float)

        for node in range(n):
            candidates = values[node]
            candidates = candidates[candidates > 0.0]

            if candidates.size == 0:
                continue

            candidates = np.sort(candidates)[::-1]
            candidates = candidates[:candidate_k]

            peak = float(candidates[0])
            weights = np.exp(
                np.clip(
                    (candidates - peak) / candidate_temperature,
                    -50.0,
                    0.0,
                )
            )
            weights /= max(float(weights.sum()), eps)

            weighted_quality = float(
                np.sum(weights * candidates)
            )
            support = min(
                candidates.size / float(candidate_k),
                1.0,
            )

            entropy = -float(
                np.sum(
                    weights * np.log(
                        np.maximum(weights, eps)
                    )
                )
            )
            entropy /= max(
                np.log(float(candidate_k) + 1.0),
                1.0,
            )

            potentials[node] = weighted_quality * support * (
                0.68 + 0.32 * entropy
            )

        return np.clip(potentials, 0.0, 1.0)

    outgoing_potential = candidate_potential(base_quality)
    incoming_potential = candidate_potential(base_quality.T)

    outgoing_projection = np.zeros(n, dtype=float)
    incoming_projection = np.zeros(n, dtype=float)

    for node in range(n):
        row = base_quality[node]
        row_count = int(np.count_nonzero(row))
        if row_count:
            outgoing_projection[node] = float(
                np.sum(row * outgoing_potential)
                / row_count
            )

        column = base_quality[:, node]
        column_count = int(np.count_nonzero(column))
        if column_count:
            incoming_projection[node] = float(
                np.sum(column * incoming_potential)
                / column_count
            )

    outgoing_future = np.clip(
        0.65 * outgoing_potential
        + 0.35 * outgoing_projection,
        0.0,
        1.0,
    )
    incoming_future = np.clip(
        0.65 * incoming_potential
        + 0.35 * incoming_projection,
        0.0,
        1.0,
    )

    continuation = np.sqrt(
        incoming_future[:, None]
        * outgoing_future[None, :]
    )
    continuation = np.clip(continuation, 0.0, 1.0)

    # Scarcity bonuses protect nodes with a pronounced local distance gap.
    sorted_rows = np.sort(work, axis=1)
    second_out = sorted_rows[:, min(1, degree - 1)]
    kth_out = sorted_rows[:, local_k - 1]

    sorted_columns = np.sort(work, axis=0)
    second_in = sorted_columns[min(1, degree - 1), :]
    kth_in = sorted_columns[local_k - 1, :]

    second_out = np.where(
        np.isfinite(second_out),
        second_out,
        global_scale,
    )
    kth_out = np.where(
        np.isfinite(kth_out),
        kth_out,
        global_scale,
    )
    second_in = np.where(
        np.isfinite(second_in),
        second_in,
        global_scale,
    )
    kth_in = np.where(
        np.isfinite(kth_in),
        kth_in,
        global_scale,
    )

    outgoing_scarcity = np.clip(
        (kth_out - outgoing_scale)
        / np.maximum(kth_out, eps),
        0.0,
        1.0,
    )
    incoming_scarcity = np.clip(
        (kth_in - incoming_scale)
        / np.maximum(kth_in, eps),
        0.0,
        1.0,
    )

    outgoing_regret = np.clip(
        (second_out - outgoing_scale)
        / np.maximum(second_out, eps),
        0.0,
        1.0,
    )
    incoming_regret = np.clip(
        (second_in - incoming_scale)
        / np.maximum(second_in, eps),
        0.0,
        1.0,
    )

    score = base_quality.copy()
    score *= 0.82 + 0.32 * continuation
    score *= 1.0 + 0.11 * (
        outgoing_scarcity[:, None]
        + incoming_scarcity[None, :]
    )
    score *= 1.0 + 0.07 * (
        outgoing_regret[:, None]
        + incoming_regret[None, :]
    )

    # Retain global scale discrimination while avoiding a hard cutoff.
    score *= np.exp(
        -0.24 * np.maximum(relative - 1.0, 0.0)
    )
    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    if not np.any(score > 0.0):
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(count, 1.0)
        return result

    row_mass = score.sum(axis=1)
    col_mass = score.sum(axis=0)

    positive_rows = row_mass[row_mass > 0.0]
    positive_cols = col_mass[col_mass > 0.0]

    reference_row = (
        float(np.median(positive_rows))
        if positive_rows.size
        else 1.0
    )
    reference_col = (
        float(np.median(positive_cols))
        if positive_cols.size
        else 1.0
    )

    row_need = np.clip(
        np.sqrt(reference_row / np.maximum(row_mass, eps)),
        0.72,
        1.55,
    )
    col_need = np.clip(
        np.sqrt(reference_col / np.maximum(col_mass, eps)),
        0.72,
        1.55,
    )

    coverage = np.sqrt(
        row_need[:, None] * col_need[None, :]
    )
    coverage /= max(
        float(np.mean(coverage[valid])),
        eps,
    )

    calibrated = score * (
        0.88 + 0.12 * np.clip(coverage, 0.60, 1.60)
    )
    calibrated[~valid] = 0.0

    # A tiny adaptive exploration component protects valid edges incident
    # to underrepresented nodes without replacing the learned ranking.
    exploration = np.sqrt(
        row_need[:, None] * col_need[None, :]
    )
    exploration = np.where(valid, exploration, 0.0)
    exploration_total = float(exploration.sum())

    calibrated_total = float(calibrated.sum())
    if calibrated_total <= 0.0 or not np.isfinite(calibrated_total):
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(count, 1.0)
        return result

    calibrated_share = 0.985 * calibrated
    if exploration_total > 0.0 and np.isfinite(exploration_total):
        calibrated_share += (
            0.015
            * calibrated_total
            * exploration
            / exploration_total
        )

    calibrated_share[~valid] = 0.0
    total = float(calibrated_share.sum())

    if not np.isfinite(total) or total <= 0.0:
        count = float(np.count_nonzero(valid))
        result[valid] = 1.0 / max(count, 1.0)
        return result

    return calibrated_share / total
