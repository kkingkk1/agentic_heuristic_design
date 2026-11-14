import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic edge-selection priors for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite_nonnegative = np.isfinite(distances) & (distances >= 0.0)
    valid = finite_nonnegative.copy()
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    observed = distances[valid]
    q10, q50, q90 = np.quantile(observed, [0.10, 0.50, 0.90])

    spread = float(q90 - q10)
    if not np.isfinite(spread) or spread <= 0.0:
        spread = float(q50)
    if not np.isfinite(spread) or spread <= 0.0:
        positive = observed[observed > 0.0]
        spread = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(spread) or spread <= 0.0:
        spread = 1.0

    upper = float(q90) + 12.0 * spread
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 16.0 * spread

    clean = np.where(
        finite_nonnegative,
        np.maximum(distances, 0.0),
        upper,
    ).astype(float, copy=True)
    np.fill_diagonal(clean, upper)

    eps = np.finfo(float).eps * max(spread, 1.0)

    scaled = np.clip((clean - q10) / max(spread, eps), -40.0, 40.0)
    soft_cost = np.exp(scaled)
    soft_cost[~valid] = 0.0

    row_best = np.min(
        np.where(valid, clean, np.inf),
        axis=1,
        keepdims=True,
    )
    col_best = np.min(
        np.where(valid, clean, np.inf),
        axis=0,
        keepdims=True,
    )

    row_best = np.where(np.isfinite(row_best), row_best, upper)
    col_best = np.where(np.isfinite(col_best), col_best, upper)

    row_second = np.full((n, 1), upper, dtype=float)
    col_second = np.full((1, n), upper, dtype=float)

    for i in range(n):
        row_values = clean[i, valid[i]]
        if row_values.size:
            ordered = np.sort(row_values, kind="stable")
            row_second[i, 0] = ordered[1] if ordered.size > 1 else ordered[0]

    for j in range(n):
        col_values = clean[valid[:, j], j]
        if col_values.size:
            ordered = np.sort(col_values, kind="stable")
            col_second[0, j] = ordered[1] if ordered.size > 1 else ordered[0]

    outgoing_regret = np.maximum(
        row_second - clean,
        0.0,
    ) / max(spread, eps)

    incoming_regret = np.maximum(
        col_second - clean,
        0.0,
    ) / max(spread, eps)

    outgoing_regret = np.clip(outgoing_regret, 0.0, 12.0)
    incoming_regret = np.clip(incoming_regret, 0.0, 12.0)

    continuation = np.full((n, n), upper, dtype=float)

    for i in range(n):
        for j in range(n):
            if not valid[i, j]:
                continue

            best_two_step = upper
            for k in range(n):
                if k == i or k == j:
                    continue
                if valid[i, k] and valid[k, j]:
                    candidate = clean[i, k] + clean[k, j]
                    if candidate < best_two_step:
                        best_two_step = candidate

            continuation[i, j] = best_two_step

    bridge_advantage = np.maximum(
        continuation - clean,
        0.0,
    ) / max(spread, eps)
    bridge_advantage = np.clip(bridge_advantage, 0.0, 16.0)

    next_best = np.min(
        np.where(valid, clean, np.inf),
        axis=1,
    )
    next_best = np.where(np.isfinite(next_best), next_best, upper)

    lookahead = clean + next_best[None, :]
    lookahead_scale = max(float(np.median(observed)), spread, eps)
    lookahead_score = np.exp(
        -np.clip(lookahead / lookahead_scale, 0.0, 60.0)
    )

    row_mass = np.sum(
        np.where(valid, np.exp(-np.clip(clean / lookahead_scale, 0.0, 60.0)), 0.0),
        axis=1,
    )
    destination_access = row_mass[None, :] / max(
        float(np.median(row_mass[row_mass > 0.0]))
        if np.any(row_mass > 0.0)
        else 1.0,
        eps,
    )
    destination_access = np.clip(destination_access, 0.25, 4.0)

    regret_factor = (
        1.0
        + 0.32 * outgoing_regret
        + 0.28 * incoming_regret
    )
    bridge_factor = 1.0 + 0.24 * bridge_advantage
    access_factor = np.sqrt(destination_access)

    score = (
        np.exp(-0.72 * np.clip(soft_cost, 0.0, 40.0))
        * regret_factor
        * bridge_factor
        * access_factor
        * (0.72 + 0.28 * lookahead_score)
    )

    score[~valid] = 0.0
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
    else:
        result = score / maximum

    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
