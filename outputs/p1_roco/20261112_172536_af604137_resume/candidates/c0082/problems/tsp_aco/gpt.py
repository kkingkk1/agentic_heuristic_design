import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    positive = distances[valid]
    global_scale = max(float(np.median(positive)), eps)

    row_scale = np.full(n, global_scale, dtype=float)
    col_scale = np.full(n, global_scale, dtype=float)

    for i in range(n):
        values = distances[i, valid[i]]
        if values.size:
            row_scale[i] = max(float(np.median(values)), eps)

    for j in range(n):
        values = distances[valid[:, j], j]
        if values.size:
            col_scale[j] = max(float(np.median(values)), eps)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        row_relative = distances / np.maximum(row_scale[:, None], eps)
        col_relative = distances / np.maximum(col_scale[None, :], eps)

        row_relative = np.where(valid, np.maximum(row_relative, eps), np.inf)
        col_relative = np.where(valid, np.maximum(col_relative, eps), np.inf)

        # Exploitation: favor edges that are locally short from both endpoints.
        base = 1.0 / np.sqrt(row_relative * col_relative)

        global_relative = np.where(
            valid,
            distances / global_scale,
            np.inf,
        )
        local_proximity = np.exp(
            -0.72 * np.clip(np.sqrt(row_relative * col_relative), 0.0, 50.0)
        )
        global_proximity = 1.0 / (
            1.0 + np.maximum(global_relative, 0.0)
        )
        proximity = 0.72 * local_proximity + 0.28 * global_proximity

        # Weak endpoint-regret correction preserves the distance ranking.
        row_best = np.min(
            np.where(valid, distances, np.inf),
            axis=1,
        )
        col_best = np.min(
            np.where(valid, distances, np.inf),
            axis=0,
        )

        row_regret = np.where(
            valid,
            row_best[:, None] / np.maximum(distances, eps),
            0.0,
        )
        col_regret = np.where(
            valid,
            col_best[None, :] / np.maximum(distances, eps),
            0.0,
        )
        regret_factor = 0.90 + 0.10 * np.sqrt(
            np.clip(row_regret, 0.0, 1.0)
            * np.clip(col_regret, 0.0, 1.0)
        )

        # Exploration: correctly compute incoming affinity by target column.
        incoming_affinity = np.where(
            valid,
            np.exp(-np.clip(np.sqrt(row_relative * col_relative), 0.0, 50.0)),
            0.0,
        )
        target_mass = np.sum(incoming_affinity, axis=0)
        target_count = np.sum(valid, axis=0).astype(float)
        target_density = target_mass / np.maximum(target_count, 1.0)

        positive_density = target_density[target_density > 0]
        density_scale = (
            max(float(np.median(positive_density)), eps)
            if positive_density.size
            else 1.0
        )
        density_factor = 1.0 / np.sqrt(
            0.85 + np.clip(target_density / density_scale, 0.0, 6.0)
        )
        density_factor = np.clip(density_factor, 0.88, 1.08)

        # Favor destinations with more than one reasonably good continuation.
        future_option = np.ones(n, dtype=float)
        for j in range(n):
            outgoing = distances[j, valid[j]]
            if outgoing.size:
                normalized = np.sort(outgoing / row_scale[j])
                second = normalized[min(1, normalized.size - 1)]
                future_option[j] = 1.0 / max(
                    0.5 * (normalized[0] + second),
                    1e-12,
                )

        positive_future = future_option[future_option > 0]
        future_scale = (
            max(float(np.median(positive_future)), eps)
            if positive_future.size
            else 1.0
        )
        future_factor = np.sqrt(
            np.clip(future_option / future_scale, 0.7, 1.4)
        )
        future_factor = np.clip(future_factor, 0.92, 1.08)

        score = (
            base
            * proximity
            * regret_factor
            * density_factor[None, :]
            * future_factor[None, :]
        )
        score = np.where(valid, score, 0.0)
        score = np.nan_to_num(
            score,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    row_max = np.max(score, axis=1, keepdims=True)
    score = score / np.where(row_max > 0.0, row_max, 1.0)

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
