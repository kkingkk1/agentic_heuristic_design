import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return direction-aware, globally diverse edge desirability scores."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    eps = np.finfo(float).eps
    valid = np.isfinite(distances) & (distances > 0.0)
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

    affinity = np.zeros((n, n), dtype=float)
    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size:
            scale = np.sqrt(row_scale[i] * col_scale[js])
            normalized = distances[i, js] / np.maximum(scale, eps)
            affinity[i, js] = np.exp(-np.clip(normalized, 0.0, 40.0))

    # Soft target capacity: estimate how strongly each target is preferred
    # after normalizing each source row, then apply only a mild correction.
    row_mass = np.sum(affinity, axis=1, keepdims=True)
    row_prob = np.divide(
        affinity,
        np.maximum(row_mass, eps),
        out=np.zeros_like(affinity),
        where=row_mass > 0.0,
    )
    target_load = np.sum(row_prob, axis=0)

    positive_load = target_load[target_load > 0.0]
    load_scale = (
        max(float(np.median(positive_load)), eps)
        if positive_load.size
        else 1.0
    )
    load_ratio = np.clip(target_load / load_scale, 0.25, 4.0)

    # Centered and capped correction prevents over-penalizing legitimate
    # targets in sparse or asymmetric graphs.
    diversity_factor = np.ones(n, dtype=float)
    occupied = target_load > 0.0
    diversity_factor[occupied] = np.exp(
        -0.16 * np.clip(np.log(load_ratio[occupied]), -1.5, 1.5)
    )

    # Estimate long-term continuation from the two best outgoing choices.
    # A single available exit receives a small uncertainty discount only.
    continuation = np.ones(n, dtype=float)
    for j in range(n):
        js = np.flatnonzero(valid[j])
        if js.size == 0:
            continuation[j] = 0.75
            continue

        options = np.sort(affinity[j, js])[::-1]
        best = float(options[0])

        if options.size >= 2:
            second = float(options[1])
            continuation[j] = np.sqrt(max(best * second, eps))
        else:
            continuation[j] = 0.88 * best

    positive_continuation = continuation[continuation > 0.0]
    continuation_scale = max(
        float(np.median(positive_continuation)),
        eps,
    )
    continuation_ratio = np.clip(
        continuation / continuation_scale,
        0.35,
        2.85,
    )
    future_factor = np.exp(
        0.18 * np.clip(np.log(continuation_ratio), -1.2, 1.2)
    )

    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size == 0:
            continue

        d = distances[i, js]
        scale = np.sqrt(row_scale[i] * col_scale[js])

        local_affinity = np.exp(
            -np.clip(d / np.maximum(scale, eps), 0.0, 40.0)
        )
        global_affinity = 1.0 / (
            1.0 + np.maximum(d / global_scale, 0.0)
        )

        proximity = 0.72 * local_affinity + 0.28 * global_affinity

        # Preserve source-specific alternatives instead of collapsing each
        # row onto a single nearest neighbor.
        source_mean = max(float(np.mean(affinity[i, js])), eps)
        source_balance = 0.92 + 0.08 * np.clip(
            affinity[i, js] / source_mean,
            0.35,
            2.0,
        )

        edge_score = (
            proximity
            * source_balance
            * diversity_factor[js]
            * future_factor[js]
        )
        edge_score = np.nan_to_num(
            edge_score,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        maximum = float(np.max(edge_score))
        if maximum > 0.0:
            scores[i, js] = edge_score / maximum

    np.fill_diagonal(scores, 0.0)
    return scores.astype(float, copy=False)
