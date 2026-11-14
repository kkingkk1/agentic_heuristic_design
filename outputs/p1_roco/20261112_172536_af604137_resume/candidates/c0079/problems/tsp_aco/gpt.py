import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return direction-aware edge desirability scores for TSP-like graphs."""
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

    scale_matrix = np.sqrt(row_scale[:, None] * col_scale[None, :])
    normalized = distances / np.maximum(scale_matrix, eps)

    affinity = np.zeros((n, n), dtype=float)
    affinity[valid] = np.exp(-np.clip(normalized[valid], 0.0, 40.0))

    # Source-normalized probabilities estimate global target competition.
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

    load_ratio = np.clip(target_load / load_scale, 0.4, 2.5)
    load_log = np.clip(np.log(load_ratio), -1.0, 1.0)

    # Adaptive but weak diversity correction.
    valid_counts = np.sum(valid, axis=1)
    ambiguity = np.mean(valid_counts > 1) if n else 0.0
    diversity_exponent = 0.10 + 0.06 * float(ambiguity)

    diversity_factor = np.ones(n, dtype=float)
    occupied = target_load > 0.0
    diversity_factor[occupied] = np.exp(
        -diversity_exponent * load_log[occupied]
    )
    diversity_factor = np.clip(diversity_factor, 0.78, 1.28)

    # Estimate destination continuation quality and optionality.
    best_exit = np.full(n, 0.75, dtype=float)
    second_exit = np.full(n, 0.75, dtype=float)
    optionality = np.full(n, 0.72, dtype=float)

    for j in range(n):
        choices = affinity[j, valid[j]]
        if choices.size == 0:
            continue

        ordered = np.sort(choices)[::-1]
        best_exit[j] = float(ordered[0])

        if ordered.size >= 2:
            second_exit[j] = float(ordered[1])
            probability = ordered / max(float(np.sum(ordered)), eps)
            entropy = -float(
                np.sum(probability * np.log(np.maximum(probability, eps)))
            )
            entropy_norm = entropy / max(np.log(float(ordered.size)), eps)
            optionality[j] = 0.72 + 0.28 * np.clip(entropy_norm, 0.0, 1.0)
        else:
            second_exit[j] = 0.90 * best_exit[j]
            optionality[j] = 0.72

    # Geometric blending rewards two-step continuation without overvaluing
    # a single exceptionally good exit.
    continuation = np.sqrt(
        np.maximum(best_exit * second_exit, eps)
    )

    continuation_scale = max(
        float(np.median(continuation[continuation > 0.0])),
        eps,
    )
    continuation_ratio = np.clip(
        continuation / continuation_scale,
        0.45,
        2.2,
    )

    continuation_factor = np.exp(
        0.14 * np.clip(np.log(continuation_ratio), -0.9, 0.9)
    )

    optionality_scale = max(
        float(np.median(optionality[optionality > 0.0])),
        eps,
    )
    optionality_ratio = np.clip(
        optionality / optionality_scale,
        0.75,
        1.35,
    )
    optionality_factor = np.exp(
        0.10 * np.clip(np.log(optionality_ratio), -0.5, 0.5)
    )

    # A weak bridge term favors targets that are both reachable and
    # capable of supporting good future exits.
    incoming_quality = np.sum(row_prob * affinity, axis=0)
    incoming_scale = max(
        float(np.median(incoming_quality[incoming_quality > 0.0])),
        eps,
    )
    bridge_ratio = np.clip(incoming_quality / incoming_scale, 0.55, 1.8)
    bridge_factor = np.exp(
        0.08 * np.clip(np.log(bridge_ratio), -0.7, 0.7)
    )

    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size == 0:
            continue

        d = distances[i, js]
        local_scale = np.sqrt(row_scale[i] * col_scale[js])

        local_affinity = np.exp(
            -np.clip(d / np.maximum(local_scale, eps), 0.0, 40.0)
        )
        global_affinity = 1.0 / (
            1.0 + np.maximum(d / global_scale, 0.0)
        )

        # Proximity remains the dominant signal.
        proximity = 0.76 * local_affinity + 0.24 * global_affinity

        row_mean = max(float(np.mean(affinity[i, js])), eps)
        local_balance = np.clip(
            affinity[i, js] / row_mean,
            0.45,
            1.8,
        )
        source_factor = 0.94 + 0.06 * local_balance

        edge_score = (
            proximity
            * source_factor
            * diversity_factor[js]
            * continuation_factor[js]
            * optionality_factor[js]
            * bridge_factor[js]
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
