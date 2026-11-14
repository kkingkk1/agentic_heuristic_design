import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return globally diverse, long-term-aware edge desirability indicators."""
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

    # Estimate the quality of a node's future choices from its two best exits.
    # This rewards destinations with several viable continuations rather than
    # destinations having only one exceptionally short outgoing edge.
    future_option = np.zeros(n, dtype=float)
    regret = np.zeros(n, dtype=float)

    for j in range(n):
        outgoing = distances[j, valid[j]]
        if outgoing.size:
            normalized = np.sort(outgoing / max(row_scale[j], eps))
            future_option[j] = 1.0 / max(
                0.5 * (normalized[0] + normalized[min(1, normalized.size - 1)]),
                1e-12,
            )
            if normalized.size >= 2:
                regret[j] = max(normalized[1] - normalized[0], 0.0)

    nonzero_future = future_option[future_option > 0]
    if nonzero_future.size:
        future_scale = max(float(np.median(nonzero_future)), eps)
        future_factor = np.sqrt(
            np.clip(future_option / future_scale, 0.25, 4.0)
        )
    else:
        future_factor = np.ones(n, dtype=float)

    # High regret means that a node depends on one unusually good exit.
    # A mild attenuation prevents the heuristic from collapsing onto such
    # brittle transitions.
    positive_regret = regret[regret > 0]
    regret_scale = (
        max(float(np.median(positive_regret)), eps)
        if positive_regret.size
        else 1.0
    )
    regret_factor = 1.0 / (
        1.0 + 0.18 * np.clip(regret / regret_scale, 0.0, 6.0)
    )

    # Correct incoming affinity: each target j is evaluated from distances[:, j].
    # The resulting soft congestion measure promotes global target diversity.
    incoming_affinity = np.zeros((n, n), dtype=float)
    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size:
            scale = np.sqrt(row_scale[i] * col_scale[js])
            normalized = distances[i, js] / np.maximum(scale, eps)
            incoming_affinity[i, js] = np.exp(
                -np.clip(normalized, 0.0, 50.0)
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
        0.65 + np.clip(target_density / density_scale, 0.0, 8.0)
    )

    for i in range(n):
        js = np.flatnonzero(valid[i])
        if js.size == 0:
            continue

        d = distances[i, js]
        local_scale = np.sqrt(row_scale[i] * col_scale[js])

        relative_distance = d / np.maximum(local_scale, eps)
        global_distance = d / global_scale

        # Two complementary proximity views: local scale invariance and
        # globally meaningful smooth decay.
        local_affinity = np.exp(-np.clip(relative_distance, 0.0, 50.0))
        global_affinity = 1.0 / (1.0 + np.maximum(global_distance, 0.0))

        proximity = 0.68 * local_affinity + 0.32 * global_affinity

        # Mild source-side density correction preserves multiple candidates
        # while preventing a flat row from becoming numerically dominant.
        source_affinity = incoming_affinity[i, js]
        source_mean = max(float(np.mean(source_affinity)), eps)
        source_diversity = 0.85 + 0.15 * np.clip(
            source_affinity / source_mean, 0.25, 2.0
        )

        edge_score = (
            proximity
            * source_diversity
            * density_factor[js]
            * future_factor[js]
            * regret_factor[js]
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
