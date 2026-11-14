import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    finite = np.isfinite(d)
    valid = finite & (d >= 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    valid_values = d[valid]
    q25, q50, q75, q95 = np.percentile(valid_values, [25, 50, 75, 95])
    scale = max(float(q75 - q25), float(q50), 1e-12)

    replacement = float(q95 + 8.0 * scale)
    clean = np.where(valid, d, replacement).astype(float, copy=True)
    np.fill_diagonal(clean, replacement)

    eps = max(scale * 1e-9, 1e-12)

    reciprocal = 1.0 / (clean + eps)
    reciprocal[~valid] = 0.0

    row_q = np.percentile(
        np.where(valid, d, np.nan),
        50,
        axis=1,
    )
    row_q = np.nan_to_num(row_q, nan=q50, posinf=q95, neginf=q50)
    row_scale = np.maximum(row_q, eps)

    local_affinity = np.exp(-np.minimum(clean / row_scale[:, None], 30.0))
    local_affinity[~valid] = 0.0

    row_max = np.maximum(reciprocal.max(axis=1, keepdims=True), eps)
    col_max = np.maximum(reciprocal.max(axis=0, keepdims=True), eps)

    outgoing = reciprocal / row_max
    incoming = reciprocal / col_max.T
    bidirectional = np.sqrt(outgoing * incoming)

    order = np.argsort(clean, axis=1)
    rank = np.empty((n, n), dtype=float)
    rank[np.arange(n)[:, None], order] = np.arange(n, dtype=float)
    rank_score = np.exp(-rank / max(1.0, np.sqrt(float(n))))
    rank_score[~valid] = 0.0

    continuation = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in np.flatnonzero(valid[i]):
            successors = np.flatnonzero(valid[j])
            successors = successors[successors != i]
            if successors.size == 0:
                continuation[i, j] = 1.0
                continue

            best_next = np.min(clean[j, successors])
            continuation[i, j] = np.exp(
                -0.5 * (clean[i, j] + best_next) / scale
            )

    continuation[~valid] = 0.0

    score = (
        0.40 * bidirectional
        + 0.25 * rank_score
        + 0.20 * local_affinity
        + 0.15 * continuation
    )
    score[~valid] = 0.0

    maximum = float(score.max())
    if not np.isfinite(maximum) or maximum <= 0:
        score = valid.astype(float)
    else:
        score /= maximum

    floor = 0.02
    score = np.where(valid, floor + (1.0 - floor) * score, 0.0)

    seed_values = np.nan_to_num(d, nan=0.0, posinf=replacement, neginf=0.0)
    seed = int(
        (
            np.sum(seed_values) * 1_000_003
            + np.sum(seed_values * np.arange(1, n + 1)[:, None]) * 97
            + n * 104_729
        )
        % (2**63 - 1)
    )
    rng = np.random.default_rng(seed)

    samples = int(np.clip(10 * n, 80, 320))
    candidate_width = int(np.clip(2.5 * np.sqrt(n), 5, 28))
    temperature = 0.45

    tours = []
    lengths = []

    for _ in range(samples):
        start = int(rng.integers(n))
        tour = [start]
        remaining = np.ones(n, dtype=bool)
        remaining[start] = False
        current = start

        while np.any(remaining):
            candidates = np.flatnonzero(remaining)

            if candidates.size > candidate_width:
                distances = clean[current, candidates]
                nearest = np.argpartition(
                    distances,
                    candidate_width - 1,
                )[:candidate_width]
                candidates = candidates[nearest]

            logits = np.log(score[current, candidates] + 1e-12)
            logits -= clean[current, candidates] / scale
            logits /= temperature
            logits -= np.max(logits)

            probabilities = np.exp(np.clip(logits, -60.0, 0.0))
            probability_sum = probabilities.sum()

            if not np.isfinite(probability_sum) or probability_sum <= 0:
                probabilities = np.full(
                    candidates.size,
                    1.0 / candidates.size,
                    dtype=float,
                )
            else:
                probabilities /= probability_sum

            next_node = int(rng.choice(candidates, p=probabilities))
            tour.append(next_node)
            remaining[next_node] = False
            current = next_node

        tour_array = np.asarray(tour, dtype=int)
        tour_length = float(
            np.sum(d[tour_array, np.roll(tour_array, -1)])
        )

        if np.isfinite(tour_length):
            tours.append(tour_array)
            lengths.append(tour_length)

    if not tours:
        return score.astype(float, copy=False)

    lengths = np.asarray(lengths, dtype=float)
    order = np.argsort(lengths, kind="stable")
    elite_count = max(1, int(np.ceil(0.20 * len(order))))
    elite_indices = order[:elite_count]

    elite_counts = np.zeros((n, n), dtype=float)
    elite_weights = np.exp(
        -(lengths[elite_indices] - lengths[elite_indices[0]])
        / max(scale, eps)
    )
    elite_weights /= np.maximum(elite_weights.sum(), eps)

    incumbent_counts = np.zeros((n, n), dtype=float)

    for weight, index in zip(elite_weights, elite_indices):
        tour = tours[int(index)]
        edges_from = tour
        edges_to = np.roll(tour, -1)

        elite_counts[edges_from, edges_to] += weight

        if int(index) == int(elite_indices[0]):
            incumbent_counts[edges_from, edges_to] = 1.0

    elite_counts[~valid] = 0.0
    incumbent_counts[~valid] = 0.0

    if elite_counts.max() > 0:
        elite_counts /= elite_counts.max()

    if incumbent_counts.max() > 0:
        incumbent_counts /= incumbent_counts.max()

    result = 0.68 * score + 0.24 * elite_counts + 0.08 * incumbent_counts
    result[~valid] = 0.0
    np.fill_diagonal(result, 0.0)

    maximum = float(result.max())
    if np.isfinite(maximum) and maximum > 0:
        result /= maximum

    return result.astype(float, copy=False)
