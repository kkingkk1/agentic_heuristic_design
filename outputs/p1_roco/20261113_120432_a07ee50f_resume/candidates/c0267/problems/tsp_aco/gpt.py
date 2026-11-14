import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric normalized edge-priority matrix for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")

    n = raw.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n < 2:
        return result

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw) & (raw >= 0.0) & ~diagonal

    # Construct a symmetric effective distance matrix.
    forward_only = finite & ~finite.T
    backward_only = finite.T & ~finite
    mutual = finite & finite.T

    distance = np.full((n, n), np.inf, dtype=float)
    distance[mutual] = 0.5 * (raw[mutual] + raw.T[mutual])
    distance[forward_only] = raw[forward_only]
    distance[backward_only] = raw.T[backward_only]
    np.fill_diagonal(distance, np.inf)

    valid = np.isfinite(distance)
    if not np.any(valid):
        return result

    eps = np.finfo(float).eps
    values = distance[valid]

    positive = values[values > 0.0]
    if positive.size:
        reference = float(np.median(positive))
    else:
        reference = max(float(np.median(values)), 1.0)

    reference = max(reference, eps)

    # Stable global percentile coordinates for every valid edge.
    sorted_values = np.sort(values)
    percentile = np.searchsorted(
        sorted_values,
        distance,
        side="right",
    ).astype(float) / float(sorted_values.size)

    percentile[~valid] = 1.0

    # Candidate sets are defined by a fixed global quantile, not by
    # mutual-neighbor or local-scale interactions.
    candidate_limit = min(6, n - 1)
    order = np.argsort(distance, axis=1, kind="stable")

    candidate_mask = np.zeros((n, n), dtype=bool)
    candidate_rank = np.full((n, n), float(candidate_limit + 1))

    for i in range(n):
        neighbors = order[i][valid[i, order[i]]]
        chosen = neighbors[:candidate_limit]
        if chosen.size:
            candidate_mask[i, chosen] = True
            candidate_rank[i, chosen] = np.arange(
                1,
                chosen.size + 1,
                dtype=float,
            )

    candidate_count = candidate_mask.sum(axis=1).astype(float)

    # For each endpoint, compute the two best alternatives.
    first_alt = np.full(n, reference, dtype=float)
    second_alt = np.full(n, reference, dtype=float)

    for i in range(n):
        neighbors = order[i][valid[i, order[i]]]
        if neighbors.size:
            row_values = distance[i, neighbors]
            first_alt[i] = row_values[0]
            second_alt[i] = row_values[1] if row_values.size > 1 else row_values[0]

    # A large regret means that omitting this edge leaves an endpoint
    # with substantially worse alternatives.
    endpoint_regret = np.maximum(
        second_alt - first_alt,
        0.0,
    ) / np.maximum(reference + first_alt, eps)

    edge_cost = distance / reference

    # Bounded short-edge preference based on reciprocal cost.
    short_edge = 1.0 / (1.0 + np.clip(edge_cost, 0.0, 1.0e6))

    # Edge-specific regret: compare this edge against the best alternative
    # available at either endpoint.
    endpoint_best = np.minimum(
        first_alt[:, None],
        first_alt[None, :],
    )
    endpoint_second = np.maximum(
        second_alt[:, None],
        second_alt[None, :],
    )

    edge_regret = np.maximum(endpoint_second - distance, 0.0)
    edge_regret /= np.maximum(
        endpoint_second + endpoint_best + reference,
        eps,
    )
    edge_regret = np.clip(edge_regret, 0.0, 1.0)

    # Penalize edges that are deep in both endpoint candidate lists.
    rank_fraction = np.sqrt(
        np.clip(
            candidate_rank / np.maximum(candidate_count[:, None], 1.0),
            0.0,
            2.0,
        )
        * np.clip(
            candidate_rank.T / np.maximum(candidate_count[None, :], 1.0),
            0.0,
            2.0,
        )
    )
    rank_quality = 1.0 / (1.0 + rank_fraction)

    # Mild degree balancing: edges incident to very dense candidate regions
    # receive less concentration pressure, while sparse endpoints are preserved.
    degree_scale = np.sqrt(
        np.maximum(candidate_count[:, None], 1.0)
        * np.maximum(candidate_count[None, :], 1.0)
    )
    degree_balance = 1.0 / np.sqrt(
        1.0 + degree_scale / max(float(candidate_limit), 1.0)
    )

    endpoint_urgency = np.sqrt(
        endpoint_regret[:, None] * endpoint_regret[None, :]
    )
    endpoint_urgency = np.clip(endpoint_urgency, 0.0, 1.0)

    # Combine independent bounded signals. Distance remains dominant.
    scores = (
        short_edge ** 1.35
        * (0.35 + 0.65 * rank_quality)
        * (0.72 + 0.28 * edge_regret)
        * (0.82 + 0.18 * endpoint_urgency)
        * (0.88 + 0.12 * degree_balance)
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    scores = np.maximum(scores, 0.0)
    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
