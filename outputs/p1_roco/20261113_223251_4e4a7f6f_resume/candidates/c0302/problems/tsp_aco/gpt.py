import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge priors for stochastic TSP solution sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    directed_valid = (
        np.isfinite(raw)
        & (raw >= 0.0)
        & ~diagonal
    )
    valid = directed_valid | directed_valid.T

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    symmetric = np.full((n, n), np.inf, dtype=float)
    both = directed_valid & directed_valid.T
    only_forward = directed_valid & ~directed_valid.T
    only_reverse = directed_valid.T & ~directed_valid

    symmetric[both] = 0.5 * (
        raw[both] + raw.T[both]
    )
    symmetric[only_forward] = raw[only_forward]
    symmetric[only_reverse] = raw.T[only_reverse]
    np.fill_diagonal(symmetric, np.inf)

    values = symmetric[valid]
    positive = values[values > 0.0]

    if positive.size:
        q25, median, q75 = np.quantile(
            positive,
            [0.25, 0.50, 0.75],
        )
        iqr = float(q75 - q25)
        mad = float(
            np.median(np.abs(positive - median))
        )
        scale = max(
            iqr / 1.349,
            mad / 0.6745,
            0.10 * float(median),
            np.finfo(float).eps,
        )
    else:
        median = 0.0
        scale = 1.0

    finite_max = float(np.max(values))
    replacement = finite_max + 8.0 * scale
    distances = np.where(valid, symmetric, replacement)
    np.fill_diagonal(distances, replacement)

    eps = np.finfo(float).eps
    scaled = np.clip(distances / max(scale, eps), 0.0, 60.0)

    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = np.exp(
        -scaled[valid]
    )

    inverse = np.zeros((n, n), dtype=float)
    inverse[valid] = 1.0 / (
        distances[valid] + 0.05 * scale
    )
    inverse_max = float(np.max(inverse))
    if inverse_max > 0.0:
        inverse /= inverse_max

    row_mass = distance_affinity.sum(axis=1)
    col_mass = distance_affinity.sum(axis=0)

    row_preference = (
        distance_affinity
        / np.maximum(row_mass[:, None], eps)
    )
    col_preference = (
        distance_affinity
        / np.maximum(col_mass[None, :], eps)
    )

    mutual_preference = np.sqrt(
        np.maximum(
            row_preference * col_preference,
            0.0,
        )
    )
    mutual_preference[~valid] = 0.0

    row_order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )
    row_rank = np.empty((n, n), dtype=float)
    row_rank[
        np.arange(n)[:, None],
        row_order,
    ] = np.arange(n, dtype=float)

    col_order = np.argsort(
        distances,
        axis=0,
        kind="stable",
    )
    col_rank = np.empty((n, n), dtype=float)
    col_rank[
        col_order,
        np.arange(n)[None, :],
    ] = np.arange(n, dtype=float)

    rank_scale = max(1.0, 0.8 * np.sqrt(float(n)))
    row_rank_score = np.exp(-row_rank / rank_scale)
    col_rank_score = np.exp(-col_rank / rank_scale)

    mutual_rank = np.sqrt(
        row_rank_score * col_rank_score
    )
    mutual_rank[~valid] = 0.0

    sorted_rows = np.sort(distances, axis=1)
    row_best = sorted_rows[:, 0]
    row_second = sorted_rows[:, min(1, n - 1)]
    row_second = np.where(
        np.isfinite(row_second),
        row_second,
        row_best + scale,
    )
    row_gap_scale = np.maximum(
        row_second - row_best,
        0.15 * scale,
    )

    sorted_cols = np.sort(distances, axis=0)
    col_best = sorted_cols[0, :]
    col_second = sorted_cols[min(1, n - 1), :]
    col_second = np.where(
        np.isfinite(col_second),
        col_second,
        col_best + scale,
    )
    col_gap_scale = np.maximum(
        col_second - col_best,
        0.15 * scale,
    )

    row_gap = np.maximum(
        row_second[:, None] - distances,
        0.0,
    )
    col_gap = np.maximum(
        col_second[None, :] - distances,
        0.0,
    )

    row_competition = 1.0 - np.exp(
        -row_gap / row_gap_scale[:, None]
    )
    col_competition = 1.0 - np.exp(
        -col_gap / col_gap_scale[None, :]
    )

    local_competition = np.sqrt(
        np.maximum(
            row_competition * col_competition,
            0.0,
        )
    )
    local_competition[~valid] = 0.0

    bypass = np.full(
        (n, n),
        replacement,
        dtype=float,
    )
    for intermediate in range(n):
        candidate = (
            distances[:, intermediate, None]
            + distances[None, intermediate, :]
        )
        bypass = np.minimum(bypass, candidate)

    bypass[diagonal] = replacement
    detour_gain = np.maximum(
        bypass - distances,
        0.0,
    )
    irreplaceability = 1.0 - np.exp(
        -detour_gain / max(0.8 * scale, eps)
    )
    irreplaceability[~valid] = 0.0

    finite_values = values[np.isfinite(values)]
    seed_value = int(
        abs(
            float(np.sum(finite_values))
            + 17.0 * float(np.sum(finite_values ** 2))
        )
    ) % (2**32 - 1)
    rng = np.random.default_rng(seed_value)

    stability = np.zeros((n, n), dtype=float)
    trials = 6

    for _ in range(trials):
        perturbation = rng.uniform(
            0.0,
            0.035 * max(scale, eps),
            size=(n, n),
        )
        perturbed = distances + perturbation
        perturbed[~valid] = np.inf
        np.fill_diagonal(perturbed, np.inf)

        visited = np.zeros(n, dtype=bool)
        best = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)

        for start in range(n):
            if visited[start]:
                continue

            best[start] = 0.0

            while True:
                candidates = np.where(
                    ~visited,
                    best,
                    np.inf,
                )
                node = int(np.argmin(candidates))

                if not np.isfinite(candidates[node]):
                    break

                visited[node] = True
                ancestor = parent[node]

                if ancestor >= 0:
                    stability[node, ancestor] += 1.0
                    stability[ancestor, node] += 1.0

                neighbors = np.flatnonzero(
                    valid[node] & ~visited
                )
                for other in neighbors:
                    if perturbed[node, other] < best[other]:
                        best[other] = perturbed[node, other]
                        parent[other] = node

    stability /= float(trials)
    stability[~valid] = 0.0

    distance_signal = 0.70 * distance_affinity + 0.30 * inverse
    local_signal = 0.58 * mutual_preference + 0.42 * mutual_rank
    structural_signal = (
        0.62 * irreplaceability
        + 0.38 * local_competition
    )

    score = (
        0.34 * distance_signal
        + 0.24 * local_signal
        + 0.18 * structural_signal
        + 0.14 * stability
        + 0.10 * mutual_rank
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    result = score / maximum
    return np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
