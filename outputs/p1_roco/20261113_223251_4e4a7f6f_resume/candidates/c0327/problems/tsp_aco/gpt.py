import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return deterministic tour-support priors for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2-D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    diagonal = np.eye(n, dtype=bool)
    finite = np.isfinite(raw)
    valid = (finite | finite.T) & ~diagonal

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    cleaned = np.zeros((n, n), dtype=float)

    both = finite & finite.T
    forward_only = finite & ~finite.T
    backward_only = ~finite & finite.T

    cleaned[both] = 0.5 * (raw[both] + raw.T[both])
    cleaned[forward_only] = raw[forward_only]
    cleaned[backward_only] = raw.T[backward_only]

    cleaned = np.maximum(cleaned, 0.0)
    cleaned = 0.5 * (cleaned + cleaned.T)
    cleaned[~valid] = 0.0
    np.fill_diagonal(cleaned, 0.0)

    values = cleaned[valid]
    positive = values[values > 0.0]

    if positive.size:
        q25, q50, q75 = np.quantile(positive, [0.25, 0.50, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(q50)
    else:
        scale = 1.0

    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    eps = np.finfo(float).eps
    safe_scale = max(scale, eps)

    maximum = float(np.max(values)) if values.size else safe_scale
    if not np.isfinite(maximum):
        maximum = safe_scale

    replacement = maximum + 20.0 * safe_scale

    distances = np.full((n, n), replacement, dtype=float)
    distances[valid] = cleaned[valid]
    np.fill_diagonal(distances, replacement)

    ordered = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=float)
    ranks[np.arange(n)[:, None], ordered] = np.arange(n, dtype=float)

    ranks[~valid] = float(n)
    rank_scale = max(1.0, 0.72 * np.sqrt(float(n)))

    rank_affinity = np.exp(-ranks / rank_scale)
    rank_affinity[~valid] = 0.0

    mutual_rank = np.sqrt(
        np.maximum(rank_affinity, 0.0)
        * np.maximum(rank_affinity.T, 0.0)
    )
    mutual_rank[~valid] = 0.0

    finite_valid = distances[valid]
    q50 = float(np.median(finite_valid)) if finite_valid.size else safe_scale
    q75 = (
        float(np.quantile(finite_valid, 0.75))
        if finite_valid.size
        else safe_scale
    )
    distance_temperature = max(0.35 * max(q50, safe_scale), eps)

    distance_affinity = np.zeros((n, n), dtype=float)
    distance_affinity[valid] = np.exp(
        -np.minimum(
            distances[valid] / distance_temperature,
            60.0,
        )
    )

    affinity_max = float(np.max(distance_affinity))
    if affinity_max > 0.0 and np.isfinite(affinity_max):
        distance_affinity /= affinity_max

    path_count = int(np.clip(6 + 2 * np.sqrt(float(n)), 8, 24))
    support = np.zeros((n, n), dtype=float)
    closure_support = np.zeros((n, n), dtype=float)

    for start in range(n):
        for variant in range(path_count):
            visited = np.zeros(n, dtype=bool)
            visited[start] = True
            path = [start]
            current = start

            for step in range(n - 1):
                candidates = np.flatnonzero(~visited)
                if candidates.size == 0:
                    break

                candidate_distances = distances[current, candidates]
                candidate_ranks = ranks[current, candidates]

                phase = (
                    (variant + 1) * 0.37
                    + (step + 1) * 0.19
                    + (candidates + 1) * 0.013
                )
                tie_bias = 1e-9 * np.sin(phase)

                adjusted = (
                    candidate_distances
                    + 0.015 * safe_scale * tie_bias
                )

                order = np.lexsort(
                    (
                        candidates,
                        candidate_ranks,
                        adjusted,
                    )
                )

                chosen = int(candidates[order[0]])
                previous = current
                current = chosen
                visited[current] = True
                path.append(current)

                if valid[previous, current]:
                    support[previous, current] += 1.0
                    support[current, previous] += 1.0

            if len(path) == n:
                first = path[0]
                last = path[-1]
                if valid[last, first]:
                    closure_support[last, first] += 1.0
                    closure_support[first, last] += 1.0

    support_scale = float(np.max(support))
    if support_scale > 0.0 and np.isfinite(support_scale):
        support /= support_scale

    closure_scale = float(np.max(closure_support))
    if closure_scale > 0.0 and np.isfinite(closure_scale):
        closure_support /= closure_scale

    local_support = support / max(float(path_count), 1.0)
    local_support = np.minimum(local_support, 1.0)

    closure_support = np.minimum(closure_support, 1.0)

    robust_affinity = np.zeros((n, n), dtype=float)
    robust_affinity[valid] = np.exp(
        -np.minimum(
            distances[valid] / max(q75 + 0.25 * safe_scale, eps),
            60.0,
        )
    )

    robust_max = float(np.max(robust_affinity))
    if robust_max > 0.0 and np.isfinite(robust_max):
        robust_affinity /= robust_max

    score = (
        0.48 * closure_support
        + 0.24 * local_support
        + 0.17 * mutual_rank
        + 0.11 * robust_affinity
    )

    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    score = np.maximum(score, 0.0)

    maximum = float(np.max(score))
    if not np.isfinite(maximum) or maximum <= 0.0:
        result = valid.astype(float)
        np.fill_diagonal(result, 0.0)
        return result

    score /= maximum

    sharpened = np.power(np.clip(score, 0.0, 1.0), 1.18)
    sharpened[~valid] = 0.0
    sharpened = 0.5 * (sharpened + sharpened.T)
    np.fill_diagonal(sharpened, 0.0)

    maximum = float(np.max(sharpened))
    if maximum > 0.0 and np.isfinite(maximum):
        sharpened /= maximum

    return np.nan_to_num(
        sharpened,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(float, copy=False)
