import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return MST-aware edge-priority indicators for stochastic TSP sampling."""
    raw = np.asarray(distance_matrix, dtype=float)

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = raw.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    finite = np.isfinite(raw)
    positive = raw[finite & (raw > 0.0)]

    if positive.size:
        fallback = float(np.percentile(positive, 90.0))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = float(np.max(positive))
    else:
        fallback = 1.0

    if not np.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0

    d = np.where(finite, raw, fallback)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)

    off_diagonal = ~np.eye(n, dtype=bool)
    positive_d = d[off_diagonal & (d > 0.0)]

    if positive_d.size:
        scale = float(np.median(positive_d))
    else:
        scale = fallback

    eps = np.finfo(float).eps
    scale = max(scale, eps)

    work = d.copy()
    np.fill_diagonal(work, np.inf)

    # Dense Prim algorithm: O(n^2) time and O(n^2) storage.
    parent = np.full(n, -1, dtype=int)
    best = np.full(n, np.inf, dtype=float)
    selected = np.zeros(n, dtype=bool)
    best[0] = 0.0

    tree = np.zeros((n, n), dtype=bool)
    degree = np.zeros(n, dtype=float)

    for _ in range(n):
        masked_best = np.where(selected, np.inf, best)
        current = int(np.argmin(masked_best))

        if selected[current] or not np.isfinite(best[current]):
            remaining = np.flatnonzero(~selected)
            if remaining.size == 0:
                break
            current = int(remaining[0])

        selected[current] = True

        if parent[current] >= 0:
            p = parent[current]
            tree[current, p] = True
            tree[p, current] = True
            degree[current] += 1.0
            degree[p] += 1.0

        improved = (~selected) & (work[current] < best)
        best[improved] = work[current, improved]
        parent[improved] = current

    # Compute the maximum edge on every MST path by tree traversals.
    path_bottleneck = np.zeros((n, n), dtype=float)

    for source in range(n):
        stack = [(source, -1, 0.0)]

        while stack:
            node, previous, current_max = stack.pop()
            path_bottleneck[source, node] = current_max

            neighbors = np.flatnonzero(tree[node])
            for neighbor in neighbors:
                if neighbor != previous:
                    next_max = max(current_max, d[node, neighbor])
                    stack.append((int(neighbor), node, next_max))

    safe_bottleneck = np.maximum(path_bottleneck, scale * 1e-12)

    # Direct-edge attraction with a bounded rational kernel.
    direct_attraction = 1.0 / (1.0 + d / scale)

    # Reward edges that improve upon their MST-path bottleneck alternative.
    shortcut_ratio = safe_bottleneck / (
        d + safe_bottleneck + scale * 1e-12
    )
    shortcut_factor = 0.78 + 0.22 * np.clip(shortcut_ratio, 0.0, 1.0)

    # Preserve the sparse backbone without making it dominant.
    tree_factor = np.where(tree, 1.24, 1.0)

    # Mildly favor edges incident to low-degree tree endpoints.
    degree_scale = max(float(np.sqrt(max(n - 1, 1))), 1.0)
    endpoint_balance = np.sqrt(
        (1.0 + 0.16 * np.exp(-degree / degree_scale))[:, None]
        * (1.0 + 0.16 * np.exp(-degree / degree_scale))[None, :]
    )

    score = (
        direct_attraction
        * shortcut_factor
        * tree_factor
        * endpoint_balance
    )

    score = 0.5 * (score + score.T)
    score[~off_diagonal] = 0.0

    score = np.nan_to_num(
        score,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(score))
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    np.fill_diagonal(score, 0.0)
    return score.astype(float, copy=False)
