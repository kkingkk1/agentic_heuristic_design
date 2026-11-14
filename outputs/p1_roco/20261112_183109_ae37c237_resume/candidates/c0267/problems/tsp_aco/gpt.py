import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return symmetric edge-priority indicators for stochastic TSP sampling."""

    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n < 2:
        return scores

    finite = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(finite, False)

    valid = finite & finite.T
    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps

    with np.errstate(all="ignore"):
        d = 0.5 * (distances + distances.T)

    d[~valid] = np.inf
    np.fill_diagonal(d, np.inf)

    valid_values = d[valid]
    positive_values = valid_values[valid_values > 0.0]

    if positive_values.size:
        global_scale = float(np.median(positive_values))
    elif valid_values.size:
        global_scale = float(np.max(valid_values))
    else:
        global_scale = 1.0

    if not np.isfinite(global_scale) or global_scale <= 0.0:
        global_scale = 1.0
    global_scale = max(global_scale, eps)

    finite_max = float(np.max(valid_values))
    fill_value = max(finite_max, global_scale) * 4.0 + global_scale

    work = np.where(valid, d, fill_value)
    np.fill_diagonal(work, fill_value)

    row_scales = np.empty(n, dtype=float)
    temperatures = np.empty(n, dtype=float)

    for i in range(n):
        values = d[i, valid[i]]
        if values.size == 0:
            row_scales[i] = global_scale
            temperatures[i] = 1.0
            continue

        positive = values[values > 0.0]
        if positive.size:
            center = float(np.median(positive))
        else:
            center = float(np.median(values))

        if not np.isfinite(center) or center <= 0.0:
            center = global_scale

        row_scales[i] = max(center, eps)

        spread = float(np.median(np.abs(values - center)))
        relative_spread = spread / max(center, eps)
        temperatures[i] = np.clip(
            0.72 + 0.22 * relative_spread,
            0.68,
            1.18,
        )

    row_scales = np.maximum(row_scales, eps)

    geometric_scale = np.sqrt(
        row_scales[:, None] * row_scales[None, :]
    )

    relative_cost = np.divide(
        work,
        geometric_scale,
        out=np.full((n, n), np.inf, dtype=float),
        where=geometric_scale > 0.0,
    )

    temperature = np.sqrt(
        temperatures[:, None] * temperatures[None, :]
    )

    # Smooth cost based on logarithmic relative distance.
    smooth_cost = np.log1p(
        np.clip(relative_cost, 0.0, 1e6)
    )

    local_preference = np.exp(
        -np.divide(
            smooth_cost,
            np.maximum(temperature, 0.5),
        )
    )
    local_preference[~valid] = 0.0
    np.fill_diagonal(local_preference, 0.0)

    row_mass = local_preference.sum(axis=1, keepdims=True)
    row_probability = np.divide(
        local_preference,
        row_mass,
        out=np.zeros_like(local_preference),
        where=row_mass > 0.0,
    )

    # Agreement between the two directed endpoint preferences.
    mutual_agreement = np.sqrt(
        np.maximum(
            row_probability * row_probability.T,
            0.0,
        )
    )

    # Mildly favor edges that are competitive at both endpoints.
    endpoint_competition = np.sqrt(
        np.maximum(
            local_preference * local_preference.T,
            0.0,
        )
    )
    competition_scale = np.max(endpoint_competition)
    if not np.isfinite(competition_scale) or competition_scale <= 0.0:
        competition_scale = 1.0

    competition_term = np.clip(
        endpoint_competition / competition_scale,
        0.0,
        1.0,
    )

    # Compute the cheapest valid two-hop replacement for every edge.
    detour = np.full((n, n), np.inf, dtype=float)

    for k in range(n):
        left_valid = valid[:, k]
        right_valid = valid[k, :]
        usable = left_valid[:, None] & right_valid[None, :]

        candidate = d[:, k, None] + d[None, k, :]
        candidate[~usable] = np.inf
        candidate[:, k] = np.inf
        candidate[k, :] = np.inf

        detour = np.minimum(detour, candidate)

    positive_edge = np.maximum(work, eps)
    detour_ratio = np.divide(
        detour,
        positive_edge,
        out=np.zeros_like(detour),
        where=np.isfinite(detour) & valid,
    )

    # Essentiality rises when removing an edge causes a much longer detour.
    essentiality = np.log1p(
        np.clip(detour_ratio - 1.0, 0.0, 12.0)
    )
    essentiality = np.clip(essentiality / np.log(13.0), 0.0, 1.0)

    # Do not let an extremely long edge receive a large structural bonus.
    quality_gate = np.exp(
        -0.38 * np.maximum(relative_cost - 1.25, 0.0)
    )
    quality_gate = np.clip(quality_gate, 0.0, 1.0)

    essentiality_term = 1.0 + (
        0.24 * essentiality * quality_gate
    )

    # Penalize severe endpoint scale mismatch without requiring symmetry
    # in the original input matrix.
    scale_balance = np.exp(
        -0.20
        * np.abs(
            np.log(
                np.maximum(row_scales[:, None], eps)
                / np.maximum(row_scales[None, :], eps)
            )
        )
    )
    scale_balance = np.clip(scale_balance, 0.82, 1.0)

    # A bounded correction for edges that are locally competitive
    # but not simultaneously dominant at both endpoints.
    disagreement = 1.0 - np.clip(mutual_agreement, 0.0, 1.0)
    exploration_term = 1.0 + (
        0.10
        * disagreement
        * competition_term
        * quality_gate
    )

    scores = (
        np.exp(-0.92 * smooth_cost)
        * (0.42 + 0.58 * mutual_agreement)
        * (0.72 + 0.28 * competition_term)
        * essentiality_term
        * exploration_term
        * scale_balance
    )

    scores[~valid] = 0.0
    scores = 0.5 * (scores + scores.T)
    np.fill_diagonal(scores, 0.0)

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    maximum = float(np.max(scores))
    if np.isfinite(maximum) and maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
