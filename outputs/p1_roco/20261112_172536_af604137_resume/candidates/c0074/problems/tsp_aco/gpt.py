import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=float)

    valid = np.isfinite(distances) & (distances > 0)
    valid &= ~np.eye(n, dtype=bool)

    if not np.any(valid):
        return np.zeros((n, n), dtype=float)

    positive = distances[valid]
    global_scale = float(np.median(positive))
    eps = np.finfo(float).eps

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        outgoing = np.where(valid, distances, np.nan)
        incoming = np.where(valid, distances, np.nan).T

        out_scale = np.nanmedian(outgoing, axis=1)
        in_scale = np.nanmedian(incoming, axis=1)

        out_scale = np.where(
            np.isfinite(out_scale) & (out_scale > 0),
            out_scale,
            global_scale,
        )
        in_scale = np.where(
            np.isfinite(in_scale) & (in_scale > 0),
            in_scale,
            global_scale,
        )

        relative_out = distances / np.maximum(out_scale[:, None], eps)
        relative_in = distances / np.maximum(in_scale[None, :], eps)

        relative_out = np.where(valid, np.maximum(relative_out, eps), np.inf)
        relative_in = np.where(valid, np.maximum(relative_in, eps), np.inf)

        base = 1.0 / np.sqrt(relative_out * relative_in)

        out_best = np.where(valid, distances, np.inf).min(axis=1)
        in_best = np.where(valid, distances, np.inf).min(axis=0)

        out_regret = np.where(
            valid,
            out_best[:, None] / np.maximum(distances, eps),
            0.0,
        )
        in_regret = np.where(
            valid,
            in_best[None, :] / np.maximum(distances, eps),
            0.0,
        )

        regret_factor = 0.85 + 0.15 * np.sqrt(
            np.clip(out_regret, 0.0, 1.0)
            * np.clip(in_regret, 0.0, 1.0)
        )

        out_affinity = np.exp(
            -0.5 * np.maximum(relative_out - 1.0, 0.0)
        )
        in_affinity = np.exp(
            -0.5 * np.maximum(relative_in - 1.0, 0.0)
        )
        affinity_factor = 0.9 + 0.1 * np.sqrt(
            out_affinity * in_affinity
        )

        score = base * regret_factor * affinity_factor
        score = np.where(valid, score, 0.0)

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0, row_max, 1.0)
    score = score / row_max

    np.fill_diagonal(score, 0.0)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    maximum = float(np.max(score))
    if maximum > 0:
        score /= maximum

    return score
