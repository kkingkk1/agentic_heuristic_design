import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return directed edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    scores = np.zeros((n, n), dtype=float)

    if n <= 1:
        return scores

    valid = np.isfinite(distances) & (distances > 0)
    np.fill_diagonal(valid, False)

    if not np.any(valid):
        return scores

    eps = np.finfo(float).eps
    positive = distances[valid]
    global_scale = max(float(np.median(positive)), eps)

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

        # Main bidirectional local-scale priority.
        base = 1.0 / np.sqrt(relative_out * relative_in)

        # Mild global-scale smoothing.
        global_affinity = 1.0 / (
            1.0 + np.maximum(distances / global_scale, 0.0)
        )

        # Bounded two-sided short-edge advantage.
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

        regret_factor = 0.88 + 0.12 * np.sqrt(
            np.clip(out_regret, 0.0, 1.0)
            * np.clip(in_regret, 0.0, 1.0)
        )

        # Estimate whether each destination has more than one viable successor.
        continuation = np.ones(n, dtype=float)

        for j in range(n):
            exits = distances[j, valid[j]]
            if exits.size == 0:
                continue

            normalized = np.sort(exits / max(out_scale[j], eps))
            first = normalized[0]
            second = normalized[min(1, normalized.size - 1)]
            continuation[j] = 1.0 / max(0.5 * (first + second), eps)

        positive_continuation = continuation[continuation > 0]
        continuation_scale = max(
            float(np.median(positive_continuation)),
            eps,
        )
        continuation_factor = 0.92 + 0.08 * np.sqrt(
            np.clip(continuation / continuation_scale, 0.5, 2.0)
        )

        score = (
            base
            * (0.82 + 0.18 * global_affinity)
            * regret_factor
            * continuation_factor[None, :]
        )

        score = np.where(valid, score, 0.0)
        score = np.nan_to_num(
            score,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    row_max = np.max(score, axis=1, keepdims=True)
    row_max = np.where(row_max > 0.0, row_max, 1.0)
    scores = score / row_max

    np.fill_diagonal(scores, 0.0)

    maximum = float(np.max(scores))
    if maximum > 0.0:
        scores /= maximum

    return scores.astype(float, copy=False)
