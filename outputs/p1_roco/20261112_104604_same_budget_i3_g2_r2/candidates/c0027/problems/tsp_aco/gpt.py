import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    result = np.zeros((n, n), dtype=float)

    if n <= 1:
        return result

    finite_nonnegative = np.isfinite(distance_matrix) & (distance_matrix >= 0.0)
    off_diagonal = ~np.eye(n, dtype=bool)

    # Undirected TSP edges require both directed entries to be usable.
    valid = finite_nonnegative & finite_nonnegative.T & off_diagonal

    if not np.any(valid):
        return result

    symmetric_distance = 0.5 * (distance_matrix + distance_matrix.T)
    work = np.where(valid, symmetric_distance, np.inf)

    # Build a deterministic minimum spanning forest with Prim's algorithm.
    parent = np.full(n, -1, dtype=int)
    component = np.full(n, -1, dtype=int)
    in_tree = np.zeros(n, dtype=bool)
    tree_edges = np.zeros((n, n), dtype=bool)

    component_id = 0

    for start in range(n):
        if in_tree[start]:
            continue

        component[start] = component_id
        in_tree[start] = True

        for j in range(n):
            if valid[start, j] and not in_tree[j]:
                if (
                    parent[j] == -1
                    or work[start, j] < work[parent[j], j]
                    or (
                        work[start, j] == work[parent[j], j]
                        and start < parent[j]
                    )
                ):
                    parent[j] = start

        while True:
            best_vertex = -1
            best_cost = np.inf

            for j in range(n):
                if not in_tree[j] and parent[j] >= 0:
                    candidate = work[parent[j], j]
                    if (
                        candidate < best_cost
                        or (
                            candidate == best_cost
                            and (best_vertex < 0 or j < best_vertex)
                        )
                    ):
                        best_vertex = j
                        best_cost = candidate

            if best_vertex < 0 or not np.isfinite(best_cost):
                break

            p = parent[best_vertex]
            in_tree[best_vertex] = True
            component[best_vertex] = component_id
            tree_edges[p, best_vertex] = True
            tree_edges[best_vertex, p] = True

            for j in range(n):
                if valid[best_vertex, j] and not in_tree[j]:
                    if (
                        parent[j] == -1
                        or work[best_vertex, j] < work[parent[j], j]
                        or (
                            work[best_vertex, j] == work[parent[j], j]
                            and best_vertex < parent[j]
                        )
                    ):
                        parent[j] = best_vertex

        component_id += 1

    tree_degree = tree_edges.sum(axis=1).astype(float)

    # Find the maximum edge weight on the unique forest path between two nodes.
    def path_maximum(source: int, target: int) -> float:
        stack = [(source, -1, 0.0)]

        while stack:
            node, previous, current_max = stack.pop()
            if node == target:
                return current_max

            neighbors = np.flatnonzero(tree_edges[node])
            for neighbor in neighbors:
                if neighbor != previous:
                    stack.append(
                        (
                            int(neighbor),
                            node,
                            max(current_max, float(work[node, neighbor])),
                        )
                    )

        return np.inf

    # Compute the best non-tree replacement crossing each tree edge's cut.
    replacement = np.full((n, n), np.inf, dtype=float)

    for u in range(n):
        for v in range(u + 1, n):
            if not tree_edges[u, v]:
                continue

            # Removing (u, v) partitions its tree component. Mark u's side.
            side = np.zeros(n, dtype=bool)
            stack = [u]
            side[u] = True

            while stack:
                node = stack.pop()
                for neighbor in np.flatnonzero(tree_edges[node]):
                    if (
                        not (
                            (node == u and neighbor == v)
                            or (node == v and neighbor == u)
                        )
                        and not side[neighbor]
                    ):
                        side[neighbor] = True
                        stack.append(int(neighbor))

            best_replacement = np.inf
            for a in range(n):
                if not side[a]:
                    continue
                for b in range(n):
                    if side[b] or not valid[a, b]:
                        continue
                    if not tree_edges[a, b]:
                        best_replacement = min(
                            best_replacement, float(work[a, b])
                        )

            replacement[u, v] = best_replacement
            replacement[v, u] = best_replacement

    structural = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if not valid[i, j]:
                continue

            edge_cost = float(work[i, j])

            if tree_edges[i, j]:
                alternative = replacement[i, j]
                if np.isfinite(alternative):
                    # Tree edges are supported when removing them is expensive.
                    exchange = alternative / (alternative + edge_cost + np.finfo(float).eps)
                else:
                    exchange = 1.0
                backbone = 1.0
            elif component[i] == component[j]:
                path_max = path_maximum(i, j)
                if np.isfinite(path_max):
                    # Non-tree edges are supported when they are competitive
                    # replacements for an expensive forest edge.
                    exchange = (path_max + np.finfo(float).eps) / (
                        path_max + edge_cost + np.finfo(float).eps
                    )
                else:
                    exchange = 0.0
                backbone = 0.0
            else:
                exchange = 0.0
                backbone = 0.0

            structural[i, j] = 0.65 * exchange + 0.35 * backbone
            structural[j, i] = structural[i, j]

    max_structural = np.max(structural[valid])
    if max_structural > 0.0 and np.isfinite(max_structural):
        structural /= max_structural

    # Prefer edges that do not concentrate too much backbone degree at either end.
    degree_scale = max(1.0, float(np.max(tree_degree)))
    endpoint_balance = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if not valid[i, j]:
                continue

            degree_penalty = 0.5 * (
                tree_degree[i] / degree_scale
                + tree_degree[j] / degree_scale
            )
            balance = np.exp(-degree_penalty)
            endpoint_balance[i, j] = balance
            endpoint_balance[j, i] = balance

    score = 0.8 * structural + 0.2 * endpoint_balance
    score[~valid] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return score.astype(float, copy=False)
