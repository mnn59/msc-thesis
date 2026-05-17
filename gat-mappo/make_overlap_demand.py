#!/usr/bin/env python3
"""
Create an overlapping-destination demand matrix for Abi.

Output:
  net_env/inputs/Abi/Abi_overlap.txt

This scenario increases the probability of OD pairs whose shortest paths
cross the Abi bottleneck area around link 1-5, zero-based edge (0, 4).

Why this helps CTDE evaluation:
  - Many active flows share intermediate links.
  - Independent PPO/DRL-OR agents may make locally good but globally conflicting decisions.
  - MAPPO/CTDE should learn more coordinated behavior through the centralized critic.
"""

from collections import deque
from pathlib import Path


TOPO_FILE = Path("net_env/inputs/Abi/Abi.txt")
OUT_FILE = Path("net_env/inputs/Abi/Abi_overlap.txt")
META_FILE = Path("net_env/inputs/Abi/Abi_overlap_pairs.txt")


def read_abi_topology(path: Path):
    with path.open("r", encoding="utf-8") as f:
        n, m = map(int, f.readline().split())
        edges = []
        for _ in range(m):
            u, v, _delay, _capacity, _loss = map(int, f.readline().split())
            u -= 1
            v -= 1
            edges.append((u, v))
    return n, edges


def shortest_path(adj, src, dst):
    n = len(adj)
    parent = [-1] * n
    q = deque([src])
    parent[src] = src

    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)

    if parent[dst] == -1:
        return []

    path = []
    cur = dst
    while cur != src:
        path.append(cur)
        cur = parent[cur]
    path.append(src)
    path.reverse()
    return path


def path_uses_edge(path, a, b):
    for u, v in zip(path, path[1:]):
        if (u == a and v == b) or (u == b and v == a):
            return True
    return False


def main():
    n, edges = read_abi_topology(TOPO_FILE)

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Bottleneck / shared area in Abi.
    # In Abi.txt this is node 1-5, zero-based 0-4.
    bottleneck_u = 0
    bottleneck_v = 4

    # Base matrix: keep a small probability for other pairs.
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = 1

    selected = []

    # Pick all OD pairs whose shortest path crosses the bottleneck area.
    for s in range(n):
        for t in range(n):
            if s == t:
                continue
            p = shortest_path(adj, s, t)
            if p and path_uses_edge(p, bottleneck_u, bottleneck_v):
                selected.append((s, t, p))

    # Strongly bias the selected overlapping OD pairs.
    # 100 means these pairs are sampled far more often than normal pairs.
    for s, t, _p in selected:
        matrix[s][t] = 100

    # Extra emphasis on common-destination flows.
    # These create destination-overlap and shared middle-link pressure.
    common_destination_pairs = [
        (10, 4),  # 11 -> 5
        (6, 4),   # 7  -> 5
        (3, 4),   # 4  -> 5
        (0, 4),   # 1  -> 5
        (6, 1),   # 7  -> 2
        (3, 1),   # 4  -> 2
        (0, 1),   # 1  -> 2
    ]

    for s, t in common_destination_pairs:
        if s != t:
            matrix[s][t] = 200

    flat = []
    for i in range(n):
        for j in range(n):
            flat.append(str(matrix[i][j]))

    OUT_FILE.write_text(" ".join(flat) + "\n", encoding="utf-8")

    with META_FILE.open("w", encoding="utf-8") as f:
        f.write("Selected OD pairs whose shortest path crosses Abi link 1-5, zero-based (0,4)\n")
        f.write("Format: src dst path, all shown as 1-based node IDs\n\n")
        for s, t, p in selected:
            p_1based = [x + 1 for x in p]
            f.write(f"{s + 1:2d} -> {t + 1:2d}: {p_1based}\n")

    print(f"Written: {OUT_FILE}")
    print(f"Metadata: {META_FILE}")
    print(f"Selected overlapping OD pairs: {len(selected)}")
    print("Use this with: --demand-matrix Abi_overlap.txt")


if __name__ == "__main__":
    main()