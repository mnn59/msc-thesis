import numpy as np

# Abilene topology from Abi.txt: 11 nodes, 14 bidirectional links
# Format: src dst distance capacity loss
links = [
    (1,4,1176), (1,5,587), (1,11,846),
    (2,5,260), (2,8,700),
    (3,6,639), (3,9,1295), (3,10,2095),
    (4,6,902), (4,7,1893),
    (5,6,548),
    (7,9,366),
    (8,11,233),
    (9,10,861),
]

N = 11
INF = 1e9
# Floyd-Warshall with hop count (not distance)
dist = [[INF]*N for _ in range(N)]
for i in range(N):
    dist[i][i] = 0
for s,t,d in links:
    # 1-indexed to 0-indexed
    dist[s-1][t-1] = 1
    dist[t-1][s-1] = 1

for k in range(N):
    for i in range(N):
        for j in range(N):
            if dist[i][k] + dist[k][j] < dist[i][j]:
                dist[i][j] = dist[i][k] + dist[k][j]

# Print shortest path matrix
total = 0
count = 0
print('Shortest path hop count matrix:')
for i in range(N):
    row = []
    for j in range(N):
        if i != j:
            total += dist[i][j]
            count += 1
        row.append(int(dist[i][j]) if dist[i][j] < INF else -1)
    print(f'  Node {i+1}: {row}')

avg = total / count
print(f'\nAverage shortest path (hop): {avg:.4f}')
print(f'Total pairs: {count}')
print(f'Min: {min(dist[i][j] for i in range(N) for j in range(N) if i!=j)}')
print(f'Max: {max(dist[i][j] for i in range(N) for j in range(N) if i!=j and dist[i][j]<INF)}')
