import math


def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int,
) -> list[tuple[float, ...]]:
    centroids = initial_centroids
    for _ in range(max_iterations):
        # assign points to centroids
        clusters: list[list[tuple[float, ...]]] = [[] for _ in range(k)]
        for p in points:
            i = min(range(k), key=lambda i: distance(p, centroids[i]))
            clusters[i].append(p)

        # update the centroids based on point assignments
        centroids = [
            tuple(sum(x[i] for x in g) / len(g) for i in range(len(g[0])))
            for g in clusters
        ]

    return [tuple(round(x, 4) for x in c) for c in centroids]


def distance(x: tuple[float, ...], y: tuple[float, ...]) -> float:
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


def test_k_means_clustering() -> None:
    points: list[tuple[float, ...]] = [
        (1, 2),
        (1, 4),
        (1, 0),
        (10, 2),
        (10, 4),
        (10, 0),
    ]
    k = 2
    initial_centroids: list[tuple[float, ...]] = [(1, 1), (10, 1)]
    max_iterations = 10
    output = [(1, 2), (10, 2)]
    assert k_means_clustering(points, k, initial_centroids, max_iterations) == output
