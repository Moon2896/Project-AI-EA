from sys import maxsize
from ..tools.toolbox import pairwise

WORST_FITNESS = maxsize

def fitness(graph: [[int]], path: [int]) -> int:
    """Computes the fitness of a path in a graph.
    The fitness is the sum of the weights of the edges in the path.
    The path must be a list of vertices, and the graph must be a matrix
    representation of the graph.

    The path must be a Hamiltonian path, i.e.
    it must visit every vertex exactly once and return to the starting vertex.

    If the path doesn't return to the starting vertex, it will append the
    weight of the edge between the last and the first vertex to the fitness."""
    fitness_value = 0
    if len(path) != len(graph) and path[0] != path[-1]:
        return WORST_FITNESS

    pairs=pairwise(path)
    for (a, b) in pairs:
        fitness_value += graph[a][b]
    if path[0] != path[-1]:
        fitness_value += graph[path[-1]][path[0]]
    return fitness_value

