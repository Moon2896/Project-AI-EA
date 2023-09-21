from argparse import ArgumentParser
from timeit import default_timer as timer

from src.evolutionary import fitness

def main():
    parser = ArgumentParser()
    parser.parse_args()

    # matrix representation of graph
    graph = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]

    print(fitness(graph, [0, 1, 2, 3]))

if __name__ == '__main__':
    main()