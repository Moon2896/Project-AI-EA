from argparse import ArgumentParser
from src.naive import naive_TSP
from timeit import default_timer as timer

def main():
    parser = ArgumentParser()
    parser.parse_args()

    # matrix representation of graph
    graph = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
    s = 0

    start = timer()
    print(naive_TSP(graph, s));
    end = timer()
    print("Time elapsed: ", end - start)

if __name__ == '__main__':
    main()