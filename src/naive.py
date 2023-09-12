# Python3 program to implement traveling salesman
# problem using naive approach. sourced from geekforgeeks.org
from sys import maxsize
from itertools import permutations

# implementation of traveling Salesman Problem
def naive_TSP(graph, s):

	# store all vertex apart from source vertex
	vertex = []
	for i in range(len(graph)):
		if i != s:
			vertex.append(i)

	# store minimum weight Hamiltonian Cycle
	min_path = maxsize
	next_permutation=permutations(vertex)
	for i in next_permutation:

		# store current Path weight(cost)
		current_pathweight = 0

		# compute current path weight
		k = s
		for j in i:
			current_pathweight += graph[k][j]
			k = j
		current_pathweight += graph[k][s]

		# update minimum
		min_path = min(min_path, current_pathweight)

	return min_path

