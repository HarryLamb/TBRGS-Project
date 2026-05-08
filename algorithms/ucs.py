# Coding style adapted from aima-python structure (Github, 2021) and naming conventions shown in the lecture material
# Adapted from two examples/online sources (GeeksforGeeks, 2025) (Soularidis, 2021)
# Created by Harry Lamb

import heapq
from node import Node, get_path

def UCS(edges, origin, destinations):
    frontier = []
    insertion_order = 0 
    heapq.heappush(frontier, (0, insertion_order, Node(origin))) # Push origin to the heap with a cost of 0
    explored = set()
    number_of_nodes = 1

    while frontier:
        cost, _, node = heapq.heappop(frontier) # Pop the lowest cost node off the heap, insertion order is ignored here on purpose
        state = node.state
        if state in explored:
            continue
        if state in destinations: # Found goal
            return state, get_path(node), number_of_nodes
        explored.add(state)

        for neighbour, edge_cost in sorted(edges[state]): 
            child = Node(neighbour, node)
            insertion_order += 1 
            heapq.heappush(frontier, (cost + edge_cost, insertion_order, child)) # Push with the cumulative node cost
            number_of_nodes += 1