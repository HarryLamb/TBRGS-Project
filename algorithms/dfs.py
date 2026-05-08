# Coding style adapted from aima-python structure (Github, 2021) and naming conventions shown in the lecture material
# Created by Harry Lamb

from node import Node, get_path

def DFS(edges, origin, destinations):
    frontier = []
    frontier.append(Node(origin))
    explored = set()
    number_of_nodes = 1

    while frontier:
        node = frontier.pop() # Pops the most recently added item off the stack (LIFO)
        state = node.state
        if state in explored:
            continue
        if state in destinations: # Found goal
            return state, get_path(node), number_of_nodes
        explored.add(state)
        neighbours = sorted([neighbour for neighbour, cost in edges[state]], reverse=True) # Adapted from (W3Schools, 2026), reversed used for LIFO methodology, cost not used but has to be included
        for neighbour in neighbours:
            child = Node(neighbour, node)
            frontier.append(child)
            number_of_nodes += 1