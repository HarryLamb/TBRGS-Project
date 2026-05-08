# Created by Ashton Foo

from math import sqrt
from node import Node, get_path

def heuristic(state, destinations, nodes):
    x1, y1 = nodes[state]
    best_distance = float("inf")

    for goal in destinations:
        x2, y2 = nodes[goal]
        distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if distance < best_distance:
            best_distance = distance

    return best_distance

def GBFS(nodes, edges, origin, destinations):
    created_counter = 0
    frontier = [
        Node(
            origin,
            parent=None,
            path_cost=0,
            heuristic=heuristic(origin, destinations, nodes),
            created_order=created_counter
        )
    ]

    explored = set()
    number_of_nodes = 1

    while frontier:
        frontier.sort(key=lambda n: (n.heuristic, n.state, n.created_order))
        node = frontier.pop(0)
        state = node.state

        if state in explored:
            continue

        if state in destinations:
            return state, get_path(node), number_of_nodes

        explored.add(state)

        neighbours = sorted(edges[state], key=lambda item: item[0])

        for neighbour, cost in neighbours:
            if neighbour not in explored:
                created_counter += 1
                child = Node(
                    neighbour,
                    parent=node,
                    path_cost=node.path_cost + cost,
                    heuristic=heuristic(neighbour, destinations, nodes),
                    created_order=created_counter
                )
                frontier.append(child)
                number_of_nodes += 1

    return None