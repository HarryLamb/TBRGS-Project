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

def AS(nodes, edges, origin, destinations):
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

    best_g = {origin: 0}
    number_of_nodes = 1

    while frontier:
        frontier.sort(key=lambda n: (n.path_cost + n.heuristic, n.state, n.created_order))
        node = frontier.pop(0)
        state = node.state

        # Skip outdated worse copies of a node
        if node.path_cost > best_g.get(state, float("inf")):
            continue

        if state in destinations:
            return state, get_path(node), number_of_nodes

        neighbours = sorted(edges[state], key=lambda item: item[0])

        for neighbour, cost in neighbours:
            new_cost = node.path_cost + cost

            if new_cost < best_g.get(neighbour, float("inf")):
                best_g[neighbour] = new_cost
                created_counter += 1

                child = Node(
                    neighbour,
                    parent=node,
                    path_cost=new_cost,
                    heuristic=heuristic(neighbour, destinations, nodes),
                    created_order=created_counter
                )

                frontier.append(child)
                number_of_nodes += 1

    return None