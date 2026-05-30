# Created by Ashton Foo

from math import sqrt, radians, sin, cos, atan2
from node import Node, get_path

def heuristic(state, destinations, nodes): # Haversine adapted from W3Tutorials.net
    x1, y1 = nodes[state]
    best = float("inf")
    for goal in destinations:
        x2, y2 = nodes[goal]
        R = 6371.0
        dlat = radians(x2 - x1)
        dlon = radians(y2 - y1)
        a = sin(dlat/2)**2 + cos(radians(x1)) * cos(radians(x2)) * sin(dlon/2)**2
        km = R * 2 * atan2(sqrt(a), sqrt(1 - a))
        seconds = (km / 60.0) * 3600.0
        if seconds < best:
            best = seconds
    return best


def AS(nodes, edges, origin, destinations):
    created_counter = 0
    frontier = [
        Node(origin, parent=None, path_cost=0,
             heuristic=heuristic(origin, destinations, nodes),
             created_order=created_counter)
    ]
    best_g = {origin: 0}
    number_of_nodes = 1

    while frontier:
        frontier.sort(key=lambda n: (n.path_cost + n.heuristic, n.state, n.created_order))
        node = frontier.pop(0)
        state = node.state

        if node.path_cost > best_g.get(state, float("inf")):
            continue
        if state in destinations:
            return state, get_path(node), number_of_nodes

        for neighbour, cost in sorted(edges[state], key=lambda x: x[0]):
            new_cost = node.path_cost + cost
            if new_cost < best_g.get(neighbour, float("inf")):
                best_g[neighbour] = new_cost
                created_counter += 1
                child = Node(neighbour, parent=node, path_cost=new_cost,
                             heuristic=heuristic(neighbour, destinations, nodes),
                             created_order=created_counter)
                frontier.append(child)
                number_of_nodes += 1

    return None