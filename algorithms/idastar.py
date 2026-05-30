# Created by Ashton Foo

from math import sqrt, radians, sin, cos, atan2


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


def IDA(site_nodes, edges, origin, destinations):
    threshold = heuristic(origin, destinations, site_nodes)
    all_created = {origin}

    while True:
        path = [origin]
        visited = {origin}

        found_path, next_threshold = depth_limited_search(
            current_node=origin,
            g_cost=0,
            threshold=threshold,
            path=path,
            visited=visited,
            site_nodes=site_nodes,
            edges=edges,
            destinations=destinations,
            all_created=all_created
        )

        if found_path is not None:
            return found_path[-1], found_path, len(all_created)

        if next_threshold == float("inf"):
            return None, [], len(all_created)

        threshold = next_threshold


def depth_limited_search(current_node, g_cost, threshold, path, visited, site_nodes, edges, destinations, all_created):
    h_cost = heuristic(current_node, destinations, site_nodes)
    f_cost = g_cost + h_cost

    if f_cost > threshold:
        return None, f_cost

    if current_node in destinations:
        return path[:], f_cost

    minimum_exceeded = float("inf")

    for neighbour, edge_cost in sorted(edges.get(current_node, [])):
        if neighbour in visited:
            continue

        all_created.add(neighbour)
        path.append(neighbour)
        visited.add(neighbour)

        found_path, temp_threshold = depth_limited_search(
            current_node=neighbour,
            g_cost=g_cost + edge_cost,
            threshold=threshold,
            path=path,
            visited=visited,
            site_nodes=site_nodes,
            edges=edges,
            destinations=destinations,
            all_created=all_created
        )

        if found_path is not None:
            return found_path, temp_threshold

        if temp_threshold < minimum_exceeded:
            minimum_exceeded = temp_threshold

        path.pop()
        visited.remove(neighbour)

    return None, minimum_exceeded