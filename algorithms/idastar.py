from math import sqrt


def heuristic(node, nodes, destinations):
    x1, y1 = nodes[node]
    best_distance = float("inf")

    for dest in destinations:
        x2, y2 = nodes[dest]
        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance < best_distance:
            best_distance = distance

    return best_distance


def IDA(nodes, edges, origin, destinations):
    threshold = heuristic(origin, nodes, destinations)
    number_of_nodes = 1

    while True:
        path = [origin]
        visited = {origin}

        found_path, next_threshold, created_nodes = depth_limited_search(
            current_node=origin,
            g_cost=0,
            threshold=threshold,
            path=path,
            visited=visited,
            nodes=nodes,
            edges=edges,
            destinations=destinations
        )

        number_of_nodes += created_nodes

        if found_path is not None:
            return found_path[-1], found_path, number_of_nodes

        if next_threshold == float("inf"):
            return None, [], number_of_nodes

        threshold = next_threshold


def depth_limited_search(current_node, g_cost, threshold, path, visited, nodes, edges, destinations):
    h_cost = heuristic(current_node, nodes, destinations)
    f_cost = g_cost + h_cost

    if f_cost > threshold:
        return None, f_cost, 0

    if current_node in destinations:
        return path[:], f_cost, 0

    minimum_exceeded = float("inf")
    created_nodes = 0

    for neighbour, edge_cost in sorted(edges.get(current_node, [])):
        if neighbour in visited:
            continue

        path.append(neighbour)
        visited.add(neighbour)
        created_nodes += 1

        found_path, temp_threshold, child_created = depth_limited_search(
            current_node=neighbour,
            g_cost=g_cost + edge_cost,
            threshold=threshold,
            path=path,
            visited=visited,
            nodes=nodes,
            edges=edges,
            destinations=destinations
        )

        created_nodes += child_created

        if found_path is not None:
            return found_path, temp_threshold, created_nodes

        if temp_threshold < minimum_exceeded:
            minimum_exceeded = temp_threshold

        path.pop()
        visited.remove(neighbour)

    return None, minimum_exceeded, created_nodes