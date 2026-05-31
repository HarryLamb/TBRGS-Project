# Created by Harry Lamb

import folium

# Offset for the coordinates
LAT_OFFSET = 0.00129
LON_OFFSET = 0.00126


def correct_coordinates(lat, lon):
    # Apply the offset to help improve longitude and latitude of the SCATS sites
    return lat + LAT_OFFSET, lon + LON_OFFSET


def visualise_route(nodes, best_path, best_name="Best", all_paths=None):
    # Puts the best path in the center of the screen
    centre_lat = sum(nodes[s]["lat"] for s in best_path) / len(best_path)
    centre_lon = sum(nodes[s]["lon"] for s in best_path) / len(best_path)
    centre_lat, centre_lon = correct_coordinates(centre_lat, centre_lon)

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=14)

    # Draw all SCATS sites as dots on the map which can be clicked to see their information
    for site, info in nodes.items():
        lat, lon = correct_coordinates(info["lat"], info["lon"])
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="grey",
            fill=True,
            fill_opacity=0.5,
            popup=f"Site {site}: {info['location']}"
        ).add_to(m)

    # Draws the other routes the user could choose to follow
    if all_paths:
        seen = set()
        for name, path in all_paths.items():
            path_tuple = tuple(path)
            if path_tuple == tuple(best_path) or path_tuple in seen:
                continue
            seen.add(path_tuple)
            coords = [correct_coordinates(nodes[s]["lat"], nodes[s]["lon"]) for s in path]
            folium.PolyLine(
                coords,
                color="blue",
                weight=3,
                opacity=0.25,
                tooltip=name
            ).add_to(m)

    # Draws the best path
    coords = [correct_coordinates(nodes[s]["lat"], nodes[s]["lon"]) for s in best_path]
    folium.PolyLine(
        coords,
        color="blue",
        weight=5,
        opacity=1.0,
        tooltip=f"Best route ({best_name})"
    ).add_to(m)

    # Marks the origin and destination on the map
    folium.Marker(coords[0], popup="Origin", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(coords[-1], popup="Destination", icon=folium.Icon(color="red")).add_to(m)

    m.save("route.html")