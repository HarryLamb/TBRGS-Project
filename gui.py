# Created by Harry Lamb

import tkinter as tk
from tkinter import messagebox
import webbrowser

from graphbuilder import build_graph
from algorithms.ucs import UCS
from algorithms.as_search import AS
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.gbfs import GBFS
from algorithms.idastar import IDA
from visualisation import visualise_route

SCATS_FILE = "Scats Data October 2006.xls"

nodes, edges = build_graph(SCATS_FILE)
simple_nodes = {site: (info["lat"], info["lon"]) for site, info in nodes.items()}
last_paths = {}


def run_search():
    global nodes, edges, simple_nodes
    try:
        origin = int(origin_entry.get())
        destination = int(destination_entry.get())
    except ValueError:
        messagebox.showerror("Error", "Origin and destination must be SCATS site numbers (integers).")
        return

    if origin not in edges:
        messagebox.showerror("Error", f"Origin {origin} not found in graph.")
        return
    if destination not in edges:
        messagebox.showerror("Error", f"Destination {destination} not found in graph.")
        return

    algorithms = {
        "BFS":  lambda: BFS(edges, origin, {destination}),
        "DFS":  lambda: DFS(edges, origin, {destination}),
        "UCS":  lambda: UCS(edges, origin, {destination}),
        "A*":   lambda: AS(simple_nodes, edges, origin, {destination}),
        "GBFS": lambda: GBFS(simple_nodes, edges, origin, {destination}),
        "IDA*": lambda: IDA(simple_nodes, edges, origin, {destination}),
    }

    results = []
    for name, fn in algorithms.items():
        result = fn()
        if result is None:
            continue
        _, path, num_nodes = result
        total_cost = sum(
            cost for i in range(len(path) - 1)
            for neighbour, cost in edges[path[i]]
            if neighbour == path[i + 1]
        )
        results.append((name, path, num_nodes, total_cost))
        last_paths[name] = path

    results.sort(key=lambda x: x[3])

    lines = ["Results by Algorithm (Top-K Routes):\n"]
    for i, (name, path, num_nodes, total_cost) in enumerate(results, 1):
        path_str = ", ".join(str(s) for s in path)
        lines.append(f"{i}. {name} ({num_nodes} nodes explored)\n   Travel Time: {total_cost:.0f}s\n   Path: {path_str}")

    if not results:
        result_label.config(text="No path found.")
        return

    result_label.config(text="\n\n".join(lines))


def show_map():
    if not last_paths:
        messagebox.showerror("Error", "Run a search first.")
        return
    path = last_paths.get("UCS") or next(iter(last_paths.values()))
    visualise_route(nodes, path)
    webbrowser.open("route.html")


root = tk.Tk()
root.title("An Group 6 Assignment 2B TBRGS")
root.geometry("800x800")

title = tk.Label(root, text="Traffic Based Route Guidance System", font=("Times", 16, "bold"))
title.pack(pady=10)

credit = tk.Label(root, text="GUI Created by Harry Lamb", font=("Times", 8, "bold"))
credit.pack(pady=10)

origin_label = tk.Label(root, text="Origin:")
origin_label.pack()

origin_entry = tk.Entry(root, width=20)
origin_entry.pack(pady=5)

destination_label = tk.Label(root, text="Destination:")
destination_label.pack()

destination_entry = tk.Entry(root, width=20)
destination_entry.pack(pady=5)

search_button = tk.Button(root, text="Find Route", command=run_search)
search_button.pack(pady=10)

map_button = tk.Button(root, text="Show Visualisation on the OpenStreetMap", command=show_map)
map_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Times", 10), wraplength=700, justify="left")
result_label.pack(pady=10)

root.mainloop()