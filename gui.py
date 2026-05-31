# Created by Harry Lamb

import json
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

# Load default settings from config file
with open("config.json") as f:
    config = json.load(f)

SCATS_FILE = config["scats_file"]
MODELS_DIR = config["models_dir"]

nodes = None
edges = None
simple_nodes = None
last_paths = {}
best_path_name = None
current_model = None
current_slot = None


def run_search():
    global nodes, edges, simple_nodes, current_model, current_slot, best_path_name

    try:
        origin = int(origin_entry.get())
        destination = int(destination_entry.get())
        time_slot = int(slot_entry.get())
        if not 0 <= time_slot <= 95:
            messagebox.showerror("Error", "Time slot must be between 0 and 95.")
            return
    except ValueError:
        messagebox.showerror("Error", "All inputs must be integers.")
        return

    selected_model = model_var.get()

    # Rebuild graph only if the ML model or the timeslot changes 
    if nodes is None or selected_model != current_model or time_slot != current_slot:
        nodes, edges = build_graph(SCATS_FILE, models_dir=MODELS_DIR,
                                   model_type=selected_model, time_slot=time_slot)
        simple_nodes = {site: (info["lat"], info["lon"]) for site, info in nodes.items()}
        current_model = selected_model
        current_slot = time_slot

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

    # Run all algorithms and store the results
    results = []
    last_paths.clear()
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

    if not results:
        result_label.config(text="No path found.")
        return

    # Sort by travel time ascending so best route is first
    results.sort(key=lambda x: x[3])
    best_path_name = results[0][0]

    hour = (time_slot * 15) // 60
    minute = (time_slot * 15) % 60
    lines = [f"Results — {hour:02d}:{minute:02d}, Model: {selected_model}\n"]
    for i, (name, path, num_nodes, total_cost) in enumerate(results, 1):
        path_str = ", ".join(str(s) for s in path)
        lines.append(f"{i}. {name} ({num_nodes} nodes explored)\n   Travel Time: {total_cost:.0f}s\n   Path: {path_str}")

    result_label.config(text="\n\n".join(lines))


def show_map():
    if not last_paths:
        messagebox.showerror("Error", "Run a search first.")
        return
    # Use the best route (lowest travel time) as the highlighted route
    best_name = best_path_name or "UCS"
    path = last_paths.get(best_name) or next(iter(last_paths.values()))
    visualise_route(nodes, path, best_name=best_name, all_paths=last_paths)
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

slot_label = tk.Label(root, text="Time Slot (0-95, e.g. 32=8am, 60=3pm, 68=5pm):")
slot_label.pack()
slot_entry = tk.Entry(root, width=20)
slot_entry.insert(0, str(config["default_time_slot"]))
slot_entry.pack(pady=5)

model_label = tk.Label(root, text="Prediction Model:")
model_label.pack()
model_var = tk.StringVar(value=config["prediction_model"])
model_menu = tk.OptionMenu(root, model_var, "random_forest", "lstm", "gru")
model_menu.pack(pady=5)

search_button = tk.Button(root, text="Find Route", command=run_search)
search_button.pack(pady=10)

map_button = tk.Button(root, text="Show Route on OpenStreetMap", command=show_map)
map_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Times", 10), wraplength=700, justify="left")
result_label.pack(pady=10)

root.mainloop()