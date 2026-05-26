import tkinter as tk
from tkinter import messagebox

from graphbuilder import build_graph
from algorithms.ucs import UCS
from algorithms.as_search import AS
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.gbfs import GBFS
from algorithms.idastar import IDA

SCATS_FILE = "Scats Data October 2006.xls"

nodes, edges = build_graph(SCATS_FILE, time_slot=32)


def run_search():
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

    algo = algo_var.get()

    if algo == "BFS":
        result = BFS(edges, origin, {destination})
    elif algo == "DFS":
        result = DFS(edges, origin, {destination})
    elif algo == "UCS":
        result = UCS(edges, origin, {destination})
    elif algo == "A*":
        result = AS(edges, nodes, origin, destination)
    elif algo == "GBFS":
        result = GBFS(edges, nodes, origin, destination)
    elif algo == "IDA*":
        result = IDA(edges, nodes, origin, destination)
    else:
        result = None

    if result is None:
        result_label.config(text="No path found.")
        return

    goal, path, num_nodes = result

    path_str = " -> ".join(str(s) for s in path)

    total_cost = 0
    for i in range(len(path) - 1):
        for neighbour, cost in edges[path[i]]:
            if neighbour == path[i + 1]:
                total_cost += cost
                break

    result_label.config(
        text=f"Path: {path_str}\nTotal Travel Time: {total_cost:.2f} seconds\nNodes explored: {num_nodes}"
    )


root = tk.Tk()
root.title("An Group 6 Assignment 2B TBRGS")
root.geometry("800x600")

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

algo_var = tk.StringVar(value="UCS")

algo_menu = tk.OptionMenu(
    root,
    algo_var,
    "BFS",
    "DFS",
    "UCS",
    "A*",
    "GBFS",
    "IDA*"
)
algo_menu.pack(pady=5)

search_button = tk.Button(root, text="Find Route", command=run_search)
search_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Times", 10), wraplength=700, justify="left")
result_label.pack(pady=10)

root.mainloop()