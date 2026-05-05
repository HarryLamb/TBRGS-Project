import tkinter as tk

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

root.mainloop()