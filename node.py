# Coding style adapted from aima-python structure (Github, 2021) and naming conventions shown in the lecture material
# Created by Harry Lamb and Ashton Foo

class Node:
    def __init__(self, state, parent=None, path_cost = 0, heuristic = 0, created_order = 0): # Modified by Ashton
        self.state = state
        self.parent = parent
        self.path_cost = path_cost # Added by Ashton
        self.heuristic = heuristic # Added by Ashton
        self.created_order = created_order # Added by Ashton


def get_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return list(reversed(path))