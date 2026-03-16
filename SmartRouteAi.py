import tkinter as tk
import random
import time
from collections import deque
import heapq
import matplotlib.pyplot as plt
import numpy as np

# GRID GENERATION
def generate_grid(size, obstacle_prob=0.3):
    grid = []
    for i in range(size):
        row = []
        for j in range(size):
            if random.random() < obstacle_prob:
                row.append(1)
            else:
                row.append(0)
        grid.append(row)
    grid[0][0] = 0
    grid[size-1][size-1] = 0
    return grid

# HEURISTIC FUNCTION
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# NEIGHBORS
def get_neighbors(node, grid):
    size = len(grid)
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    neighbors = []
    for d in directions:
        r = node[0] + d[0]
        c = node[1] + d[1]
        if 0 <= r < size and 0 <= c < size and grid[r][c] == 0:
            neighbors.append((r, c))
    return neighbors

# RECONSTRUCT PATH
def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node in parent:
        path.append(node)
        node = parent[node]
    path.append(start)
    path.reverse()
    return path

# BFS
def bfs(grid, start, goal):
    start_time = time.time()
    queue = deque([start])
    visited = set([start])
    parent = {}
    nodes = 0
    while queue:
        node = queue.popleft()
        nodes += 1
        if node == goal:
            break
        for n in get_neighbors(node, grid):
            if n not in visited:
                visited.add(n)
                parent[n] = node
                queue.append(n)
    return reconstruct_path(parent, start, goal), nodes, time.time()-start_time

# DFS
def dfs(grid, start, goal):
    start_time = time.time()
    stack = [start]
    visited = set([start])
    parent = {}
    nodes = 0
    while stack:
        node = stack.pop()
        nodes += 1
        if node == goal:
            break
        for n in get_neighbors(node, grid):
            if n not in visited:
                visited.add(n)
                parent[n] = node
                stack.append(n)
    return reconstruct_path(parent, start, goal), nodes, time.time()-start_time

# GREEDY
def greedy(grid, start, goal):
    start_time = time.time()
    heap = []
    heapq.heappush(heap, (heuristic(start, goal), start))
    visited = set([start])
    parent = {}
    nodes = 0
    while heap:
        _, node = heapq.heappop(heap)
        nodes += 1
        if node == goal:
            break
        for n in get_neighbors(node, grid):
            if n not in visited:
                visited.add(n)
                parent[n] = node
                heapq.heappush(heap, (heuristic(n, goal), n))
    return reconstruct_path(parent, start, goal), nodes, time.time()-start_time

# A*
def a_star(grid, start, goal):
    start_time = time.time()
    heap = []
    heapq.heappush(heap, (0, start))
    g_cost = {start: 0}
    parent = {}
    nodes = 0
    while heap:
        _, node = heapq.heappop(heap)
        nodes += 1
        if node == goal:
            break
        for n in get_neighbors(node, grid):
            new_cost = g_cost[node] + 1
            if n not in g_cost or new_cost < g_cost[n]:
                g_cost[n] = new_cost
                priority = new_cost + heuristic(n, goal)
                heapq.heappush(heap, (priority, n))
                parent[n] = node
    return reconstruct_path(parent, start, goal), nodes, time.time()-start_time

# VISUALIZATION
def visualize(grid, path):
    size = len(grid)
    display = np.array(grid)
    for (r,c) in path:
        display[r][c] = 2
    plt.figure()
    plt.title("Path Visualization")
    plt.imshow(display)
    plt.show()

# RUN PROJECT
def run_project():
    try:
        size = int(entry.get())
        grid = generate_grid(size)
        start = (0,0)
        goal = (size-1, size-1)
        results = {}
        results["BFS"] = bfs(grid, start, goal)
        results["DFS"] = dfs(grid, start, goal)
        results["Greedy"] = greedy(grid, start, goal)
        results["A*"] = a_star(grid, start, goal)
        
        print("\n--- AI Path Finding Comparison ---\n")
        print(f"{'Algorithm':<10} {'Path Length':<12} {'Nodes':<10} {'Time':<10}")
        for name, (path, nodes, t) in results.items():
            print(f"{name:<10} {len(path):<12} {nodes:<10} {t:<10.6f}")
        
        visualize(grid, results["A*"][0])
    except ValueError:
        print("Please enter a valid integer for grid size.")

# GUI
root = tk.Tk()
root.title("AI Path Finder")
tk.Label(root, text="Enter Grid Size:").pack()
entry = tk.Entry(root)
entry.pack()
tk.Button(root, text="Generate & Solve", command=run_project).pack()

root.mainloop()