# file: astar_app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import heapq

# ----------------------------
# A* Algorithm
# ----------------------------
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j            
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue
            else:
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return False


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🔍 A* Pathfinding Visualization")

grid_size = st.slider("Grid size", 5, 20, 10)
start = (0, 0)
goal = (grid_size-1, grid_size-1)

# Obstacle generation
np.random.seed(42)
grid = np.zeros((grid_size, grid_size))
for _ in range(grid_size*2):
    x, y = np.random.randint(0, grid_size, 2)
    if (x, y) != start and (x, y) != goal:
        grid[x, y] = 1

path = astar(grid, start, goal)

fig, ax = plt.subplots()
ax.imshow(grid, cmap="gray_r")

if path:
    px, py = zip(*path)
    ax.plot(py, px, marker="o", color="red")

ax.plot(start[1], start[0], "go")  # start
ax.plot(goal[1], goal[0], "bo")   # goal

st.pyplot(fig)
