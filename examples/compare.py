import numpy as np
import plotly.graph_objects as go
import random
import time

from pathfinding3d.core.diagonal_movement import DiagonalMovement
from pathfinding3d.core.grid import Grid
from pathfinding3d.finder.a_star import AStarFinder
from pathfinding3d.finder.theta_star import ThetaStarFinder
from pathfinding3d.finder.best_first import BestFirst
from pathfinding3d.finder.bi_a_star import BiAStarFinder
from pathfinding3d.finder.dijkstra import DijkstraFinder
from pathfinding3d.finder.ida_star import IDAStarFinder
from pathfinding3d.finder.msp import MinimumSpanningTree
from pathfinding3d.finder.breadth_first import BreadthFirstFinder


# Create a 3D numpy array with 0s as obstacles and 1s as walkable paths
matrix = np.ones((10, 10, 10), dtype=np.int8)
# mark the center of the grid as an obstacle

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        for k in range(matrix.shape[2]):
            rand = random.randrange(0, 20)
            if rand == 0:
                matrix[i][j][k] = 0

# Create a grid object from the numpy array
grid = Grid(matrix=matrix)

# Mark the start and end points
x_start = random.randrange(0, 10)
y_start = 0
z_start = random.randrange(0, 10)
x_end = random.randrange(0, 10)
y_end = 9
z_end = random.randrange(0, 10)

matrix[x_start][y_start][z_start] = 1
matrix[x_end][y_end][z_end] = 1

start = grid.node(x_start, y_start, z_start)
end = grid.node(x_end, y_end, z_end)

# List of finders to compare
finders = {
    "Theta*": ThetaStarFinder(diagonal_movement=DiagonalMovement.always),
    "A*": AStarFinder(diagonal_movement=DiagonalMovement.always),
    "Best First": BestFirst(diagonal_movement=DiagonalMovement.always),
    "Bi A*": BiAStarFinder(diagonal_movement=DiagonalMovement.always),
    "Dijkstra": DijkstraFinder(diagonal_movement=DiagonalMovement.always),
    # "IDA*": IDAStarFinder(diagonal_movement=DiagonalMovement.always),
    "Minimum Spanning Tree": MinimumSpanningTree(diagonal_movement=DiagonalMovement.always),
    "Breadth First": BreadthFirstFinder(diagonal_movement=DiagonalMovement.always),
}

paths = {}
costs = {}
runs = {}

for name, finder in finders.items():
    # Reset the grid to its initial state before each search
    grid.cleanup()
    start_time = time.perf_counter_ns()
    path, run = finder.find_path(start, end, grid)
    end_time = time.perf_counter_ns()
    print(f"{name} runtime:", (end_time - start_time), "nanoseconds")
    path = [p.identifier for p in path]
    paths[name] = path
    runs[name] = run

    def calculate_path_cost(path):
        cost = 0
        for pt, pt_next in zip(path[:-1], path[1:]):
            dx, dy, dz = pt_next[0] - pt[0], pt_next[1] - pt[1], pt_next[2] - pt[2]
            cost += (dx**2 + dy**2 + dz**2) ** 0.5
        return cost

    costs[name] = calculate_path_cost(path)

    # print(f"{name} operations: {run}, path length: {len(path)}, path cost: {costs[name]}")
    # print(f"{name} path: {path}")

# Create a plotly figure to visualize the paths
fig = go.Figure()

colors = ["blue", "red", "green", "purple", "orange", "cyan", "magenta", "yellow"]
for i, (name, path) in enumerate(paths.items()):
    fig.add_trace(
        go.Scatter3d(
            x=[pt[0] + 0.5 for pt in path],
            y=[pt[1] + 0.5 for pt in path],
            z=[pt[2] + 0.5 for pt in path],
            mode="lines + markers",
            line=dict(color=colors[i], width=4),
            marker=dict(size=4, color=colors[i]),
            name=f"{name} path",
            hovertext=[f"{name} path point"] * len(path),
        )
    )

# Add start, end, and obstacle points
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        for k in range(matrix.shape[2]):
            if matrix[i][j][k] == 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=[i + 0.5],
                        y=[j + 0.5],
                        z=[k + 0.5],
                        mode="markers",
                        marker=dict(color="black", size=7.5),
                        name="Obstacle",
                        hovertext=["Obstacle point"],
                )
            )

fig.add_trace(
    go.Scatter3d(
        x=[x_start + 0.5],
        y=[y_start + 0.5],
        z=[z_start + 0.5],
        mode="markers",
        marker=dict(color="green", size=7.5),
        name="Start",
        hovertext=["Start point"],
    )
)
fig.add_trace(
    go.Scatter3d(
        x=[x_end + 0.5],
        y=[y_end + 0.5],
        z=[z_end + 0.5],
        mode="markers",
        marker=dict(color="orange", size=7.5),
        name="End",
        hovertext=["End point"],
    )
)

# Define the camera position
camera = {
    "up": {"x": 0, "y": 0, "z": 1},
    "center": {"x": 0.1479269806756467, "y": 0.06501594452841505, "z": -0.0907033779622012},
    "eye": {"x": 1.3097359159706334, "y": 0.4710974884501846, "z": 2.095154166796815},
    "projection": {"type": "perspective"},
}

# Update the layout of the figure
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title="x - axis",
            backgroundcolor="white",
            gridcolor="lightgrey",
            showbackground=True,
            zerolinecolor="white",
            range=[0, 10],
            dtick=1,
        ),
        yaxis=dict(
            title="y - axis",
            backgroundcolor="white",
            gridcolor="lightgrey",
            showbackground=True,
            zerolinecolor="white",
            range=[0, 10],
            dtick=1,
        ),
        zaxis=dict(
            title="z - axis",
            backgroundcolor="white",
            gridcolor="lightgrey",
            showbackground=True,
            zerolinecolor="white",
            range=[0, 10],
            dtick=1,
        ),
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.7)",
    ),
    title=dict(text="Comparison of Pathfinding Algorithms"),
    scene_camera=camera,
)

# Save the figure as an HTML file (optional)
fig.write_html("compare.html", full_html=False, include_plotlyjs="cdn")
# Show the figure in a new tab
fig.show()