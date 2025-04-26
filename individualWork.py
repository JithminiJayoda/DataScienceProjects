import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq

# Create Grid
grid = [
    [
        {
            'x': x,
            'y': y,
            'is_obstacle': False,
            'is_start': (x == 0 and y == 0),
            'is_goal': (x == 6 and y == 6),
            'cost': 1
        }
        for y in range(7)
    ]
    for x in range(7)
]

# Place random obstacles
def place_obstacles(grid, num_obstacles=10):
    count = 0
    while count < num_obstacles:
        x, y = random.randint(0, 6), random.randint(0, 6)
        if (x, y) not in [(0, 0), (6, 6)] and not grid[x][y]['is_obstacle']:
            grid[x][y]['is_obstacle'] = True
            count += 1

def clear_obstacles(grid):
    for row in grid:
        for cell in row:
            cell['is_obstacle'] = False

def generate_valid_grid(grid, num_obstacles=10):
    while True:
        clear_obstacles(grid)
        place_obstacles(grid, num_obstacles)
        path, _ = a_star((0, 0), (6, 6), grid)
        if path:
            return path

# A* Algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = []

    while open_set:
        _, current = heapq.heappop(open_set)
        visited.append(current)

        if current == goal:
            return reconstruct_path(came_from, current), visited

        x, y = current
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 7 and 0 <= ny < 7:
                if not grid[nx][ny]['is_obstacle']:
                    neighbor = (nx, ny)
                    tentative_g_score = g_score[current] + grid[nx][ny]['cost']

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None, visited

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Visualization Functions
def draw_grid(grid, path=None, visited=None):
    fig, ax = plt.subplots()
    for x in range(7):
        for y in range(7):
            cell = grid[x][y]
            if cell['is_start']:
                ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='green'))
                ax.text(y+0.5, 6-x+0.5, 'S', ha='center', va='center', color='white')
            elif cell['is_goal']:
                ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='red'))
                ax.text(y+0.5, 6-x+0.5, 'G', ha='center', va='center', color='white')
            elif cell['is_obstacle']:
                ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='black'))
            else:
                ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='white', edgecolor='black'))

    if visited:
        for (x, y) in visited:
            ax.add_patch(plt.Circle((y+0.5,6-x+0.5),0.2,color='yellow'))

    if path:
        for (x, y) in path:
            ax.add_patch(plt.Circle((y+0.5,6-x+0.5),0.2,color='blue'))

    ax.set_xlim(0,7)
    ax.set_ylim(0,7)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Animated Search Visualization
def animate_search(grid, path, visited):
    fig, ax = plt.subplots()

    patches = []

    def init():
        ax.clear()
        for x in range(7):
            for y in range(7):
                cell = grid[x][y]
                if cell['is_start']:
                    ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='green'))
                    ax.text(y+0.5, 6-x+0.5, 'S', ha='center', va='center', color='white')
                elif cell['is_goal']:
                    ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='red'))
                    ax.text(y+0.5, 6-x+0.5, 'G', ha='center', va='center', color='white')
                elif cell['is_obstacle']:
                    ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='black'))
                else:
                    ax.add_patch(plt.Rectangle((y,6-x),1,1,fc='white', edgecolor='black'))

        ax.set_xlim(0,7)
        ax.set_ylim(0,7)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.gca().set_aspect('equal', adjustable='box')

    def update(frame):
        if frame < len(visited):
            x, y = visited[frame]
            ax.add_patch(plt.Circle((y+0.5,6-x+0.5),0.2,color='yellow'))
        if frame == len(visited):
            for (x, y) in path:
                ax.add_patch(plt.Circle((y+0.5,6-x+0.5),0.2,color='blue'))

    ani = animation.FuncAnimation(fig, update, frames=len(visited)+1, init_func=init, repeat=False, interval=200)
    plt.show()

# Run
random.seed(42)
path = generate_valid_grid(grid, num_obstacles=10)
path, visited = a_star((0,0), (6,6), grid)

if path:
    print("Path found!")
    draw_grid(grid, path, visited)
    animate_search(grid, path, visited)
else:
    print("No path found.")
