import matplotlib.pyplot as plt

def plot_environment(workspace, obstacles, cells, drone_starts): 
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(workspace[0], workspace[2])
    ax.set_ylim(workspace[1], workspace[3])
    ax.set_aspect('equal')

    for obs in obstacles:
        x, y = obs.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.7)

    for cell in cells:
        if cell.is_empty:
            continue
        x, y = cell.exterior.xy
        ax.fill(x, y, alpha=0.5)

    ax.scatter(drone_starts[:, 0], drone_starts[:, 1], c='red', marker='x', label='Drone Starts')
    ax.legend()
    plt.show()

