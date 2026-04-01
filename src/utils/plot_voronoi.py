
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


def _as_2d_array(points):
    """Return points as a NumPy array with shape (N, 2)."""
    if points is None:
        return None
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.atleast_2d(array)


def _extract_cells(voronoi_partition):
    """Support either a partition object or a plain mapping of cells."""
    if hasattr(voronoi_partition, "Voronoi_Cells"):
        cells = voronoi_partition.Voronoi_Cells
    else:
        cells = voronoi_partition

    if isinstance(cells, dict):
        return list(cells.values())

    return list(cells)


def plot_voronoi_partition(
    map3d,
    voronoi_partition,
    drone_positions=None,
    waypoints=None,
    *,
    ax=None,
    show=True,
    title="Voronoi Partition",
):
    """Plot a 2D Voronoi partition with seeds and drone starting positions."""
    created_figure = ax is None
    if created_figure:
        _, ax = plt.subplots(figsize=(8, 8))

    workspace = np.asarray(map3d.workspace, dtype=float)
    cells = _extract_cells(voronoi_partition)
    drone_positions = _as_2d_array(drone_positions)
    waypoints = _as_2d_array(waypoints)
    cmap = plt.get_cmap("tab20", max(len(cells), 1))

    # Plot workspace boundary first so the partition remains inside it.
    workspace_patch = Polygon(
        workspace,
        closed=True,
        fill=False,
        edgecolor="black",
        linewidth=2.0,
        label="Workspace",
    )
    ax.add_patch(workspace_patch)

    for index, cell in enumerate(cells):
        polygon = _as_2d_array(getattr(cell, "polygon", None))
        if polygon is None or len(polygon) < 3:
            continue

        color = cmap(index)
        cell_patch = Polygon(
            polygon,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=1.5,
            alpha=0.30,
        )
        ax.add_patch(cell_patch)

        seed = _as_2d_array(getattr(cell, "seed", None))
        if seed is not None and len(seed) > 0:
            ax.scatter(
                seed[:, 0],
                seed[:, 1],
                color=color,
                marker="x",
                s=110,
                linewidths=2.0,
                label="Seed" if index == 0 else None,
                zorder=4,
            )

    for index, obstacle in enumerate(map3d.obstacles):
        obstacle_patch = Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            facecolor="dimgray",
            edgecolor="black",
            alpha=0.55,
            label="Obstacle" if index == 0 else None,
        )
        ax.add_patch(obstacle_patch)

    # Plot dei Waypoint
    if waypoints is not None and len(waypoints) > 0:
        ax.scatter(
            waypoints[:, 0],
            waypoints[:, 1],
            color="black",
            marker="o",
            s=15,
            alpha=0.6,
            label="Coverage Waypoints",
            zorder=3, # Sotto droni e seed, sopra le celle
        )

    if drone_positions is not None and len(drone_positions) > 0:
        ax.scatter(
            drone_positions[:, 0],
            drone_positions[:, 1],
            color="royalblue",
            s=70,
            edgecolors="white",
            linewidths=0.8,
            label="Drone initial positions",
            zorder=5,
        )

    ax.set_xlim(map3d.x_bounds[0], map3d.x_bounds[1])
    ax.set_ylim(map3d.y_bounds[0], map3d.y_bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

    if created_figure:
        plt.tight_layout()
        if show:
            plt.show()

    return ax
