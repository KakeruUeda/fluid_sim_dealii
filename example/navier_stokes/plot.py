import matplotlib
matplotlib.use("Agg")  # for docker (without GUI)

import matplotlib.pyplot as plt
import numpy as np
import h5py
from mpl_toolkits.mplot3d import Axes3D  

def visualize_solution_3d(file_path, output_path="solution_3d.png"):
    with h5py.File(file_path, "r") as f:
        nodes = f["nodes"][:]            
        solution = f["solution"][:, 0]    

    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(x, y, z, c=solution, cmap='viridis', s=3)
    fig.colorbar(p, ax=ax, label="Solution value")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Solution Field")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved 3D plot to {output_path}")

visualize_solution_3d("build/solution.h5")
