import os
import glob
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# io
i_dir = "inputs/aneurysm_A/pinn"
o_dir = "outputs/aneurysm_A/pinn"
os.makedirs(o_dir, exist_ok=True)

i_vtk_pattern = os.path.join(i_dir, "velocity_inlet_region_*.vtk") 
i_config = os.path.join(i_dir, "config.dat")

vel_name = "Velocity"
eps = 1e-10

config = np.loadtxt(i_config)
normal = np.array(config[:3], dtype=float)
normal /= np.linalg.norm(normal) 

results = [] 
n_steps = 0
mag_sum = 0.0

# vtk data series
for fname in sorted(glob.glob(i_vtk_pattern)):
    timestep = int(os.path.splitext(os.path.basename(fname))[0].split("_")[-1])
    
    # read vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(fname)
    reader.Update()
    poly = reader.GetOutput()

    # triangulate
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    mesh = tri.GetOutput()

    points = vtk_to_numpy(mesh.GetPoints().GetData())
    vel_cell = vtk_to_numpy(mesh.GetCellData().GetArray(vel_name))

    areas = []
    vels = []
    sum_vel = np.zeros(3)

    for i in range(mesh.GetNumberOfCells()):
        v = vel_cell[i]
        if np.linalg.norm(v) < eps:
            continue

        v_normal = np.dot(v, normal) * normal
        cell = mesh.GetCell(i)
        ids = [cell.GetPointId(j) for j in range(3)]
        p0, p1, p2 = points[ids]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        areas.append(area)
        vels.append(v_normal)

    if len(areas) == 0:
        avg_vel = np.array([0.0, 0.0, 0.0])
    else:
        areas = np.array(areas)
        vels = np.array(vels)
        avg_vel = (vels * areas[:, None]).sum(axis=0) / areas.sum()
    
    results.append((timestep, *avg_vel, np.linalg.norm(avg_vel)))
    print(f"timestep: {timestep}, velocity: {avg_vel}, velocity magnitude: {np.linalg.norm(avg_vel)}")

    mag_sum += np.linalg.norm(avg_vel)
    n_steps += 1

time_avg_vel = mag_sum / n_steps
print(f"time-averaged velocity: {time_avg_vel}")