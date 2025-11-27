import numpy as np
from matplotlib import pyplot as plt
import os
import h5py
import meshio

i_name = "stenosis_asymmetric_mid_Wo_dt_obs_4e-2"
i_dir = "inputs/" + i_name
o_dir = "outputs/" + i_name

o_dir_vtu_point_cloud = o_dir + "/vtu_point_cloud"
os.makedirs(o_dir_vtu_point_cloud, exist_ok=True)

o_vtu_vel = o_dir_vtu_point_cloud + "/velocity.vtu"
o_vtu_acc = o_dir_vtu_point_cloud + "/acceleration.vtu"

i_h5 = o_dir + "/data_obs.h5"

# --- helper: write VTU point cloud ---
def write_pointcloud_vtu(out_path, pts, val, name):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    npts = pts.shape[0]
    vertices = np.arange(npts, dtype=np.int64).reshape(-1, 1)
    mesh = meshio.Mesh(
        points=pts.astype(np.float64),
        cells=[("vertex", vertices)],
        point_data={name: val.astype(np.float32)},
    )
    meshio.write(out_path, mesh, file_format="vtu")
    print(f"Saved VTU: {out_path}")

with h5py.File(i_h5, "r") as h5:
    print("keys under root:", list(h5.keys()))  # ['a', 'u', 'x']

    x = h5["x"][...]
    u = h5["u"][...]   # shape could be (N,3) or (n_steps,N,3)
    a = h5["a"][...]   # same
    
    print("x shape:", x.shape)
    print("u shape:", u.shape)
    print("a shape:", a.shape)

    write_pointcloud_vtu(o_vtu_vel, x, u, "velocity")
    write_pointcloud_vtu(o_vtu_acc, x, a, "acceleration")