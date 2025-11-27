import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
import xml.etree.ElementTree as et
import scienceplots
import yaml
import h5py
import meshio

i_config = "config.yaml"
with open(i_config, "r") as f:
    config = yaml.safe_load(f)

dim   = config["base"]["dim"]
i_dir = config["io"]["i_dir"]
o_dir = config["io"]["o_dir"] 
os.makedirs(o_dir, exist_ok=True)

i_h5   = i_dir + "/solution.h5"
i_xdmf = i_dir + "/solution.xdmf"
i_msh  = i_dir + "/mesh.msh"
o_h5   = o_dir + "/solution.h5"
o_xdmf = o_dir + "/solution.xdmf"

# --- read target mesh (reference) ---
mesh = meshio.read(i_msh)
x_ref = mesh.points[:, :dim]

def extract_cells_ref(mesh, dim):
    want = "tetra" if dim == 3 else "triangle"
    for cb in mesh.cells:
        if cb.type == want:
            return cb.data
    return mesh.cells[0].data

cells_ref = extract_cells_ref(mesh, dim).astype(np.uint32)

# --- read physical fields from .h5 ---
def open_h5(path):
    with h5py.File(path, "r") as f:
        x_src = f["/nodes"][()]                
        cells_src = f["/cells"][()]            
        step_keys = sorted([k for k in f.keys() if k.startswith("step")])
        vel = [f[f"{k}/velocity"][()] for k in step_keys]
        pre = [f[f"{k}/pressure"][()] for k in step_keys]
        vel = np.array(vel)  # (n_steps, n_points, dim or 3)
        pre = np.array(pre)  # (n_steps, n_points) or (n_steps, n_points, 1)
        return x_src, cells_src, vel, pre

x_cfd, cells_cfd, vel_cfd, pre_cfd = open_h5(i_h5)
n_steps = vel_cfd.shape[0]

x_cfd   = x_cfd[:, :dim]
vel_cfd = vel_cfd[:, :, :dim]   # (n_steps, n_points_src, dim)

# --- get time values from xdmf file ---
tree = et.parse(i_xdmf)
root = tree.getroot()
time_values = [float(t.attrib["Value"]) for t in root.iter("Time")]
time = np.array(time_values, dtype=float)

def interpolate_vel_series(x_src, vel_src, x_tgt):
    vel_tgt = np.zeros((n_steps, x_tgt.shape[0], dim), dtype=vel_src.dtype)
    tree = cKDTree(x_src)
    _, idx_nearest = tree.query(x_tgt)
    for t in range(n_steps):
        for d in range(dim):
            print(f"---Interpolating Velocity .. step {t+1}/{n_steps}, dim={d+1}/{dim}")
            interp = LinearNDInterpolator(x_src, vel_src[t, :, d], fill_value=np.nan)
            v = interp(x_tgt)
            nan_mask = np.isnan(v)
            v[nan_mask] = vel_src[t, idx_nearest[nan_mask], d]
            vel_tgt[t, :, d] = v
    return vel_tgt


def interpolate_scalar_series(x_src, s_src, x_tgt):
    s_tgt = np.zeros((n_steps, x_tgt.shape[0]), dtype=s_src.dtype)
    tree = cKDTree(x_src)
    _, idx_nearest = tree.query(x_tgt)
    for t in range(n_steps):
        print(f"---Interpolating Pressure .. step {t+1}/{n_steps}")
        interp = LinearNDInterpolator(x_src, s_src[t, :], fill_value=np.nan)
        v = interp(x_tgt)
        nan_mask = np.isnan(v)
        v[nan_mask] = s_src[t, idx_nearest[nan_mask]]
        s_tgt[t, :] = v
    return s_tgt
    
vel_ref = interpolate_vel_series(x_cfd, vel_cfd, x_ref)

pre_cfd = np.squeeze(pre_cfd)  # (n_steps, n_points[, 1]) -> (n_steps, n_points)
pre_ref = interpolate_scalar_series(x_cfd, pre_cfd, x_ref)  # (n_steps, n_nodes_ref)

if dim == 2:
    vel_add = np.zeros_like(vel_ref[..., :1])  
    vel_ref = np.concatenate([vel_ref, vel_add], axis=-1)

# --- write output H5 ---
with h5py.File(o_h5, "w") as f_out:
    f_out.create_dataset("nodes", data=x_ref, compression="gzip")         
    f_out.create_dataset("cells", data=cells_ref, compression="gzip")     

    for i in range(n_steps):
        g = f_out.create_group(f"step{i:08d}")
        g.create_dataset("velocity", data=vel_ref[i], compression="gzip")           
        g.create_dataset("pressure", data=pre_ref[i, :, None], compression="gzip") 

def write_xdmf(xdmf_path, h5_name, n_nodes, n_cells, dim):
    with open(xdmf_path, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write('<Xdmf Version="2.0">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">\n')

        for i in range(n_steps):
            time_val = time[i]
            f.write('      <Grid Name="mesh" GridType="Uniform">\n')
            f.write(f'        <Time Value="{time_val}"/>\n')

            geom_type = "XYZ" if dim == 3 else "XY"
            f.write(f'        <Geometry GeometryType="{geom_type}">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} {dim}" '
                    f'NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/nodes\n')
            f.write('          </DataItem>\n')
            f.write('        </Geometry>\n')

            if dim == 3:
                topo_type, verts = "Tetrahedron", 4
            else:
                topo_type, verts = "Triangle", 3

            f.write(f'        <Topology TopologyType="{topo_type}" NumberOfElements="{n_cells}">\n')
            f.write(f'          <DataItem Dimensions="{n_cells} {verts}" NumberType="UInt" Format="HDF">\n')
            f.write(f'            {h5_name}:/cells\n')
            f.write('          </DataItem>\n')
            f.write('        </Topology>\n')

            f.write('        <Attribute Name="velocity" AttributeType="Vector" Center="Node">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} 3" NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/step{i:08d}/velocity\n')
            f.write('          </DataItem>\n')
            f.write('        </Attribute>\n')

            f.write('        <Attribute Name="pressure" AttributeType="Scalar" Center="Node">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} 1" NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/step{i:08d}/pressure\n')
            f.write('          </DataItem>\n')
            f.write('        </Attribute>\n')

            f.write('      </Grid>\n')

        f.write('    </Grid>\n')
        f.write('  </Domain>\n')
        f.write('</Xdmf>\n')

with h5py.File(o_h5, "r") as f:
    n_nodes_ref = f["nodes"].shape[0]
    n_cells_ref = f["cells"].shape[0]

write_xdmf(o_xdmf, os.path.basename(o_h5), n_nodes_ref, n_cells_ref, dim=dim)
