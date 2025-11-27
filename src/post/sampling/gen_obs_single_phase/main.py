import os
import h5py
import numpy as np
import yaml
from scipy.interpolate import CubicSpline
import xml.etree.ElementTree as et

# --- config ---
i_config = "config.yaml"
with open(i_config, "r") as f:
    config = yaml.safe_load(f)

dim   = config["base"]["dim"]
i_dir = config["io"]["i_dir"]
o_dir = config["io"]["o_dir"]

t_sample = config["sampling"]["t_sample"]
step     = config["sampling"]["step"]
# --------------

os.makedirs(o_dir, exist_ok=True)

i_xdmf = i_dir + "/solution.xdmf"
i_h5   = i_dir + "/solution.h5"
o_h5   = o_dir + "/solution.h5"
o_xdmf = o_dir + "/solution.xdmf"

# --- get time values from xdmf file ---
tree = et.parse(i_xdmf)
root = tree.getroot()
time_values = []
for time_tag in root.iter("Time"):
    val = float(time_tag.attrib["Value"])
    time_values.append(val)

time = np.array(time_values, dtype=float)

def show_h5_structure(path):
    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")
    with h5py.File(path, "r") as f:
        f.visititems(visit)

# --- read velocity fields ---
with h5py.File(i_h5, "r") as f:
    x_cfd = f["/nodes"][()]   
    step_keys = sorted([k for k in f.keys() if k.startswith("step")])

    vel_t_cfd = [f[f"{k}/velocity"][()] for k in step_keys]
    vel_t_cfd = np.array(vel_t_cfd)  

n_step, n_points, _ = vel_t_cfd.shape
accel = np.zeros((n_points, 3))

print("Interpolating ...")
for p in range(n_points):
    for d in range(3):
        if dim == 2 and d == 3:
            continue
        vel_series = vel_t_cfd[:, p, d]
        spline = CubicSpline(time, vel_series, bc_type="natural", extrapolate=False)
        accel[p, d] = spline(t_sample, 1) 

with h5py.File(i_h5, "r") as f_in, h5py.File(o_h5, "w") as f_out:
    # copy nodes and cells
    f_out.create_dataset("nodes", data=f_in["/nodes"][()], compression="gzip")
    f_out.create_dataset("cells", data=f_in["/cells"][()], compression="gzip")

    step_keys = sorted([k for k in f_in.keys() if k.startswith("step")])
    k = step_keys[step] 

    g = f_out.create_group(k)
    # --- velocity ---
    g.create_dataset("velocity", data=f_in[f"{k}/velocity"][()], compression="gzip")
    # --- pressure ---
    g.create_dataset("pressure", data=f_in[f"{k}/pressure"][()], compression="gzip")
    # --- acceleration ---
    g.create_dataset("acceleration", data=accel, compression="gzip")

def write_xdmf(xdmf_path, h5_name, n_nodes, n_cells, step_name, dim=3):
    with open(xdmf_path, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write('<Xdmf Version="2.0">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="mesh" GridType="Uniform">\n')

        f.write(f'      <Time Value="0"/>\n')

        # Geometry
        geom_type = "XYZ" if dim == 3 else "XY"
        f.write(f'      <Geometry GeometryType="{geom_type}">\n')
        f.write(f'        <DataItem Dimensions="{n_nodes} {dim}" '
                f'NumberType="Float" Precision="8" Format="HDF">\n')
        f.write(f'          {h5_name}:/nodes\n')
        f.write('        </DataItem>\n')
        f.write('      </Geometry>\n')

        # Topology
        if dim == 3:
            topo_type, verts = "Tetrahedron", 4
        else:
            topo_type, verts = "Triangle", 3

        f.write(f'      <Topology TopologyType="{topo_type}" NumberOfElements="{n_cells}">\n')
        f.write(f'        <DataItem Dimensions="{n_cells} {verts}" '
                f'NumberType="UInt" Format="HDF">\n')
        f.write(f'          {h5_name}:/cells\n')
        f.write('        </DataItem>\n')
        f.write('      </Topology>\n')

        # Attributes
        f.write('      <Attribute Name="velocity" AttributeType="Vector" Center="Node">\n')
        f.write(f'        <DataItem Dimensions="{n_nodes} 3" '
                f'NumberType="Float" Precision="8" Format="HDF">\n')
        f.write(f'          {h5_name}:/{step_name}/velocity\n')
        f.write('        </DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Name="pressure" AttributeType="Scalar" Center="Node">\n')
        f.write(f'        <DataItem Dimensions="{n_nodes} 1" '
                f'NumberType="Float" Precision="8" Format="HDF">\n')
        f.write(f'          {h5_name}:/{step_name}/pressure\n')
        f.write('        </DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Name="acceleration" AttributeType="Vector" Center="Node">\n')
        f.write(f'        <DataItem Dimensions="{n_nodes} 3" '
                f'NumberType="Float" Precision="8" Format="HDF">\n')
        f.write(f'          {h5_name}:/{step_name}/acceleration\n')
        f.write('        </DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('    </Grid>\n')
        f.write('  </Domain>\n')
        f.write('</Xdmf>\n')

with h5py.File(o_h5, "r") as f:
    n_nodes = f["nodes"].shape[0]
    n_cells = f["cells"].shape[0]

write_xdmf(o_xdmf, os.path.basename(o_h5), n_nodes, n_cells, k, dim=dim)