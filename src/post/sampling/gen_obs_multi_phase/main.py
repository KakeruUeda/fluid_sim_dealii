import os
import h5py
import numpy as np
import xml.etree.ElementTree as et
import yaml

# --- config ---
i_config = "config.yaml"
with open(i_config, "r") as f:
    config = yaml.safe_load(f)

dim     = config["base"]["dim"]
i_dir   = config["io"]["i_dir"]
o_dir   = config["io"]["o_dir"]
start   = config["sampling"]["start"]
skip    = config["sampling"]["skip"]
n_phase = config["sampling"]["n_phase"]
# --------------

# --- make directory ----
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
time_skipped = time[start::skip][:n_phase]

time_skipped = time[start::skip][:n_phase]
time_skipped = time_skipped - time_skipped[0] 

print(time_skipped)
time_skipped = np.round(time_skipped, 6)

if len(time_skipped) != n_phase:
    raise ValueError(
        f"len(time_skipped) doesn't match with n_phase"
    )

def print_h5_structure(path):
    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"[Group] {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")
    with h5py.File(path, "r") as f:
        f.visititems(visit)

# print_h5_structure(i_h5)

with h5py.File(i_h5, "r") as f_in, h5py.File(o_h5, "w") as f_out:
    # copy nodes and cells
    f_out.create_dataset("nodes", data=f_in["/nodes"][()], compression="gzip")
    f_out.create_dataset("cells", data=f_in["/cells"][()], compression="gzip")

    step_keys = sorted([k for k in f_in.keys() if k.startswith("step")])
    selected_keys = step_keys[start::skip][:n_phase]

    for i, k in enumerate(selected_keys):
        g = f_out.create_group(f"step{i:08d}")
        # --- velocity ---
        g.create_dataset("velocity", data=f_in[f"{k}/velocity"][()], compression="gzip")
        # --- pressure ---
        g.create_dataset("pressure", data=f_in[f"{k}/pressure"][()], compression="gzip")


def write_xdmf(xdmf_path, h5_name, n_nodes, n_cells, n_phase, dt=1.0, dim=3):
    with open(xdmf_path, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write('<Xdmf Version="2.0">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">\n')

        for i in range(n_phase):
            time_val = time_skipped[i]
            f.write(f'      <Grid Name="mesh" GridType="Uniform">\n')
            f.write(f'        <Time Value="{time_val}"/>\n')

            # Geometry
            geom_type = "XYZ" if dim == 3 else "XY"
            f.write(f'        <Geometry GeometryType="{geom_type}">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} {dim}" '
                    f'NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/nodes\n')
            f.write('          </DataItem>\n')
            f.write('        </Geometry>\n')

            # Topology
            if dim == 3:
                topo_type, verts = "Tetrahedron", 4
            else:
                topo_type, verts = "Triangle", 3

            f.write(f'        <Topology TopologyType="{topo_type}" '
                    f'NumberOfElements="{n_cells}">\n')
            f.write(f'          <DataItem Dimensions="{n_cells} {verts}" '
                    f'NumberType="UInt" Format="HDF">\n')
            f.write(f'            {h5_name}:/cells\n')
            f.write('          </DataItem>\n')
            f.write('        </Topology>\n')

            # Attributes (Velocity)
            vel_dim = 3
            f.write('        <Attribute Name="velocity" AttributeType="Vector" Center="Node">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} 3" '
                    f'NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/step{i:08d}/velocity\n')
            f.write('          </DataItem>\n')
            f.write('        </Attribute>\n')

            # Attributes (Pressure)
            f.write('        <Attribute Name="pressure" AttributeType="Scalar" Center="Node">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} 1" '
                    f'NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/step{i:08d}/pressure\n')
            f.write('          </DataItem>\n')
            f.write('        </Attribute>\n')

            f.write('      </Grid>\n')

        f.write('    </Grid>\n')
        f.write('  </Domain>\n')
        f.write('</Xdmf>\n')


with h5py.File(o_h5, "r") as f:
    n_nodes = f["nodes"].shape[0]
    n_cells = f["cells"].shape[0]

write_xdmf(o_xdmf, os.path.basename(o_h5), n_nodes, n_cells, n_phase, dt=skip, dim=dim)
