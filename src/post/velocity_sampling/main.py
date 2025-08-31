import os
import h5py
import numpy as np

i_dir = "../../example/3D/navier_stokes/outputs/aneurysm_A/"
o_dir = "output/aneurysm_A"
os.makedirs(o_dir, exist_ok=True)

i_h5   = i_dir + "/solution.h5"
o_h5   = o_dir + "/solution.h5"
o_xdmf = o_dir + "/solution.xdmf"

start = 120
skip = 5 
num_phases = 8   

with h5py.File(i_h5, "r") as f_in, h5py.File(o_h5, "w") as f_out:
    # copy nodes and cells
    f_out.create_dataset("nodes", data=f_in["/nodes"][()], compression="gzip")
    f_out.create_dataset("cells", data=f_in["/cells"][()], compression="gzip")

    step_keys = sorted([k for k in f_in.keys() if k.startswith("step")])
    selected_keys = step_keys[start::skip][:num_phases]

    for i, k in enumerate(selected_keys):
        g = f_out.create_group(f"step{i:08d}")
        # --- velocity ---
        g.create_dataset("velocity", data=f_in[f"{k}/velocity"][()], compression="gzip")
        # --- pressure ---
        g.create_dataset("pressure", data=f_in[f"{k}/pressure"][()], compression="gzip")

def write_xdmf(xdmf_path, h5_name, n_nodes, n_cells, num_phases, dt=1.0):
    with open(xdmf_path, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
        f.write('<Xdmf Version="2.0">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="CellTime" GridType="Collection" CollectionType="Temporal">\n')

        for i in range(num_phases):
            time_val = i * dt
            f.write(f'      <Grid Name="mesh" GridType="Uniform">\n')
            f.write(f'        <Time Value="{time_val}"/>\n')
            f.write('        <Geometry GeometryType="XYZ">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} 3" NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/nodes\n')
            f.write('          </DataItem>\n')
            f.write('        </Geometry>\n')
            f.write(f'        <Topology TopologyType="Tetrahedron" NumberOfElements="{n_cells}">\n')
            f.write(f'          <DataItem Dimensions="{n_cells} 4" NumberType="UInt" Format="HDF">\n')
            f.write(f'            {h5_name}:/cells\n')
            f.write('          </DataItem>\n')
            f.write('        </Topology>\n')

            # --- Velocity ---
            f.write('        <Attribute Name="velocity" AttributeType="Vector" Center="Node">\n')
            f.write(f'          <DataItem Dimensions="{n_nodes} 3" NumberType="Float" Precision="8" Format="HDF">\n')
            f.write(f'            {h5_name}:/step{i:08d}/velocity\n')
            f.write('          </DataItem>\n')
            f.write('        </Attribute>\n')

            # --- Pressure ---
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
    n_nodes = f["nodes"].shape[0]
    n_cells = f["cells"].shape[0]

write_xdmf(o_xdmf, os.path.basename(o_h5), n_nodes, n_cells, num_phases, dt=skip)