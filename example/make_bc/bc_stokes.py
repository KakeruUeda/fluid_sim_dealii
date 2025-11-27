import numpy as np
import os
import h5py
from collections import OrderedDict

# ---- parabolic profile params ----
radius = 0.5                  
center = [-1.0, 1.0, 1.0]

# ---- output settings ---
output_dir = "output_stokes"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "bc.h5")

t_values = np.array([0.0])
value_bc0 = np.full_like(t_values, 1.0)
value_bc1 = np.full_like(t_values, 0.0)
value_bc2 = np.full_like(t_values, 0.0)
value_bc3 = np.full_like(t_values, 0.0)

BCS = [
    dict(id=6, dir=0, profile="uniform", value=value_bc0),
    dict(id=6, dir=1, profile="uniform", value=value_bc1),
    dict(id=7, dir=0, profile="uniform", value=value_bc2),
    dict(id=7, dir=1, profile="uniform", value=value_bc3),
]

grouped = OrderedDict()
for bc in BCS:
    grouped.setdefault(bc["id"], []).append(bc)

with h5py.File(output_file, "w") as f:
    f.create_dataset("time", data=t_values)

    for bc_idx, (bc_id, items) in enumerate(grouped.items()):
        g_bc = f.create_group(f"bc{bc_idx}")
        g_bc.create_dataset("id", data=np.array(bc_id, dtype=np.int32))

        for item in sorted(items, key=lambda x: x["dir"]):
            d = int(item["dir"])
            g_dir = g_bc.create_group(f"dir{d}")   
            g_dir.create_dataset("dir", data=np.array(d, dtype=np.int32))
            g_dir.create_dataset("value", data=item["value"])
            g_dir.attrs["profile"] = item["profile"]

            if item["profile"] == "parabolic":
                p = g_dir.create_group("parabolic")
                p.create_dataset("radius", data=np.array(item["radius"], dtype=np.float64))
                p.create_dataset("center", data=np.array(item["center"], dtype=np.float64))
            elif item["profile"] == "uniform":
                g_dir.create_group("uniform")
            else:
                raise ValueError(f"Unknown profile: {item['profile']}")

print(f"HDF5 written: {output_file}")

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")

with h5py.File(output_file, "r") as f:
    print("\n--- HDF5 file structure ===")
    f.visititems(print_hdf5_structure)