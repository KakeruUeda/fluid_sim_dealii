import numpy as np
import os
import h5py
import yaml
from collections import OrderedDict

# --- input info ---
i_dir = "inputs/2D/bend_tube"
i_config = i_dir + "/config.yaml"
with open(i_config, "r") as f:
    config = yaml.safe_load(f)
# --------------------

# --- output info ---
o_dir = "outputs/bend_tube"
o_h5 = o_dir + "/bc.h5"
os.makedirs(o_dir, exist_ok=True)
# --------------------

# boundary settings
boundaries = config["boundary"]
n_bc = len(boundaries)  
ids, names, profile_space, profile_time, dirs, values = [], [], [], [], [], []

for bd in boundaries:
    ids.append(bd["id"])
    names.append(bd["name"])
    profile_space.append(bd["profile_space"])
    profile_time.append(bd["profile_time"]) 
    dirs.append(bd["dir"])
    values.append(bd["value"])

# time params
dt    = float(config["time"]["dt"])
t_end = float(config["time"]["t_end"])
n_steps = int(t_end / dt) + 1
t_values = np.linspace(0.0, t_end, n_steps)

# waveform
a = float(config["waveform"]["a"])
b = float(config["waveform"]["b"])
T = float(config["waveform"]["T"])

# parabolic profile params 
if "parabolic" in config:
    radius = config["parabolic"]["radius"]
    center = config["parabolic"]["center"]
else:
    radius = None
    center = None

values_series = []
for i in range(n_bc):
    if profile_time[i] == "uniform":
        vec = np.full_like(t_values, float(values[i]), dtype=float)  
    elif profile_time[i] == "wave":
        vec = a * np.sin(2.0 * np.pi * t_values / float(T)) + b
    else:
        raise ValueError("Unknown profile_time")
    values_series.append(vec)

BCS = []
for i in range(n_bc):
    BCS.append({
        "id": ids[i],
        "dir": dirs[i],
        "profile": profile_space[i],
        "value": values_series[i],
    })

grouped = OrderedDict()
for bc in BCS:
    grouped.setdefault(bc["id"], []).append(bc)

with h5py.File(o_h5, "w") as f:
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
                if radius is None or center is None:
                    raise ValueError("profile=parabolic needs radius/center")
                p = g_dir.create_group("parabolic")
                p.create_dataset("radius", data=np.array(radius, dtype=np.float64))
                p.create_dataset("center", data=np.array(center, dtype=np.float64))
            elif item["profile"] == "uniform":
                g_dir.create_group("uniform")
            else:
                raise ValueError(f"Unknown profile: {item['profile']}")

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")

with h5py.File(o_h5, "r") as f:
    print("\n--- HDF5 file structure ===")
    f.visititems(print_hdf5_structure)
