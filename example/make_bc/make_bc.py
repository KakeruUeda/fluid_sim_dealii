import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import csv
import h5py
import yaml
from collections import OrderedDict

i_name = "aneurysmA"

# --- input info ---
i_dir = "inputs/2D/" + i_name
i_config = i_dir + "/config.yaml"
with open(i_config, "r") as f:
    config = yaml.safe_load(f)
# -------------------

# --- output info ---
o_dir = "outputs/2D/" + i_name
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

wave_form = False
for i in range(n_bc):
    if profile_time[i] == "wave":
        wave_form = True

if wave_form:      
    i_bc = i_dir + "/inlet_fourier.csv"
    with open(i_bc) as f:
        reader = csv.reader(f)
        data_bc = [[float(x) for x in row] for row in reader]

    data_bc = np.array(data_bc)
    t_obs   = data_bc[:, 0]
    vel_obs = data_bc[:, 1]

# time params
dt_cfd = float(config["time"]["dt_cfd"])
t_end = float(config["time"]["t_end"])

# normal
normal = config["normal"]

# parabolic profile params 
if "parabolic" in config:
    radius = config["parabolic"]["radius"]
    center = config["parabolic"]["center"]
else:
    radius = None
    center = None

eps = 1e-12
t_cfd = np.arange(-eps, t_end + eps, dt_cfd)

if wave_form:
    spline = CubicSpline(
        t_obs, vel_obs, 
        bc_type="natural", 
        extrapolate=True
    )
    
    # --- interpolate onto cfd resolution ---
    vel_cfd = spline(t_cfd)
    print(len(vel_cfd))
    
    # --- check interpolation ----
    if np.isfinite(vel_cfd).all():
        print("Interpolation successful")
    else:
        raise ValueError("Interpolation failed")

values_bc = []
for i in range(n_bc):
    if profile_time[i] == "uniform":
        d = int(dirs[i])
        vec = np.full_like(t_cfd, float(values[i]), dtype=float)  
        vec = vec * normal[d]  
    elif profile_time[i] == "wave":
        d = int(dirs[i])
        vec = vel_cfd * normal[d]  
    else:
        raise ValueError("Unknown profile_time")
    values_bc.append(vec)

BCS = []
for i in range(n_bc):
    BCS.append({
        "id": ids[i],
        "dir": dirs[i],
        "profile": profile_space[i],
        "value": values_bc[i],
    })

grouped = OrderedDict()
for bc in BCS:
    grouped.setdefault(bc["id"], []).append(bc)

with h5py.File(o_h5, "w") as f:
    f.create_dataset("time", data=t_cfd)

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
                p.create_dataset("radius", data=np.array(radius, dtype=np.float64))
                p.create_dataset("center", data=np.array(center, dtype=np.float64))
            elif item["profile"] == "uniform":
                g_dir.create_group("uniform")
            else:
                raise ValueError(f"Unknown profile: {item['profile']}")

if wave_form:
    plt.figure(figsize=(7, 4))
    dim = len(normal)
    for i, comp in enumerate(["x", "y", "z"][:dim]):
        plt.plot(t_cfd, vel_cfd * normal[i], label=f"vel_cfd_{comp} (spline)", linewidth=2)
        plt.scatter(t_obs, vel_obs * normal[i], s=15, label=f"obs·n{comp}", alpha=0.6)
    plt.xlabel("t [s]")
    plt.ylabel("velocity [m/s]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(o_dir + "/velocity_component.png")

    plt.figure(figsize=(7, 4))
    dim = len(normal)
    vel_cfd_mag = np.sqrt(sum((vel_cfd * normal[i])**2 for i in range(dim)))
    vel_obs_mag = np.sqrt(sum((vel_obs * normal[i])**2 for i in range(dim)))
    plt.plot(t_cfd, vel_cfd_mag, color="blue", linewidth=2)
    #plt.scatter(t_obs, vel_obs_mag, c="k", s=15, label="obs·n_mag", alpha=0.6)
    plt.xlabel("t [s]")
    plt.ylabel("Velocity (mag) [m/s]")
    plt.grid(True, alpha=0.3)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(o_dir + "/velocity_magnitude.png")

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")

with h5py.File(o_h5, "r") as f:
    print("\n--- HDF5 file structure ===")
    f.visititems(print_hdf5_structure)
