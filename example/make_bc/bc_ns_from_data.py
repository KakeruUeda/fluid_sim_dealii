import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import h5py
from collections import OrderedDict

i_name = "aneurysm_A"
i_dir = "inputs/" + i_name
i_config = i_dir + "/config.dat"

o_dir = "outputs/" + i_name
os.makedirs(o_dir, exist_ok=True)
o_file = o_dir + "/bc.h5"

with open(i_config) as f:
    values = [
        float(parts[0]) if len(parts := line.split()) == 1
        else [float(x) for x in parts]
        for line in f if not line.startswith("#") and line.strip()
    ]

# --- parameters ---
n_phases   = int(values[0])
dt_obs     = float(values[1])
dt_cfd     = float(values[2])
t_end      = float(values[3])
v_inlet    = values[4]  # list
normal     = values[5]  # list
id_inlet   = int(values[6])
id_wall    = int(values[7])

vel_obs = np.asarray(v_inlet, dtype=float)
t_obs = np.arange(len(vel_obs))*dt_obs
t_cfd = np.arange(0.0, t_end + 1e-12, dt_cfd)

print(len(t_obs), len(t_cfd))
# print(t_cfd)

spline = CubicSpline(
    t_obs, vel_obs, 
    bc_type="natural", 
    extrapolate=True
)

# --- interpolate onto cfd resolution
vel_cfd = spline(t_cfd) 
print(len(vel_cfd))

# --- check interpolation ----
if np.isfinite(vel_cfd).all():
    print("Interpolation successful")
else:
    print("Interpolation failed")
    exit()

values_bc0 = vel_cfd*normal[0]
values_bc1 = vel_cfd*normal[1]
values_bc2 = vel_cfd*normal[2]
values_bc3 = np.full_like(vel_cfd, 0.0)
values_bc4 = np.full_like(vel_cfd, 0.0)
values_bc5 = np.full_like(vel_cfd, 0.0)

BCS = [
    dict(id=id_inlet, dir=0, profile="uniform", value=values_bc0),
    dict(id=id_inlet, dir=1, profile="uniform", value=values_bc1),
    dict(id=id_inlet, dir=2, profile="uniform", value=values_bc2),
    dict(id=id_wall, dir=0, profile="uniform", value=values_bc3),
    dict(id=id_wall, dir=1, profile="uniform", value=values_bc4),
    dict(id=id_wall, dir=2, profile="uniform", value=values_bc5),
]

grouped = OrderedDict()
for bc in BCS:
    grouped.setdefault(bc["id"], []).append(bc)

with h5py.File(o_file, "w") as f:
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
                p.create_dataset("radius", data=np.array(item["radius"], dtype=np.float64))
                p.create_dataset("center", data=np.array(item["center"], dtype=np.float64))
            elif item["profile"] == "uniform":
                g_dir.create_group("uniform")
            else:
                raise ValueError(f"Unknown profile: {item['profile']}")

plt.figure(figsize=(7, 4))
plt.plot(t_cfd, vel_cfd, label="vel_cfd (spline)", linewidth=2)
plt.scatter(t_obs, vel_obs, s=20, label="vel_obs (original)", color="red")  

plt.xlabel("t [s]")
plt.ylabel("velocity [m/s]")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

fig_path = o_dir + "/check_interpolation.png"
plt.savefig(fig_path, dpi=300)

# print(f"Saved figure to {fig_path}")
# def print_hdf5_structure(name, obj):
#     if isinstance(obj, h5py.Group):
#         print(f"[Group] {name}")
#     elif isinstance(obj, h5py.Dataset):
#         print(f"  [Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}")

# with h5py.File(o_file, "r") as f:
#     print("\n--- HDF5 file structure ===")
#     f.visititems(print_hdf5_structure)
