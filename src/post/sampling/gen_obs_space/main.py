import os
import h5py
import numpy as np
import meshio
from scipy.interpolate import griddata, LinearNDInterpolator, CubicSpline
import xml.etree.ElementTree as et
import yaml

i_config = "config.yaml"
with open(i_config, "r") as f:
    config = yaml.safe_load(f)

dim         = config["base"]["dim"]
i_name      = config["io"]["i_name"]
o_name      = config["io"]["o_name"]
nx          = config["grid"]["nx"]
ny          = config["grid"]["ny"]
step_sample = config["sample"]["step"]

i_dir = "inputs/" + i_name
o_dir = "outputs/" + o_name
os.makedirs(o_dir, exist_ok=True)

i_h5   = i_dir + "/solution.h5"
i_xdmf = i_dir + "/solution.xdmf"

o_h5   = o_dir + "/data_obs.h5"

# --- get time values from xdmf file ---
tree = et.parse(i_xdmf)
root = tree.getroot()
time_values = []
for time_tag in root.iter("Time"):
    val = float(time_tag.attrib["Value"])
    time_values.append(val)

time = np.array(time_values, dtype=float)

# --- read velocity fields from cfd resutls (already cropped in time) ---
with h5py.File(i_h5, "r") as f:
    x_cfd = f["/nodes"][()]   
    step_keys = sorted([k for k in f.keys() if k.startswith("step")])

    vel_t_cfd = []
    for k in step_keys:
        vel = f[f"{k}/velocity"][()] 
        vel_t_cfd.append(vel)

    vel_t_cfd = np.array(vel_t_cfd)

print("x_cfd shape:", x_cfd.shape)
print("velocities shape:", vel_t_cfd.shape)
x_cfd = x_cfd[:, :dim]

# --- build a regular grid for interpolation (2D) ---
pad = 0     

x_min, y_min = x_cfd.min(axis=0)
x_max, y_max = x_cfd.max(axis=0)
dx = (x_max - x_min) * pad
dy = (y_max - y_min) * pad

xs = np.linspace(x_min - dx, x_max + dx, nx)
ys = np.linspace(y_min - dy, y_max + dy, ny)
xx, yy = np.meshgrid(xs, ys)

x_obs = np.c_[xx.ravel(), yy.ravel()]   

# --- linear interpolation ---
n_steps = vel_t_cfd.shape[0]

# initialize
vel_t_cfd_on_obs = np.zeros(
    (n_steps, x_obs.shape[0], 3), 
    dtype=np.float64
)

if len(time) != vel_t_cfd_on_obs.shape[0]:
    raise ValueError(
        f"len(time) doesn't match with vel_t_cfd_on_obs.shape[0]"
    )

for step in range(n_steps):
    for d in range(dim):
        print("Interpolating... time step =", step, " dim =", d)
        vals = griddata(
            points=x_cfd,
            values=vel_t_cfd[step, :, d],
            xi=x_obs,
            method="linear"
        )
        nanmask = np.isnan(vals)
        if nanmask.any():
            vals[nanmask] = 0.0
        vel_t_cfd_on_obs[step, :, d] = vals

mag = np.linalg.norm(vel_t_cfd_on_obs[:, :, :dim], axis=2)
max_mag = np.max(mag, axis=0)
max_mag_all = np.max(max_mag)
thres = max_mag_all * 1e-8

valid_mask = max_mag > thres
x_obs = x_obs[valid_mask]
vel_t_cfd_on_obs = vel_t_cfd_on_obs[:, valid_mask, :]

# --- save velocity series to vtu file ---
o_dir_vtu = o_dir + "/vtu_series"
os.makedirs(o_dir_vtu, exist_ok=True)
n_steps, n_points, _ = vel_t_cfd_on_obs.shape
points = np.c_[x_obs, np.zeros((x_obs.shape[0],), dtype=np.float64)]

cells = [("vertex", np.arange(n_points, dtype=np.int32).reshape(-1, 1))]

vtu_files = []
for k in range(n_steps):
    point_data = {
        "velocity":     vel_t_cfd_on_obs[k],  
    }
    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    vtu_path = os.path.join(o_dir_vtu, f"fields_{k:04d}.vtu")
    meshio.write(vtu_path, mesh)  
    vtu_files.append(vtu_path)


# --- estimate accelaration ---
n_points = vel_t_cfd_on_obs.shape[1]
accel = np.zeros_like(vel_t_cfd_on_obs)

for p in range(n_points):
    for d in range(dim):  
        vel_series = vel_t_cfd_on_obs[:, p, d]                
        spline = CubicSpline(time, vel_series, bc_type="natural", extrapolate=False)
        accel[:, p, d] = spline(time, 1)  
    if dim == 2:
        accel[:, p, 2] = 0.0   

mag = np.linalg.norm(vel_t_cfd_on_obs[step_sample, :, :dim], axis=1)
mean = np.mean(mag)
print("mean: ", mean)
with h5py.File(o_h5, "w") as h5:
    h5.create_dataset("x", data=x_obs, compression="gzip")
    h5.create_dataset("u", data=vel_t_cfd_on_obs[step_sample])
    h5.create_dataset("a", data=accel[step_sample])

# ------------------------------------------------------------
# ------------------ Visualization ---------------------------
# ------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

i_png       = o_dir + "/velocity_spline.png"
i_png_slope = o_dir + "/velocity_spline_with_slope.png"

# Select a sample point and direction for visualization               
sample_point = min(100, vel_t_cfd_on_obs.shape[1] - 1)  
sample_dir   = 0                     

# Extract the velocity time-series at the sample point
sample_vel_series = vel_t_cfd_on_obs[:, sample_point, sample_dir]
spline = CubicSpline(time, sample_vel_series, bc_type="natural", extrapolate=False)

t_refined = np.linspace(time[0], time[-1], 400)
spline_interp = spline(t_refined)

# Set global font style and size
plt.rcParams.update({
    "font.family": "DejaVu Sans",      # use Times New Roman
    "font.size": 14,                   # base font size
    "axes.labelsize": 16,              # axis labels
    "xtick.labelsize": 14,             # x-tick labels
    "ytick.labelsize": 14,             # y-tick labels
    "legend.fontsize": 14,             # legend text
    "figure.titlesize": 16             # figure title
})

# --- Figure 1: Sampled values vs spline interpolation ---
plt.figure(figsize=(6, 4))
plt.plot(
    time, sample_vel_series, 
    label='Observed', marker='x', 
    markersize=10, markeredgewidth=2.0,
    linestyle='None', color="blue"
)
plt.plot(t_refined, spline_interp, label='Spline', linestyle='-', color="gray")
plt.xlabel("Time [s]")
plt.ylabel("Velocity (x-dir) [m/s]")
plt.legend()
plt.tight_layout()
plt.savefig(i_png, dpi=300)   # dpi=300 for better quality
plt.close()

# --- Figure 2: Add tangent (slope) at a specific time ---
t_target = time[step_sample]

# Evaluate spline and derivative at the target time
v_target    = spline(t_target)        # velocity value
dvdt_target = spline(t_target, 1)     # acceleration

# Define a short time interval around t_target for the tangent line
dt_line = 0.1 * (time[-1] - time[0])
t_line = np.array([t_target - dt_line, t_target + dt_line])
v_line = v_target + dvdt_target * (t_line - t_target)

plt.figure(figsize=(6, 4))
plt.plot(
    time, sample_vel_series, 
    label='Observed', marker='x', 
    markersize=10, markeredgewidth=2.0,
    linestyle='None', color="blue"
)
plt.plot(t_refined, spline_interp, label='Spline', linestyle='-', color="gray")
plt.plot(t_line, v_line, '--', label=f'Slope', color="red")
plt.plot([t_target], [v_target], 'ro')  # mark the tangent point
plt.xlabel("Time [s]")
plt.ylabel("Velocity (x-dir) [m/s]")
plt.legend()
plt.tight_layout()
plt.savefig(i_png_slope, dpi=300)
plt.close()
# --------------------------------------------------------------------
