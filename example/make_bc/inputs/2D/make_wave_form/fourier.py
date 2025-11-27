import csv
import numpy as np
import matplotlib.pyplot as plt

# --- parameters ---
T = 1.0                   # period [s]
omega = 2 * np.pi / T     # angular frequency
ubar = 0.2                # mean velocity [m/s]

# amplitudes
A1 = 0.15
A2 = 0.05
A3 = 0.01

# phases
phi1, phi2, phi3 = 0.0, np.pi/2, 0.0

# --- time vector ---
t = np.linspace(0, 4*T, 1000)  # plot 2 cycles

# --- waveform ---
u = (ubar
     + A1 * np.cos(omega*t + phi1)
     + A2 * np.cos(2*omega*t + phi2)
     + A3 * np.cos(3*omega*t + phi3))

# --- save to CSV using csv.writer ---
with open("inlet_fourier.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for ti, ui in zip(t, u):
        writer.writerow([ti, ui])

# --- plot ---
plt.figure(figsize=(7,4))
plt.plot(t, u, label="u(t)")
plt.axhline(ubar, color="gray", linestyle="--", label="mean")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Inlet velocity (4th-order Fourier series)")
plt.legend()
plt.grid(True)
plt.savefig("fourier.png")
plt.show()
