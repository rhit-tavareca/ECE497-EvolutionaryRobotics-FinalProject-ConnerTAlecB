import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters ---
T = 1.0           # seconds of simulation
dt = 0.01         # time step
t = np.arange(0, T, dt)

# --- Joint signal parameters ---
amplitude = 0.5   # radians
frequency = 1.0   # Hz
phase_offsets = [0, np.pi/2, np.pi, 3*np.pi/2]  # Front, Left, Right, Tail

# --- Generate the four signals ---
signals = []
for phi in phase_offsets:
    signals.append(amplitude * np.sin(2 * np.pi * frequency * t + phi))

# --- Plot setup ---
plt.figure(figsize=(10,6))
labels = ["Front Arm", "Left Arm", "Right Arm", "Tail"]
colors = ["red", "blue", "green", "purple"]

for i in range(4):
    plt.plot(t, signals[i], label=labels[i], color=colors[i], linewidth=2)

# --- Beautify the plot ---
plt.title("CrawlerBot Joint Control Signals")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (radians)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()
