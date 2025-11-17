import numpy as np

def sine_controller(t, num_joints, amplitude=0.5, frequency=1.0):
    phase_offsets_base = [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]
    phase_offsets = [
        phase_offsets_base[j % len(phase_offsets_base)]
        for j in range(num_joints)
    ]

    targets = [
        amplitude * np.sin(2.0 * np.pi * frequency * t + phase_offsets[j])
        for j in range(num_joints)
    ]
    return np.asarray(targets, dtype=float)