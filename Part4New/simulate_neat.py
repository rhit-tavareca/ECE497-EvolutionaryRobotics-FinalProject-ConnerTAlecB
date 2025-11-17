import os
import time
import numpy as np
import matplotlib.pyplot as plt

import neat
import pybullet as p
import pyrosim.pyrosim as pyrosim
import pybullet_data

from demo_controller import sine_controller

physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -9.8)
p.setTimeStep(1.0 / 120.0)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("crawlerbot.urdf")
pyrosim.Prepare_To_Simulate(robot_id)

num_joints = p.getNumJoints(robot_id)
for link_i in range(-1, num_joints):
    p.changeDynamics(
        robot_id,
        link_i,
        lateralFriction=1.4,
        rollingFriction=0.01,
        spinningFriction=0.01,
    )

joint_names = [
    p.getJointInfo(robot_id, j)[1].decode()
    for j in range(num_joints)
    if p.getJointInfo(robot_id, j)[2] == p.JOINT_REVOLUTE
]

motor_outputs = len(joint_names)
sensor_inputs = 1

transient = 300
duration = 1500
NN_SUBSTEPS = 3


def reset_robot(body_id, base_pos=(0.0, 0.0, 0.15), base_orn=(0.0, 0.0, 0.0, 1.0)):
    p.resetBasePositionAndOrientation(body_id, base_pos, base_orn)
    p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    for j in range(p.getNumJoints(body_id)):
        p.resetJointState(body_id, j, targetValue=0.0, targetVelocity=0.0)


def track_xyz(body_id):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    return pos


def set_motors_from_outputs(y):
    m = min(len(y), len(joint_names))
    for i in range(m):
        jname = joint_names[i]
        out_val = float(y[i])
        target = (out_val * 2.0 - 1.0) * (np.pi / 4.0)
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robot_id,
            jointName=jname,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            maxForce=120.0,
        )


class NEATController:
    def __init__(self, genome, config):
        self.net = neat.nn.RecurrentNetwork.create(genome, config)

    def step(self, inputs):
        outputs = self.net.activate(inputs)
        return np.asarray(outputs, dtype=float)


def load_winner(run_idx=5):
    here = os.path.dirname(__file__)
    npz_path = os.path.join(here, f"neat_run{run_idx}.npz")
    data = np.load(npz_path, allow_pickle=True)

    winner = data["winner"].item()

    config_path = os.path.join(here, "config_crawler_neat.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    return winner, config


def simulate_winner(run_idx=5, sleep_gui=True):
    winner, config = load_winner(run_idx)
    ctrl = NEATController(winner, config)

    reset_robot(robot_id)

    for _ in range(transient):
        ctrl.step([0.0])
        p.stepSimulation()
        if sleep_gui:
            time.sleep(1.0 / 240.0)

    out_hist = []
    dist_hist = []
    path_x = []
    path_y = []

    x0, y0, _ = track_xyz(robot_id)

    for i in range(duration):
        for _ in range(NN_SUBSTEPS):
            y = ctrl.step([0.0])

        out_hist.append(y.copy())
        set_motors_from_outputs(y)

        p.stepSimulation()
        if sleep_gui:
            time.sleep(1.0 / 240.0)

        x, y_pos, _z = track_xyz(robot_id)
        path_x.append(x)
        path_y.append(y_pos)
        dist = np.hypot(x, y_pos)
        dist_hist.append(dist)

    out_hist = np.asarray(out_hist)

    plot_logs(out_hist, dist_hist, path_x, path_y,
              title_prefix=f"NEAT winner (run {run_idx})")


def simulate_sine_controller(amplitude=0.5, frequency=1.0, sleep_gui=True):
    reset_robot(robot_id)

    out_hist = []
    dist_hist = []
    path_x = []
    path_y = []

    t = 0.0
    dt = 1.0 / 120.0

    x0, y0, _ = track_xyz(robot_id)

    for i in range(duration):
        joint_targets = sine_controller(t, motor_outputs, amplitude, frequency)

        y = (joint_targets / (np.pi / 4.0)) / 2.0 + 0.5
        y = np.clip(y, 0.0, 1.0)

        out_hist.append(y.copy())

        for j in range(motor_outputs):
            pyrosim.Set_Motor_For_Joint(
                bodyIndex=robot_id,
                jointName=joint_names[j],
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(joint_targets[j]),
                maxForce=120.0,
            )

        p.stepSimulation()
        if sleep_gui:
            time.sleep(dt)

        t += dt

        x, y_pos, _z = track_xyz(robot_id)
        path_x.append(x)
        path_y.append(y_pos)
        dist = np.hypot(x, y_pos)
        dist_hist.append(dist)

    out_hist = np.asarray(out_hist)

    plot_logs(out_hist, dist_hist, path_x, path_y,
              title_prefix="Sine Controller")



def plot_logs(out_hist, dist_hist, path_x, path_y, title_prefix=""):
    plt.figure(figsize=(15, 5))

    # motor outputs
    plt.subplot(1, 3, 1)
    for m in range(min(2, out_hist.shape[1])):
        plt.plot(out_hist[:, m], label=f"motor{m}")
    plt.title(f"{title_prefix}: Neural activity")
    plt.xlabel("Timestep")
    plt.ylabel("Output")
    plt.legend()

    # Distance vs Time
    plt.subplot(1, 3, 2)
    plt.plot(dist_hist, "r", label="Distance from origin")
    plt.title(f"{title_prefix}: Distance vs time")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.legend()

    # X-Y path
    plt.subplot(1, 3, 3)
    plt.plot(path_x, path_y, "b-", label="Robot path")
    if path_x:
        plt.plot(path_x[0],  path_y[0],  "r*", label="Start")
        plt.plot(path_x[-1], path_y[-1], "g*", label="End")
    plt.title(f"{title_prefix}: X-Y path")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        # choose what you want to visualize by changing run_idx=#:
        simulate_winner(run_idx=1, sleep_gui=True)


    finally:
        p.disconnect()