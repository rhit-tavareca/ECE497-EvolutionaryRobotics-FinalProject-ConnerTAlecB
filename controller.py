import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("crawlerbot.urdf")

num_joints = p.getNumJoints(robot_id)
print("Number of joints:", num_joints)

# --- Motion parameters ---
t = 0
dt = 1/240
amplitude = 0.5
frequency = 1.0
phase_offsets = [0, np.pi/2, np.pi, 3*np.pi/2]  # each limb moves out of phase

# --- Run simulation loop ---
while True:
    for j in range(num_joints):
        target_angle = amplitude * np.sin(2*np.pi*frequency*t + phase_offsets[j])
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_angle,
            force=50
        )
    p.stepSimulation()
    t += dt
    time.sleep(dt)
