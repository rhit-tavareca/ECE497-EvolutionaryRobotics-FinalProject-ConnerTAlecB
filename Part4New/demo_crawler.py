import pybullet as p
import pybullet_data
import time
import numpy as np

from demo_controller import sine_controller
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
robot_id = p.loadURDF("crawlerbot.urdf")
num_joints = p.getNumJoints(robot_id)
print("Number of joints:", num_joints)

t = 0.0
dt = 1.0 / 240.0
amplitude = 0.5
frequency = 1.0

while True:
    joint_targets = sine_controller(t, num_joints, amplitude, frequency)

    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=float(joint_targets[j]),
            force=50.0
        )
    p.stepSimulation()
    t += dt
    time.sleep(dt)