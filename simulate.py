import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data 
import time   

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
# robotId = p.loadURDF("robot.urdf")
robotId = p.loadURDF("crawlerbot.urdf")

#p.loadSDF("boxes.sdf")

pyrosim.Prepare_To_Simulate(robotId)

duration = 100000

for i in range(duration):
    p.stepSimulation()
    # Make that joint be X 
    time.sleep(1/60)

print("Simulation has ended")
p.disconnect() 