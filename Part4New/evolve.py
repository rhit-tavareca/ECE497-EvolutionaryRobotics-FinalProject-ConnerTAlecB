import eas
import ctrnn

import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data
import time   

import numpy as np
import matplotlib.pyplot as plt

# physicsClient = p.connect(p.GUI) 
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)
p.setTimeStep(1/120)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf")

pyrosim.Prepare_To_Simulate(robotId)


nj = p.getNumJoints(robotId)
for link_i in range(-1, nj):
    p.changeDynamics(robotId, link_i, lateralFriction = 1.4, rollingFriction = 0.01, spinningFriction = 0.01)

joint_names = [
    p.getJointInfo(robotId,j)[1].decode()
    for j in range(nj)
    if p.getJointInfo(robotId,j)[2] == p.JOINT_REVOLUTE
    ]

motor_outputs = len(joint_names)
sensor_inputs = 1

transient = 300
duration = 1500                 

nnsize = 8
dt = 0.01
TimeConstMin = 0.4
TimeConstMax = 1.2
WeightRange = 10.0
BiasRange = 10.0
NN_SUBSTEPS = 3

nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)

def reset_robot(robotId, base_pos=[0,0,0.15], base_orn=[0,0,0,1]):
    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robotId, base_pos, base_orn)

    # zero out velocity
    p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

    # Reset all joint angles and velocities
    num_joints = p.getNumJoints(robotId)
    for j in range(num_joints):
        p.resetJointState(robotId, j, targetValue=0.0, targetVelocity=0.0)

def track_xyz(robot_id):
    pos, _ = p.getBasePositionAndOrientation(robot_id)
    return pos

def set_motors_from_outputs(y):
    m = min(len(y), len(joint_names))
    # y = np.clip(y, 0.02, 0.98)
    for i in range(m):
        jname = joint_names[i]
        out = y[i]
        target = (out * 2 - 1) * (np.pi/4)
        pyrosim.Set_Motor_For_Joint(
            bodyIndex=robotId,
            jointName= jname,
            controlMode = p.POSITION_CONTROL,
            targetPosition=float(target),
            maxForce=120.0
            )

LAMBDA_ROLLPITCH = 2.0    
LAMBDA_BOUNCE    = 0.5   
LAMBDA_PATHXY    = 0.2    
LAMBDA_DISP      = 8.0    
LAMBDA_AMP       = 0.2  

def fitnessFunction(genotype):
    # Reset joints / Reset the coordinates the body 
    reset_robot(robotId)

    # Set and initialize the neural network
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))

    # Simulate both NN and Body for a little while 
    # without connecting them, so that transients pass 
    # for both of them
    for i in range(transient):
        nn.step(dt,[0.0])
        p.stepSimulation()

    # Test period
    # Get starting position (after the transient)
    x0, y0, z0 = track_xyz(robotId)
    xp, yp, zp = x0, y0, z0
    path_xy = 0
    bounce = 0 
    tilt_pen = 0
    amp_pen = 0
    steps_taken = 0

    # Simulate both NN and Body
    for i in range(duration):
        for t in range(NN_SUBSTEPS):
            nn.step(dt, [0])

        y = nn.out()
        amp_pen += float(np.mean(np.abs(y - 0.5)))
        set_motors_from_outputs(y)
        p.stepSimulation()

        x, y_, z = track_xyz(robotId)
        path_xy += np.hypot(x - xp, y_ - yp)
        bounce  += abs(z - zp)
        xp, yp, zp = x, y_, z

        orn = p.getBasePositionAndOrientation(robotId)[1]
        roll, pitch, _yaw = p.getEulerFromQuaternion(orn)
        tilt_pen += (abs(roll) + abs(pitch))

        steps_taken = i + 1

        if (abs(roll) + abs(pitch)) > 1.2 or z < 0.02:
            break

    xf, yf, _ = track_xyz(robotId)
    disp = np.hypot(xf - x0, yf - y0)

    fitness  = (LAMBDA_DISP * disp)
    fitness += (LAMBDA_PATHXY * path_xy)
    fitness -= (LAMBDA_BOUNCE * bounce)
    fitness -= (LAMBDA_ROLLPITCH * tilt_pen / steps_taken)
    fitness -= (LAMBDA_AMP * amp_pen / steps_taken)     
    return fitness

# EA Params
popsize = 10
genesize = nnsize*nnsize + sensor_inputs*nnsize + motor_outputs*nnsize + nnsize+nnsize
recombProb = 0.5
mutatProb = 0.01
demeSize = 2
generations = 100

# Evolve and visualize fitness over generations
ga = eas.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()

# Get best evolved network
_,_,best_ind = ga.fitStats()   

# Save 
np.save("bestgenotype.npy",best_ind)

p.disconnect() 

