# import eas
# import ctrnn

# import pybullet as p  
# import pyrosim.pyrosim as pyrosim
# import pybullet_data
# import time   

# import numpy as np
# import matplotlib.pyplot as plt

# physicsClient = p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
# p.setGravity(0,0,-9.8)

# planeId = p.loadURDF("plane.urdf")
# robotId = p.loadURDF("robot.urdf")

# pyrosim.Prepare_To_Simulate(robotId)

# transient = 1000
# duration = 4000  
# t = np.linspace(0,1,num=duration)

# nnsize = 5
# sensor_inputs = 1
# motor_outputs = 2

# dt = 0.01
# TimeConstMin = 1.0
# TimeConstMax = 2.0
# WeightRange = 10.0
# BiasRange = 10.0

# def reset_robot(robotId, base_pos=[0,0,1], base_orn=[0,0,0,1]):
#     # Reset base position and orientation
#     p.resetBasePositionAndOrientation(robotId, base_pos, base_orn)

#     # zero out velocity
#     p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

#     # Reset all joint angles and velocities
#     num_joints = p.getNumJoints(robotId)
#     for j in range(num_joints):
#         p.resetJointState(robotId, j, targetValue=0.0, targetVelocity=0.0)

# def fitnessFunction(genotype):
#     reset_robot(robotId)

#     nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)
#     nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
#     nn.initializeState(np.zeros(nnsize))

#     for i in range(transient):
#         nn.step(dt,[0])
#         p.stepSimulation()

#     output = np.zeros((duration,nnsize))
#     motorout = np.zeros((duration,motor_outputs))
#     legsensor_hist = np.zeros(duration)

#     linkState = p.getLinkState(robotId,0)
#     posx_start = linkState[0][0]
#     posy_start = linkState[0][1]

#     # Test period
#     # Get starting position (after the transient)
#     linkState = p.getLinkState(robotId,0)
#     posx_start = linkState[0][0]
#     posy_start = linkState[0][1]
#     posx_current = linkState[0][0]
#     posy_current = linkState[0][1]
#     posz_current = linkState[0][2]   

#     distance_traveled = 0.0 # distance traveled in the x-y plane at each step, to be maximized
#     distance_jumped = 0.0 # amount of movement up and down, to be minimized

#     for i in range(duration):
#         legsensor = pyrosim.Get_Touch_Sensor_Value_For_Link("2")
#         legsensor_hist[i] = legsensor
#         for ttt in range(10):
#             nn.step(dt,[legsensor])
#         output[i] = nn.Output
#         motorout[i] = nn.out()
#         motoroutput = nn.out()
#         p.stepSimulation()

#         pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
#                                     jointName="0_1", 
#                                     controlMode = p.POSITION_CONTROL,
#                                     targetPosition = (motoroutput[0]*2-1)*np.pi/4,
#                                     maxForce = 500
#                                     )
        
#         pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
#                                     jointName="1_2", 
#                                     controlMode = p.POSITION_CONTROL,
#                                     targetPosition =  (motoroutput[1]*2-1)*np.pi/4,
#                                     maxForce = 500
#                                     )    
        
#         time.sleep(1/1000) # NEW 
#         posx_past = posx_current
#         posy_past = posy_current
#         posz_past = posz_current   
#         linkState = p.getLinkState(robotId,0)
#         posx_current = linkState[0][0]
#         posy_current = linkState[0][1]
#         posz_current = linkState[0][2]    
#         distance_traveled += np.sqrt((posx_current - posx_past)**2 + (posy_current - posy_past)**2)
#         distance_jumped += np.sqrt((posz_current - posz_past)**2)
        

#     linkState = p.getLinkState(robotId,0)
#     posx_end = linkState[0][0]
#     posy_end = linkState[0][1]

#     distance_final = np.sqrt((posx_start - posx_end)**2 + (posy_start - posy_end)**2)

#     print(distance_final, distance_traveled, distance_jumped)

#     return distance_final + distance_traveled - distance_jumped, output, motorout, legsensor_hist

# # Load
# best = np.load("bestgenotype.npy")
# fit, output, motorout, sensor = fitnessFunction(best)
# print(fit)

# p.disconnect() 

# # Plot activity
# plt.plot(output,alpha=0.5)
# plt.plot(motorout.T[0],'k')
# plt.plot(motorout.T[1],'k:')
# plt.plot(sensor/2 + 0.5,'r')
# plt.xlabel("Time")
# plt.ylabel("Output")
# plt.title("Neural activity")
# plt.show()


import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data 
import time   
import numpy as np
import matplotlib.pyplot as plt
import ctrnn

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf")

#p.loadSDF("boxes.sdf")

pyrosim.Prepare_To_Simulate(robotId)

nj = p.getNumJoints(robotId)
for link_i in range(-1, nj):
    p.changeDynamics(robotId, link_i, lateralFriction = 1.2, rollingFriction = 0.01, spinningFriction = 0.01)

joint_names = [
    p.getJointInfo(robotId,j)[1].decode()
    for j in range(nj)
    if p.getJointInfo(robotId,j)[2] == p.JOINT_REVOLUTE
    ]

motor_outputs = len(joint_names)
sensor_inputs = 1

nnsize = 8
dt = 0.01
TimeConstMin = 0.4
TimeConstMax = 1.2
WeightRange = 10.0
BiasRange = 10.0

USE_RANDOM = True

nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)

if USE_RANDOM:
    nn.randomizeParameters()
else:
    genotype = np.load("bestgenotype.npy")
    nn.setParameters(genotype, WeightRange, BiasRange, TimeConstMin, TimeConstMax)

nn.initializeState(np.zeros(nnsize))

def reset_robot(robotId, base_pos=[0,0,0.15], base_orn=[0,0,0,1]):
    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robotId, base_pos, base_orn)

    # zero out velocity
    p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

    # Reset all joint angles and velocities
    num_joints = p.getNumJoints(robotId)
    for j in range(num_joints):
        p.resetJointState(robotId, j, targetValue=0.0, targetVelocity=0.0)


def set_motors(y):
    m = min(len(y), len(joint_names))
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

reset_robot(robotId)

out_hist = []

for t in range(150):
    nn.step(dt, [0])
    p.stepSimulation()

steps = 4000
NN_SUBSTEPS = 3

for i in range(steps):
    for t in range(NN_SUBSTEPS):
        nn.step(dt, [0])
    y = nn.out()
    out_hist.append(y.copy())
    set_motors(y)
    p.stepSimulation()
    # if i % 2 == 0:
    #     time.sleep(1/240)

p.disconnect()

out_hist = np.array(out_hist)
plt.plot(out_hist[:, 0], label = "motor0")
if out_hist.shape[1] > 1:
    plt.plot(out_hist[:, 1], label="motor1")
plt.title("Neural activity (motor outputs)")
plt.xlabel("Timestep"); plt.ylabel("output")
plt.legend(); plt.show()

# dt = 1 / 240
# freq = 2.5
# A = 0.6
# phase = np.pi/3
# direction = 1

# xlog = np.zeros(steps)

# log_names = [joint_names[0], joint_names[len(joint_names)//2], joint_names[-1]]
# theta_log = {}
# for n in log_names:
#     theta_log[n] = []

# def setmotor(name, val):
#     pyrosim.Set_Motor_For_Joint(
#         bodyIndex=robotId,
#         jointName=name,
#         controlMode=p.POSITION_CONTROL,
#         targetPosition=float(val),
#         maxForce=160
#         )

# for i in range(steps):
#     t = i * dt
#     base_phase = 2*np.pi*freq*t

#     k = 0
#     for name in joint_names:
#         theta = A * np.sin(direction *(base_phase - k * phase))
#         setmotor(name, theta)
#         if name in theta_log:
#             theta_log[name].append(theta)

#         k += 1
#     p.stepSimulation()
#     if i % 2 == 0 : 
#         time.sleep(dt)
#     xlog[i] = p.getBasePositionAndOrientation(robotId)[0][0]
# p.disconnect() 


# plt.plot(xlog)
# plt.title("Forward Progress (x vs time)")
# plt.xlabel("timestep")
# plt.ylabel("x (m)")
# plt.show()

# plt.figure()
# for n in log_names:
#     plt.plot(theta_log[n][:500], label=n)
# plt.title("Joint Angles for 3 segments")
# plt.xlabel("timestep")
# plt.ylabel("angle (rad)")
# plt.legend()
# plt.show()