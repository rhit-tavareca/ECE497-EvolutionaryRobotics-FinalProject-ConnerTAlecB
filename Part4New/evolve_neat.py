import os
import time

import neat
import numpy as np

import pybullet as p
import pyrosim.pyrosim as pyrosim
import pybullet_data
import matplotlib.pyplot as plt

physics_client = p.connect(p.DIRECT)
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

# Fitness weights (same structure as your P4 fitness)
LAMBDA_ROLLPITCH = 2.0
LAMBDA_BOUNCE = 0.5
LAMBDA_PATHXY = 0.2
LAMBDA_DISP = 8.0
LAMBDA_AMP = 0.2


def reset_robot(body_id, base_pos=(0.0, 0.0, 0.15), base_orn=(0.0, 0.0, 0.0, 1.0)):
    p.resetBasePositionAndOrientation(body_id, base_pos, base_orn)
    p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    num_joints_local = p.getNumJoints(body_id)
    for j in range(num_joints_local):
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



def simulate_controller(controller):
    reset_robot(robot_id)

    for _ in range(transient):
        controller.step([0.0])
        p.stepSimulation()

    x0, y0, z0 = track_xyz(robot_id)
    xp, yp, zp = x0, y0, z0

    path_xy = 0.0
    bounce = 0.0
    tilt_pen = 0.0
    amp_pen = 0.0
    steps_taken = 0

    for i in range(duration):
        for _ in range(NN_SUBSTEPS):
            y = controller.step([0.0])

        amp_pen += float(np.mean(np.abs(y - 0.5)))
        set_motors_from_outputs(y)
        p.stepSimulation()

        x, y_pos, z = track_xyz(robot_id)
        path_xy += np.hypot(x - xp, y_pos - yp)
        bounce += abs(z - zp)
        xp, yp, zp = x, y_pos, z

        _, orn = p.getBasePositionAndOrientation(robot_id)
        roll, pitch, _yaw = p.getEulerFromQuaternion(orn)
        tilt_pen += abs(roll) + abs(pitch)

        steps_taken = i + 1

        if (abs(roll) + abs(pitch)) > 1.2 or z < 0.02:
            break

    xf, yf, _ = track_xyz(robot_id)
    disp = np.hypot(xf - x0, yf - y0)

    # Same structure as your P4 fitness function
    fitness = LAMBDA_DISP * disp
    fitness += LAMBDA_PATHXY * path_xy
    fitness -= LAMBDA_BOUNCE * bounce
    fitness -= LAMBDA_ROLLPITCH * (tilt_pen / max(1, steps_taken))
    fitness -= LAMBDA_AMP * (amp_pen / max(1, steps_taken))

    return float(fitness)



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ctrl = NEATController(genome, config)
        genome.fitness = simulate_controller(ctrl)


def run_neat(run_id=None, seed=None):
    here = os.path.dirname(__file__)
    config_path = os.path.join(here, "config_crawler_neat.ini")

    if seed is not None:
        np.random.seed(seed)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    generations = 100
    winner = pop.run(eval_genomes, generations)
    best_fitness = np.asarray(stats.get_fitness_stat(max), dtype=float)
    avg_fitness = np.asarray(stats.get_fitness_mean(), dtype=float)

    if run_id is None:
        fname = "neat_run.npz"
    else:
        fname = f"neat_run{run_id}.npz"

    np.savez(
        fname,
        avghist=avg_fitness,
        besthist=best_fitness,
        winner=winner,
    )

    return winner, best_fitness, avg_fitness

def plot_fitness(best, avg, title="Best and average fitness"):
    plt.figure()
    plt.plot(best, label="Best Fitness")
    plt.plot(avg, label="Average Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    try:
        num_runs = 5
        all_best = []
        all_avg = []

        for r in range(1, num_runs + 1):
            print(f"\n=== NEAT run {r} ===")
            seed = 1000 + r
            winner, best, avg = run_neat(run_id=r, seed=seed)
            all_best.append(best)
            all_avg.append(avg)


            plot_fitness(best, avg, title=f"NEAT run {r}: best and average fitness")

        all_best = np.vstack(all_best)
        all_avg = np.vstack(all_avg)

        mean_best = np.mean(all_best, axis=0)
        mean_avg = np.mean(all_avg, axis=0)

        plot_fitness(mean_best, mean_avg, title="NEAT: mean best/avg over runs")

    finally:
        p.disconnect()