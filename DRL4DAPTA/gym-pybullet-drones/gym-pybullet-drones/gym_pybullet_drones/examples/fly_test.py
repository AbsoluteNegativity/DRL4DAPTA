"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `ControlAviary` ` environment.
The control is given by the PID implementation in `DSLPIDControl`.


"""
from http.cookiejar import FileCookieJar
import os
from re import A
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/software/gym-pybullet-drones")
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.ControlAviary import ControlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
if __name__ == "__main__":

    gui = False
    output_folder='results'
    colab=False
    simulation_freq_hz=240
    duration_sec=30
    env = ControlAviary(freq=simulation_freq_hz,
                                    gui = False,obstacles=True)
    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    simulation_freq_hz=240
    AGGR_PHY_STEPS=5
    num_drones=2
    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    control_freq_hz=48
    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(simulation_freq_hz/control_freq_hz))
    START = time.time()
    totalReward=0
    curreward=0
    epsilons=[]
    rewardts=[]
    for epi in range(1):
        env.reset()
        print (curreward)
        curreward=0
        for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

            #### Make it rain rubber ducks #############################
            # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)
            action=env.action_space.sample()
            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)
            adsbinfo,noise,typeA,num_drones,pos_true,mean_d,close_d,epsilon,rewardt=env.receiveinfo()
            # print (obs,action)
            curreward+=reward
            totalReward+=reward
            epsilons.append(epsilon)
            rewardts.append(rewardt)

            
            logger.log(drone=0,
                        timestamp=i/env.SIM_FREQ,
                        state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (1))]),
                        control=np.zeros(12)
                        )
            if i%env.SIM_FREQ == 0:
                env.render()
    print (curreward)
    print (totalReward/10)
    ep=np.array(epsilons)
    re=np.array(rewardts)
    np.savetxt('reward_PID.txt', re, fmt="%.4f", delimiter=',')
    np.savetxt('output_PID.txt', ep, fmt="%.4f", delimiter=',')
    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    # logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    # if plot:
    logger.plot()
