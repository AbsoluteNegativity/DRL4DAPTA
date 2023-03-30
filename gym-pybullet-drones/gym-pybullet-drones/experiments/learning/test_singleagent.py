"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import re
from turtle import pos
import numpy as np
import gym
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy as a2cppoCnnPolicy
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy
from stable_baselines3.sac import CnnPolicy as sacCnnPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.td3 import CnnPolicy as td3ddpgCnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import sys
sys.path.append("C:/software/gym-pybullet-drones")

from gym_pybullet_drones.envs.ControlAviary import ControlAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool

import shared_constants

DEFAULT_GUI = False
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = 'results'
class OnnxablePolicy(torch.nn.Module):
  def __init__(self, extractor, action_net, value_net):
      super(OnnxablePolicy, self).__init__()
      self.extractor = extractor
      self.action_net = action_net
      self.value_net = value_net

  def forward(self, observation):
      # NOTE: You may have to process (normalize) observation in the correct
      #       way before using this. See `common.preprocessing.preprocess_obs`
      action_hidden, value_hidden = self.extractor(observation)
      return self.action_net(action_hidden), self.value_net(value_hidden)
def run(exp, gui=DEFAULT_GUI, plot=DEFAULT_PLOT, output_folder=DEFAULT_OUTPUT_FOLDER):
    #### Load the model from file ##############################
    exp="./results/save-ControlAviary-ppo-kin-tun-48HZ-1M" #load the model
    ###Note: change the corresponding ADS-B signal frequency in ControlAviary###
    algo = exp.split("-")[2]

    if os.path.isfile(exp+'/success_model.zip'):
        path = exp+'/success_model.zip'
    elif os.path.isfile(exp+'/best_model.zip'):
        path = exp+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", exp)
    if algo == 'a2c':
        model = A2C.load(path)
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)
    if algo == 'td3':
        model = TD3.load(path)
    if algo == 'ddpg':
        model = DDPG.load(path)
    model.policy.to("cpu")
    #onnxable_model = OnnxablePolicy(model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net)

    dummy_input = torch.randn(1, 15)
    #torch.onnx.export(onnxable_model, dummy_input, "my_ppo_model.onnx", opset_version=9)
    #### Parameters to recreate the environment ################
    env_name = exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN if exp.split("-")[3] == 'kin' else ObservationType.RGB

    # Parse ActionType instance from file name
    action_name = exp.split("-")[4]
    ACT = [action for action in ActionType if action.value == action_name]
    if len(ACT) != 1:
        raise AssertionError("Result file could have gotten corrupted. Extracted action type does not match any of the existing ones.")
    ACT = ACT.pop()

    ### Evaluate the model ####################################
    eval_env = gym.make(env_name,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
    mean_reward, std_reward = evaluate_policy(model,
                                              eval_env,
                                              n_eval_episodes=10
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    ### Show, record a video, and log the model's performance #
    test_env = gym.make(env_name,
                        gui=gui,
                        record=False,
                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                        obs=OBS,
                        act=ACT
                        )
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=output_folder
                    )
    
    #Run Simulation for 10 times to find the average cloest distance,step reward and position error
    for j in range(10):
        obs = test_env.reset()
        start = time.time()
        adsbinfoes=[]
        noises=[]
        poses_true=[]
        poses_True=[]
        close_ds=[]
        mean_ds=[]
        epsilons=[]
        rewardts=[]
        for i in range(30*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
            action, _states = model.predict(obs,
                                            deterministic=True # OPTIONAL 'deterministic=False'
                                            )
            # action=test_env.action_space.sample()
            # print(obs,action)
            obs, reward, done, info = test_env.step(action)
            test_env.render()
            adsbinfo,noise,typeA,num_drones,pos_true,mean_d,close_d,epsilon,rewardt=test_env.receiveinfo()
            adsbinfoes.append(adsbinfo)
            noises.append(noise)
            poses_true=pos_true.copy()
            poses_True.append(poses_true)
            close_ds.append(close_d)
            mean_ds.append(mean_d)
            epsilons.append(epsilon)
            rewardts.append(rewardt)
            if OBS==ObservationType.KIN:
                logger.log(drone=0,
                        timestamp=i/test_env.SIM_FREQ,
                        state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (1))]),
                        control=np.zeros(12)
                        )
            sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
            # if done: obs = test_env.reset() # OPTIONAL EPISODE HALT
        #nums = list(map(int, cloest_d))
        nums=np.array(close_ds)
        True_close_d= nums[0]
        for i in range(len(nums)):
            if nums[i] < True_close_d:
                True_close_d = nums[i]
        meannums=np.array(mean_ds)
        True_mean_ds=sum(nums)/len(nums)
        print(True_close_d,True_mean_ds)
    ep=np.array(epsilons)
    re=np.array(rewardts)
    np.savetxt('reward.txt', re, fmt="%.4f", delimiter=',')
    np.savetxt('output.txt', ep, fmt="%.4f", delimiter=',')
    test_env.close()

##Plot the path of drones
    logger.save_as_csv("sa") # Optional CSV save
    if plot:
        
         #logger.plot()

        fig = plt.figure(figsize=(12,12))
        ax1 = plt.subplot(projection='3d')
        noisesdata=np.array(noises)
        
        ax1.scatter(noisesdata[:,typeA-1,0],noisesdata[:,typeA-1,1], noisesdata[:,typeA-1,2],cmap='Blues', s=1,label='original input')
        
        adsbinfoesdata=np.array(adsbinfoes)
        ax1.plot3D(adsbinfoesdata[:,typeA-1,0],adsbinfoesdata[:,typeA-1,1],adsbinfoesdata[:,typeA-1,2],'gray',label='filter output')

        posdata=np.array(poses_True)
        ax1.plot3D(posdata[:,typeA-1,0],posdata[:,typeA-1,1],posdata[:,typeA-1,2],'red',label='actual position')
        ax1.legend()
        ax1.set_xlabel('x-axis (m)')
        ax1.set_ylabel('y-axis (m)')
        ax1.set_zlabel('z-axis (m)')
        fig = plt.figure(figsize=(12,12))
        ax1 = plt.subplot(projection='3d')#fig.gca(projection='3d')
        for i in range(typeA-3):
            ax1.plot3D(adsbinfoesdata[:,i,0],adsbinfoesdata[:,i,1],adsbinfoesdata[:,i,2],'gray')
        ax1.plot3D(adsbinfoesdata[:,typeA-2,0],adsbinfoesdata[:,typeA-2,1],adsbinfoesdata[:,typeA-2,2],'gray',label='Paths of the Type B UAV')
        ax1.plot3D(adsbinfoesdata[:,typeA-1,0],adsbinfoesdata[:,typeA-1,1],adsbinfoesdata[:,typeA-1,2],'red', label='Paths of Type A UAV')
        ax1.legend()
        ax1.set_xlabel('x-axis (m)')
        ax1.set_ylabel('y-axis (m)')
        ax1.set_zlabel('z-axis (m)')
        plt.show()
    

    # with np.load(exp+'/evaluations.npz') as data:
    #     print(data.files)
    #     print(data['timesteps'])
    #     print(data['results'])
    #     print(data['ep_lengths'])

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
    parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
    parser.add_argument('--gui',            default=DEFAULT_GUI,               type=str2bool,      help='Whether to use PyBullet GUI (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))