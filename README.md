# DRL4DAPTA
UAV PATH TRACKING AND DYNAMIC AVOIDANCE BASED ON ADS-B AND DEEP REINFORCEMENT LEARNING for Univerisity of Bristol RP3 final
# Backgorund
The code is the final submission code for University of Bristol aerospace engineering--Research Project 3. And constructed based on https://github.com/utiasDSL/gym-pybullet-drones
Contact cf20770@bristol.ac.uk if you have any question


# Abstraction
This paper presents a novel dynamic avoidance and path tracking algorithm for unmanned aerial vehicles (UAVs) called DRL4DAPTA. The algorithm utilizes deep reinforcement learning and is implemented using ADS-B as a signal transmission intermediary and PyBullet as a simulation and training platform. The algorithm is designed to adapt to real-life situations by considering the error of the ADS-B signal. The reward function is constructed by predicting collision time, and the path-following algorithm is based on interacting with a set path and designing the corresponding reward function. To ensure safety despite the maximum error, the ADS-B signal error is processed by designing a sufficient detection zone. Two comparison algorithms, Proximal Policy Optimization (PPO) and deep Deterministic Policy Gradient (DDPG), are evaluated using the same reward function. Simulation experiments demonstrate that DRL4DAPTA achieves stable path tracking and dynamic obstacle avoidance in the presence of errors and low ADS-B signal frequency in a relative stable manner, verifying the feasibility of the algorithm.


# Contents
The main contents includes four objects: ‘ControlAviary’, ‘adsbcentre’, ‘Singleagent’, and ‘test_singleagent’. Among them, ‘ControlAviary’ is derived from ‘gym.Env’ and is used to control the actions of the agent drone, which implementing the interface functions of OpenAI gym, such as step, reward, reset, etc. ‘ControlAviary’ is responsible for calculating the states s_t  and actions a_t of each step of the agent drone and updating the information to the neural network using the 'step' function. ‘Adsbcentre’ is used to store the information of the agent drone at each step and add ADS-B errors to the information. Meanwhile, the transmission frequency of ‘ControlAviary’ to adsbcentre is set to simulate the ADS-B frequency. ‘Singleagent’ is responsible for training, while ‘test_singleagent’ is responsible for conducting 10 experiments with a successful model to obtain the results.

In addition, it contains the successful results below:
-PPO algorithm, 48Hz ADS-B signal, 1M trainning times.
-DDPG algorithm, 48Hz ADS-B signal, 1M trainning times.
-PPO algorithm, 12Hz ADS-B signal, 1M trainning times.
-PPO algorithm, 8Hz ADS-B signal, 1M trainning times.
-PPO algorithm, 2Hz ADS-B signal, 1M trainning times.
-PPO algorithm, 2Hz ADS-B signal, 2M trainning times.
