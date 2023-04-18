import os
import random
from tkinter import PhotoImage
import numpy as np
from gym import spaces
from enum import Enum
import math
import pybullet as p
from random import choice
from gym_pybullet_drones.utils.adsbcenter import adsbcenter as adsbcenter
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType
from gym_pybullet_drones.utils.utils import nnlsRPM
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType


class ControlAviary(BaseAviary):
    """Multi-drone environment class for control applications."""

    ################################################################################
    

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=30,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=5,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.TUN
                 ):
        self.PERIOD = 30
        vision_attributes = True if obs == ObservationType.RGB else False
        dynamics_attributes = True if act in [ActionType.DYN, ActionType.ONE_D_DYN] else False
        self.typeA = num_drones
        self.typeB = num_drones-1      
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.ctrl_b = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(self.typeB)]
        self.reward_c=0
        self.reward_collision=0
        self.reward_p=0
        self.reward_t=0
        self.Done=False
        self.INIT_XYZSforA=[0,0,0.5]
        self.INIT_RPYSforA=[0.0,0.0,0.0]
        self.INIT_XYZSforB = np.zeros((self.typeB,3))
        self.INIT_RPYSforB = np.zeros((self.typeB,3))
        self.epsilon=0
        self.NUM_WP = int((freq * self.PERIOD) /aggregate_phy_steps)
        self.ADSB_info = np.zeros((num_drones,6))
        self.ADSB_unchange=np.zeros((num_drones,6))
        self.ADSB_noise=np.zeros((num_drones,6))
        self.TARGET_POS_B = np.zeros((self.typeB,self.NUM_WP,3))
        self.cloest_d_collision=[]
        self.mean_d_collision=[]
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.TUN, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
                if act == ActionType.TUN:
                    self.TUNED_P_POS = np.array([.4, .4, 1.25])
                    self.TUNED_I_POS = np.array([.05, .05, .05])
                    self.TUNED_D_POS = np.array([.2, .2, .5])
                    self.TUNED_P_ATT = np.array([70000., 70000., 60000.])
                    self.TUNED_I_ATT = np.array([.0, .0, 500.])
                    self.TUNED_D_ATT = np.array([20000., 20000., 12000.])
            else:
                print("[ERROR] in BaseSingleAgentAviary.__init()__, no controller is available for the specified drone_model")
        """Initialization of an aviary environment for control applications.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """

        self.TRAJ_STEPS = int((freq* self.PERIOD) / aggregate_phy_steps)
        self.CTRL_TIMESTEP = 1./freq*aggregate_phy_steps
        self.TARGET_POSITION = np.array([[50*i/self.TRAJ_STEPS+self.INIT_XYZSforA[0],10*np.sin(5*i/self.TRAJ_STEPS)+self.INIT_XYZSforA[1],30*i/self.TRAJ_STEPS+self.INIT_XYZSforA[2]] for i in range(self.TRAJ_STEPS)])
        #### Derive the trajectory to obtain target velocity #######
        self.TARGET_VELOCITY = np.zeros([self.TRAJ_STEPS, 3])
        self.TARGET_VELOCITY[1:, :] = (self.TARGET_POSITION[1:, :] - self.TARGET_POSITION[0:-1, :]) / self.CTRL_TIMESTEP

        self._initialpositionforB()
        self._pathforBtype()
        
        initial_xyzs=np.vstack((self.INIT_XYZSforB,self.INIT_XYZSforA))
        initial_rpys=np.vstack((self.INIT_RPYSforB,self.INIT_RPYSforA))
        self.EPISODE_LEN_SEC = 30   
        self.adsbserver = adsbcenter(num_drones,self.TRAJ_STEPS)

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         dynamics_attributes=dynamics_attributes
                         )

    ################################################################################
    def _initialpositionforB(self):
        """.
        Returns
        -------
        ndarray[NUM_DRONES,3],ndarray[NUM_DRONES,3]
            A array of Box(3,) with NUM_DRONES entries,
        """     
        for j in range (self.typeB):
            #k_1 = random.uniform(10, -10)
            k_1=random.uniform(24.0+0,50.0+0 )
            k_2=random.uniform(-24.0+0,24.0+0)
            k_3=random.uniform(3.5,5)
        
            self.INIT_XYZSforB[j,:] = k_1, k_2, k_3
            self.INIT_RPYSforB[j,:] = 0.0, 0.0, 0.0
        
    
    def _pathforBtype(self):
        
        NUM_WP = self.NUM_WP
        for j in range (self.typeB):
            aindex=random.randint(0.0,6.0)
            list=[1,-1] 
            k_0=choice(list)
            k_1=random.uniform(120,150)
            k_2=random.uniform(5.0,6.0)
            k_3=random.uniform(80.0,100.0)
            k_4=random.uniform(0.0,45.05)
            k_5=random.uniform(20.0,25.0)
            R = random.uniform(0.0,1.0)
            for i in range(NUM_WP):
                if aindex==0:
                    waypoints=np.array([self.INIT_XYZSforB[j, 0]+k_0*k_1*i/NUM_WP, self.INIT_XYZSforB[j, 1]+k_0*k_1*i/NUM_WP,self.INIT_XYZSforB[j, 2]+k_4*i/NUM_WP]) #Straight line
                elif aindex==1:
                    waypoints=np.array([k_0*k_2*np.cos((i/NUM_WP)*6+np.pi/2)+self.INIT_XYZSforB[j, 0], k_2*k_0*np.sin((i/NUM_WP)*6)-R+self.INIT_XYZSforB[j, 1], self.INIT_XYZSforB[j, 2]+k_4*i/NUM_WP]) #Spirals
                elif aindex==2:
                    waypoints=np.array([k_3*k_0*i/NUM_WP+self.INIT_XYZSforB[j, 0], k_0*k_5*math.log(10*i/NUM_WP+1)+self.INIT_XYZSforB[j, 1], self.INIT_XYZSforB[j, 2]+k_4*i/NUM_WP]) #Logarithmic curves
                elif aindex==3:
                    waypoints=np.array([k_3*k_0*i/NUM_WP+self.INIT_XYZSforB[j, 0], k_2*k_0*np.sin(i/NUM_WP*6)+self.INIT_XYZSforB[j, 1], self.INIT_XYZSforB[j, 2]+(i/NUM_WP)*k_4]) #Sine
                elif aindex==4:
                    waypoints=np.array([k_3*k_0*i/NUM_WP+self.INIT_XYZSforB[j, 0], k_2*k_0*np.cos(i/NUM_WP*6+0.5*np.pi)+self.INIT_XYZSforB[j, 1], self.INIT_XYZSforB[j, 2]+(i/NUM_WP)*k_4]) #Cosine
                else: 
                    waypoints=np.array([k_3*k_0*i/NUM_WP+self.INIT_XYZSforB[j, 0], k_3*k_0*(i/NUM_WP)*(i/NUM_WP)+self.INIT_XYZSforB[j, 1], self.INIT_XYZSforB[j, 2]+(i/NUM_WP)*k_4]) #Parabola
                self.TARGET_POS_B[j, i, :]=waypoints

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        dict[str, ndarray]
            A Dict of Box(4,) with NUM_DRONES entries,
            indexed by drone Id in string format.

        """
        self.ACT_TYPE == ActionType.TUN
        act_lower_bound = np.array([-0.2,-0.2,-0.2,-0.2,-0.2,-0.2])
        act_upper_bound = np.array([0.2, 0.2, 0.2, 0.2,0.2, 0.2])
        return spaces.Box(low=act_lower_bound,
                          high=act_upper_bound,
                          dtype=np.float32
                          )
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """
        #x,y,z,r,p,y,u,v,w,dh,dv,xyr,zR,rpR,TRR

        return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,0, 0,-1,0,-1,0,-1]),
                               high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1,1,1,1]),
                               dtype=np.float32
                               )

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of key "state", see the implementation of `_getDroneStateVector()`,
        the value of key "neighbors" is the drone's own row of the adjacency matrix.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            A Dict with NUM_DRONES entries indexed by Id in string format,
            each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES)}.

        """

        return self._clipAndNormalizeState(self._ProcessADSB())
        ############################################################
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # return obs
        ############################################################
        #### OBS SPACE OF SIZE 12
        #ret = np.hstack(obs[0:14]).reshape(12,)
        #ret = np.hstack(obs[0:14]).reshape(14,)
        
        #return ret.astype('float32')
        # adjacency_mat = self._getAdjacencyMatrix()
        # return {str(i): {"state": self._getDroneStateVector(i), "neighbors": adjacency_mat[i, :]} for i in range(self.NUM_DRONES)}

    ################################################################################


    def _calculatestate(self):  

        #output: state for type B UAVs

        Phi = np.zeros(self.NUM_DRONES)
        GS = np.zeros(self.NUM_DRONES)
        theta=np.zeros(self.NUM_DRONES)
        Ve= np.zeros(self.NUM_DRONES)
        for j in range(self.NUM_DRONES):
            if self.vel[j,0]==0:
               Phi[j]==0
               GS[j] = math.sqrt(math.pow(self.vel[j,0],2)+math.pow(self.vel[j,1],2))
               Ve[j] = math.sqrt(math.pow(self.vel[j,1],2)+math.pow(self.vel[j,2],2))
            elif self.vel[j,1]==0:
               Phi[j]==np.pi*0.5
               GS[j] = math.sqrt(math.pow(self.vel[j,0],2)+math.pow(self.vel[j,1],2))
               Ve[j] = math.sqrt(math.pow(self.vel[j,1],2)+math.pow(self.vel[j,2],2))
            else:
               Phi[j]= math.atan(self.vel[j,0]/self.vel[j,1])
               GS[j] = math.sqrt(math.pow(self.vel[j,0],2)+math.pow(self.vel[j,1],2))
               Ve[j] = math.sqrt(math.pow(self.vel[j,1],2)+math.pow(self.vel[j,2],2))
        for j in range(self.NUM_DRONES):
            if self.vel[j,2]==0:
               theta[j]==0
            if self.vel[j,1]==0:
               theta[j]==math.pi
            else:
               theta[j]= math.atan(self.vel[j,2]/self.vel[j,1])

        return GS,Phi,theta,Ve

    def _collisiondetection(self):

        #ID start 1
        #Detection collision
        P_min, P_max = p.getAABB((self.typeA))
        id_tuple = p.getOverlappingObjects(P_min, P_max)
        if id_tuple==None:
            self.reward_collision=-1
            self.Done=True
            return P_max,self.reward_collision
        if len(id_tuple) > 1:
            for ID, _ in id_tuple:
                if ID == (self.typeA):
                    continue
                else:
                    print(f"UAV hits the obstacle {p.getBodyInfo(ID)}")
                    self.reward_collision=-1
                    self.Done=True
                    break
        return P_max, self.reward_collision


    def _collisionforA(self,GS,Phi,theta,Ve):
        #def the detection/punishment area
        P_max, reward_collision=self._collisiondetection()
        r=24.0 #maximum horizontal speed for target UAV*time
        h=3.5 #maximum vertical speed for target UAV*time
        min_collision=1E15 
        d_collisions=[]
        j_min=-1
        reward_c=-1
        ADSB_info = np.zeros((self.NUM_DRONES,6))
        ADSB_noise=[]
        frequency=48 #Set the frequency of ADS-B signal e.g. 2,4,6,8,12,24,48
        time_step=48/frequency
        for j in range(self.NUM_DRONES):
            ADSB_info[j,0:3]=self.pos[j]
            ADSB_info[j,3]=Phi[j]
            ADSB_info[j,4]=GS[j]
            ADSB_info[j,5]=self.vel[j,2]
        step=int(self.step_counter / self.AGGR_PHY_STEPS)
        ADSB_noise=self.adsbserver.addnoise(ADSB_info) #add error for ADS-B signal

        if (step==0):
            self.adsbserver.SendToadsb(ADSB_noise,step) #Encode the ADS-B information
            ADSB_info=self.adsbserver.ReceiveFromadsb(step) #Decode the ADS-B signal
            ADSB_info=self.adsbserver.filterdata(ADSB_info,step) #send to Kalman filter
            self.ADSB_unchange=ADSB_info #save the information for this step
        elif (step % time_step==0):
            self.adsbserver.SendToadsb(ADSB_noise,step)
            ADSB_info=self.adsbserver.ReceiveFromadsb(step)
            ADSB_info=self.adsbserver.filterdata(ADSB_info,step)
            self.ADSB_unchange=ADSB_info
        else:
            ADSB_info=self.ADSB_unchange

        for j in range(self.typeB):
                #horizontal velocity and vertical velocity
                v_rh=math.sqrt(math.pow(ADSB_info[j,4],2)+math.pow(ADSB_info[self.typeA-1,4],2)-2*abs(ADSB_info[j,4])*abs(ADSB_info[self.typeA-1,4])*math.cos(ADSB_info[j,3]-ADSB_info[self.typeA-1,3]))
                v_ve=math.sqrt(math.pow(ADSB_info[j,5],2)+math.pow(ADSB_info[self.typeA-1,5],2)-2*abs(ADSB_info[j,5])*abs(ADSB_info[self.typeA-1,5])*math.cos(theta[j]-theta[(self.typeA-1)]))
                d_h=np.linalg.norm(ADSB_info[j,:3]-ADSB_info[self.typeA-1,:3],ord=2)
                V_rh=max(0.00001,v_rh)
                V_ve=max(0.00001,v_ve)
                d_v=np.linalg.norm([ADSB_info[j,1]-ADSB_info[self.typeA-1,1],ADSB_info[j,2]-ADSB_info[self.typeA-1,2]])
                t_collision_h=d_h/V_rh
                t_collision_v=d_v/V_ve
                #determine the collision time
                t_collision=min(t_collision_h,t_collision_v)
                cloest_distance=min(d_h,d_v)
                d_collisions.append(cloest_distance)
                if min_collision>t_collision:
                    min_collision=t_collision
                    j_min=j #feedback the cloest UAV
                    if (abs(ADSB_info[j,0]-ADSB_info[self.typeA-1,0])**2+abs(ADSB_info[j,1]-ADSB_info[self.typeA-1,1])**2)<=r**2 or \
                       (ADSB_info[self.typeA-1,2]+h)>ADSB_info[j,2] or \
                       (ADSB_info[self.typeA-1,2]-h)<ADSB_info[j,2]:
                       reward_c=0
                    else:
                       dpmin=math.sqrt(d_h**2+d_v**2)
                       reward_c=np.tanh((np.linalg.norm(dpmin,ord=2)-r)/r)
        nums = list(map(float, d_collisions))
        self.cloest_d_collision = nums[0]
 
        for i in range(len(nums)):
            if nums[i] < self.cloest_d_collision:
               self.cloest_d_collision = nums[i]
        self.mean_d_collision=sum(d_collisions)/len(d_collisions)
        self.reward_c=reward_c
        return reward_c, j_min, ADSB_info, ADSB_noise


    def _ProcessADSB(self):
        GS,Phi,theta,Ve=self._calculatestate()
        j_min,reward_c,ADSB_info,ADSB_noise=self._collisionforA(GS,Phi,theta,Ve)
        #output for type A UAV
        rela_d_h=np.linalg.norm(ADSB_info[self.typeA-1,:3]-ADSB_info[j_min,:3],ord=2)
        rela_d_v=ADSB_info[self.typeA-1, 2]-ADSB_info[j_min,2]
        rela_v_h=math.sqrt(math.pow(ADSB_info[j_min,4],2)+math.pow(ADSB_info[self.typeA-1,4],2)-2*abs(ADSB_info[j_min,4])*abs(ADSB_info[self.typeA-1,4])*math.cos(ADSB_info[j_min,3]-ADSB_info[self.typeA-1,3]))
        rela_v_v=self.vel[self.typeA-1,2]-ADSB_info[j_min,5]
        rpR=np.arctan(abs(ADSB_info[j_min,0]-ADSB_info[self.typeA-1,0])/abs(ADSB_info[j_min,1]-ADSB_info[self.typeA-1,1]))
        TRR=np.arctan(rela_d_v/rela_d_h)
        state=np.hstack((ADSB_info[self.typeA-1,:3],self.rpy[(self.typeA-1), :],ADSB_info[self.typeA-1,4]*math.cos(ADSB_info[self.typeA-1,3]),ADSB_info[self.typeA-1,4]*math.sin(ADSB_info[self.typeA-1,3]),ADSB_info[self.typeA-1,5],rela_d_h,rela_d_v,rela_v_h,rela_v_v,rpR,TRR))
        self.ADSB_noise=ADSB_noise
        self.ADSB_info=ADSB_info
        return state.reshape(15,)

    def receiveinfo(self):

        return self.ADSB_info, self.ADSB_noise,self.typeA,self.NUM_DRONES,self.pos,self.mean_d_collision,self.cloest_d_collision,self.epsilon,self.reward_t

        
    def _trajectoryTrackingRPMsforA(self):
        """Computes the RPMs values to target a hardcoded trajectory.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        
        ####
        Phi_diff=-1.0
        Error=1
        eta=3
        k=0.8
        state_a = self._getDroneStateVector(self.typeA-1)
        i = min(int(self.step_counter / self.AGGR_PHY_STEPS), self.TRAJ_STEPS - 1)
        rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP, 
                                             cur_pos=state_a[0:3],
                                             cur_quat=state_a[3:7],
                                             cur_vel=state_a[10:13],
                                             cur_ang_vel=state_a[13:16],
                                             target_pos=self.TARGET_POSITION[i, :],
                                             target_vel=self.TARGET_VELOCITY[i, :]
                                             )
        #For path error
        self.epsilon=[self.TARGET_POSITION[i, 0]-state_a[0],self.TARGET_POSITION[i, 1]-state_a[1],self.TARGET_POSITION[i, 2]-state_a[2]]
        self.reward_p=max(Phi_diff,k*np.tanh((Error-abs(self.epsilon[0]))*eta))+max(Phi_diff,k*np.tanh((Error-abs(self.epsilon[1]))*eta))+max(Phi_diff,k*np.tanh((Error-abs(self.epsilon[2]))*eta)) 
        return rpm,self.reward_p


    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Clips and converts a dictionary into a 2D array.

        Parameters
        ----------
        action : dict[str, ndarray]
            The (unbounded) input action for each drone, to be translated into feasible RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        action_b = {str(i): np.array([0.0,0.0,0.0,0.0]) for i in range(self.typeB)}
        actiona=action
        if self.ACT_TYPE == ActionType.TUN:
            self.ctrl.setPIDCoefficients(p_coeff_pos=(actiona[0]+1)*self.TUNED_P_POS,
                                         i_coeff_pos=(actiona[1]+1)*self.TUNED_I_POS,
                                         d_coeff_pos=(actiona[2]+1)*self.TUNED_D_POS,
                                         p_coeff_att=(actiona[3]+1)*self.TUNED_P_ATT,
                                         i_coeff_att=(actiona[4]+1)*self.TUNED_I_ATT,
                                         d_coeff_att=(actiona[5]+1)*self.TUNED_D_ATT
                                         )
            for j in range(self.typeB):
                state_b= self._getDroneStateVector(j)
                i = min(int(self.step_counter / self.AGGR_PHY_STEPS), self.TRAJ_STEPS - 1)
                action_b[str(j)], _, _ =self.ctrl_b[j].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                   cur_pos=state_b[0:3],
                                   cur_quat=state_b[3:7],
                                   cur_vel=state_b[10:13],
                                   cur_ang_vel=state_b[13:16],
                                   target_pos=self.TARGET_POS_B[j, i,:],
                                   target_rpy=self.INIT_RPYSforB[j, :]
                                   )
        rpm_a,self.reward_p=self._trajectoryTrackingRPMsforA()
        clipped_action = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            if int(k) <self.NUM_DRONES-1:
                  clipped_action[int(k), :] = action_b[str(k)]
            else:
                  clipped_action[int(k), :] = rpm_a
        return clipped_action
        

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.
        """

        if self.Done:
            reward_t=self.reward_p+self.reward_collision 
        else:
            reward_t=0.5*self.reward_p+0.5*self.reward_c
        self.reward_t=reward_t       
        return  reward_t

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        if self.Done or self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            self._pathforBtype()
            self.Done=False
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} 

    # def _addObstacles(self):
    #   super()._addObstacles()
    #   p.loadURDF("cube_small.urdf",
    #                 [13.9, 9.8, 8.8],
    #                p.getQuaternionFromEuler([0, 0, 0]),
    #                physicsClientId=self.CLIENT)
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        MAX_LIN_VEL_XY=10.0
        MAX_LIN_VEL_Z=2.0

        MAX_XY=MAX_LIN_VEL_XY*self.PERIOD
        MAX_Z=MAX_LIN_VEL_Z*self.PERIOD
        MAX_PITCH_ROLL = np.pi # Full range

        MAX_RELAV_H=14.0
        MAX_RELAV_V=2.0
        MAX_RELAD_H=MAX_RELAV_H*self.PERIOD
        MAX_RELAD_V=MAX_RELAV_V*self.PERIOD
        MAX_PITCH_ROLL_R = np.pi # Full range
        
        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[3:5], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[6:8], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[8], 0, MAX_LIN_VEL_Z)
        clipped_pos_relaD_H=np.clip(state[9], 0, MAX_RELAD_H)
        clipped_pos_relaD_V=np.clip(state[10], -MAX_RELAD_V, MAX_RELAD_V)
        clipped_vel_xyR=np.clip(state[11], 0, MAX_RELAV_H)
        clipped_vel_zR=np.clip(state[12], -MAX_RELAV_V, MAX_RELAV_V)
        clipped_rpR=np.clip(state[13],0, MAX_PITCH_ROLL_R)
        
        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[5] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_pos_relaD_H=clipped_pos_relaD_H/MAX_RELAD_H
        normalized_pos_relaD_V=clipped_pos_relaD_V/MAX_RELAD_V
        normalized_vel_xyR=clipped_vel_xyR/MAX_RELAV_H
        normalized_vel_zR=clipped_vel_zR/MAX_RELAV_V
        normalized_rpR=clipped_rpR/MAX_PITCH_ROLL_R
        normalized_TRR=state[14]/np.pi
        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_pos_relaD_H,
                                      normalized_pos_relaD_V,
                                      normalized_vel_xyR,
                                      normalized_vel_zR,
                                      normalized_rpR,
                                      normalized_TRR
                                      ]).reshape(15)

        return norm_and_clipped
        
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      clipped_pos_relaD_H,
                                      clipped_pos_relaD_V,
                                      clipped_vel_xyR,
                                      clipped_vel_zR,
                                      clipped_rpR,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[3:5])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[3], state[5]))
        if not(clipped_vel_xy == np.array(state[6:8])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[6], state[7]))
        if not(clipped_vel_z == np.array(state[8])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[8]))
        if not(clipped_pos_relaD_H == np.array(state[9])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[9]))
        if not(clipped_pos_relaD_V == np.array(state[10])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[10]))
        if not(clipped_vel_xyR == np.array(state[11])).all():
             print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[11]))
        if not(clipped_vel_zR == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
        if not(clipped_rpR == np.array(state[13])).all():
             print("[WARNING] it", self.step_counter, "in ControlAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[13]))

            










