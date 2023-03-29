from adsb.sbs import message
from pyproj import Proj
from pyproj import Transformer
from cmath import sqrt
import math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from pyproj import CRS
import numpy as np
import string
# from gym_pybullet_drones.envs.ControlAviary import ControlAviary
utm=Proj("+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs ")
#EPSG:4326
    # # See :class:`MessageType`
    # message_type = 0
    # # See :class:`TransmissionType`
    # transmission_type = 1
    # # SBS Database session record number.
    # session_id = 2
    # # SBS Database aircraft record number.
    # aircraft_id = 3
    # # 24-bit ICAO ID in hex.
    # hex_ident = 4
    # # SBS Database flight record number.
    # flight_id = 5
    # # Date the message was generated.
    # generated_date = 6
    # # Time the message was generated.
    # generated_time = 7
    # # Date the message was logged by SBS
    # logged_date = 8
    # # Time the message was logged by SBS.
    # logged_time = 9
    # # Eight character flight ID or callsign.
    # callsign = 10
    # # Altitude (ft) relative to 1013 mb (29.92" Hg).
    # altitude = 11
    # # Speed over ground (kt)
    # ground_speed = 12
    # # ground heading
    # track = 13
    # # Latitude in degrees
    # lat = 14
    # # Longitude in degrees
    # lon = 15
    # # Rate of climb
    # vertical_rate = 16
    # # Squawk
    # squawk = 17
    # # Squawk flag - indicating squawk has changed.
    # alert = 18
    # # Squawk flag indicating emergency code has been set.
    # emergency = 19
    # # Flag indicating the Special Position Indicator has been set.
    # spi = 20
    # # Flag indicating whether aircraft is on the ground
    # is_on_ground = 21
class adsbcenter():
    def __init__(self,num_drones: int=50,traj_steps:int=1):
        D=3
        N=traj_steps+5
        self.msglist=[[0]*N for _ in range(num_drones)]
        self.NUMDRONES=num_drones
        self.TRAJ_STEPS=traj_steps
        self.TEST_MSG_STR = "MSG,3,1,1,7C79B7,1,2017/03/25,10:41:45.365,2017/03/25,10:41:45.384,,2850,,,-34.84658,138.67962,,,,,,"
        #SBSMessage(message_type='MSG', transmission_type=3, session_id=1, aircraft_id=1, hex_ident='7C79B7', \
        # flight_id=1, generated_date=datetime.date(2017, 3, 25), generated_time=datetime.time(10, 41, 45, 365000), \
        # logged_date=datetime.date(2017, 3, 25), logged_time=datetime.time(10, 41, 45, 384000), callsign=None, \
        # altitude=2850, ground_speed=None, track=None, lat=-34.84658, lon=138.67962, vertical_rate=None, \
        # squawk=None, alert=None, emergency=None, spi=None, is_on_ground=None)
        self.MSGARRAY=self.TEST_MSG_STR.split(",")
        #Define filters, only for position variables, also extendable to velocity and acceleration, one filter per drone
        self.myfilter=[]
        for i in range(num_drones):
            self.myfilter.append(KalmanFilter(dim_x=3,dim_z=3))
            #Initial value
            self.myfilter[i].x=np.zeros([D,N])
            #Filtered data
            self.myfilter[i].z=np.zeros([D,N])
            #State transfer matrix, no conversion to the previous state, unit matrix
            self.myfilter[i].F=np.eye(D)
            #The covariance matrix, assuming no relationship between velocity and position, is taken as the unit matrix
            self.myfilter[i].P=np.eye(D) 
            #Process noise covariance matrix, estimated values
            self.myfilter[i].Q=1e-2*np.eye(D)
            #Measurement noise covariance matrix
            self.myfilter[i].R=[[0.5102,0,0],[0,4.5918,0],[0,0,7.62]]
            #Observation Matrix
            self.myfilter[i].H=np.eye(D)
       

    #send from controlaviary 
    def SendToadsb(self,adsbData,step):
        #msglist[][]Two-dimensional arrays，self.NUMDRONES-1:msg
        for i in range(self.NUMDRONES):
            
            lon,lat=self.convertlong(adsbData[i,0]+527978.0745555542,adsbData[i,1]+5700200.643251519)
            self.MSGARRAY[14]=str(lon)#Latitude in degrees
            self.MSGARRAY[15]=str(lat)#Longitude in degrees
            self.MSGARRAY[5]=str(i)#flight id
            self.MSGARRAY[17]=str(i)#Squawk
            self.MSGARRAY[11]=str(adsbData[i,2])#altitude(m) 
            self.MSGARRAY[12]=str(adsbData[i,4])#Speed over ground (m/s)
            self.MSGARRAY[16]=str(adsbData[i,5])#Rate of climb
            self.MSGARRAY[13]=str(adsbData[i,3])#ground heading
            msg_data = ",".join(self.MSGARRAY)
            # msg = message.fromString(msg_data)
            self.msglist[i][step]=msg_data
            # print(msg_data.aircraft_id)
        

    def ReceiveFromadsb(self,step):
        msg_data=[[0]*6 for _ in range(self.NUMDRONES)]
        Phi = np.zeros(self.NUMDRONES)
        GS = np.zeros(self.NUMDRONES)
        pos_x=np.zeros(self.NUMDRONES)
        pos_y= np.zeros(self.NUMDRONES)
        pos_z= np.zeros(self.NUMDRONES)
        vel = np.zeros(self.NUMDRONES)
        ADSB_info=np.zeros([self.NUMDRONES,6])
        for i in range(self.NUMDRONES):
            msg=self.msglist[i][step]
            recovered_msg_data = msg#message.toString(msg)
            # print(recovered_msg_data)
            self.MSGARRAY= recovered_msg_data.split(",")
            GS[i]=(self.MSGARRAY[12])
            Phi[i]=(self.MSGARRAY[13])
            pos_x[i],pos_y[i]=self.convertXY(self.MSGARRAY[14],self.MSGARRAY[15])
            pos_z[i]=(self.MSGARRAY[11])
            vel[i]=(self.MSGARRAY[16])
            ADSB_info[i,0]=pos_x[i]-527978.0745555542
            ADSB_info[i,1]=pos_y[i]-5700200.643251519
            ADSB_info[i,2]=pos_z[i]
            ADSB_info[i,3]=Phi[i]
            ADSB_info[i,4]=GS[i]
            ADSB_info[i,5]=vel[i]
            #msg_data[i][0]=msg.aircraft_id
        return ADSB_info

    def convertlong(self,X,Y):
        #3.Convert  XY coordinate to WGS84 coordinate
        lon, lat =utm(X,Y,inverse=True)
        #4.wgs84-30N-UTM
        X2, Y2 =utm(lon, lat)
        return lon, lat
    def convertXY(self,lon,lat):
        #3.Convert WGS84 to XY coordinate
        X, Y =utm(lon,lat,inverse=False)
        return X, Y

    def get_normal_random_number(self,loc, scale):
            """
            :param loc: mean number
            :param scale: Standard deviation of normal distribution
            :return:Random numbers generated from a normal distribution
             """
            # generate random number in normal distribution
            number = np.random.normal(loc=loc, scale=scale)
            return number
    
    def addnoise(self,adsbData):
        #add noise to adsb info
        Error_vh = np.zeros(self.NUMDRONES)
        Error_z = np.zeros(self.NUMDRONES)
        Error_h = np.zeros(self.NUMDRONES)
        error_h = np.zeros(self.NUMDRONES)
        GS_r=np.zeros(self.NUMDRONES)
        x_r=np.zeros(self.NUMDRONES)
        y_r=np.zeros(self.NUMDRONES)
        z_r=np.zeros(self.NUMDRONES)
        for j in range(self.NUMDRONES):
        #Simulate the error of the ADS-B info
                Error_vh[j] =self.get_normal_random_number(loc=-0.2, scale=2.78)
                Error_z[j]=self.get_normal_random_number(loc=-3.5,scale=11.16)
                if Error_z[j]>0:
                   Error_z[j]=min(Error_z[j],7.62)
                else:
                   Error_z[j]=max(Error_z[j],-7.62)
                Error_h[j]=self.get_normal_random_number(loc=0,scale=5.102)
                GS_r[j]=adsbData[j,4]+Error_vh[j]
                error_h[j]=abs(Error_h[j])
                if error_h[j]>Error_h[j]:
                   x_r[j]=adsbData[j,0]-0.1*sqrt(error_h[j])
                   y_r[j]=adsbData[j,1]-0.9*sqrt(error_h[j])
                else:
                   x_r[j]=adsbData[j,0]+0.1*sqrt(error_h[j])
                   y_r[j]=adsbData[j,1]+0.9*sqrt(error_h[j])
                z_r[j]=adsbData[j,2]+Error_z[j]
                adsbData[j,0]=x_r[j]
                adsbData[j,1]=y_r[j]
                adsbData[j,2]=z_r[j]
                adsbData[j,4]=GS_r[j]
        return adsbData

    def filterdata(self,adsbData,step):
         #updated every step
         #（x,y,z,phi,gs,vel）
        for j in range(self.NUMDRONES):
            self.myfilter[j].predict()
            #The parameter is the value of the step observation, with random error
            znoise=adsbData[j,:3]
            self.myfilter[j].update(znoise)
            #size:j-3
            #return to receive adsbdata
            adsbData[j,0]=self.myfilter[j].x[0,step]
            adsbData[j,1]=self.myfilter[j].x[1,step]
            adsbData[j,2]=self.myfilter[j].x[2,step]
        return adsbData
   
if __name__ == "__main__":
    adsbcenter=adsbcenter(3)
    # ADSB_info=np.zeros((3,2))
    # for j in range(3):
    #     ADSB_info[j,0]=-2.60376
    #     ADSB_info[j,1]=51.4585712

    # adsbcenter.SendToadsb(ADSB_info)
    # adsbcenter.ReceiveFromadsb()
    lon=-2.5973512
    lat=51.452295
    X,Y=adsbcenter.convertXY(lon,lat)
    print (X,Y)
    lon,lat=adsbcenter.convertlong(X,Y)
    print (lon,lat)