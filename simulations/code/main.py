import numpy as np
import matplotlib.pyplot as plt

from ftocp import FTOCP
from map import MAP
from riccati_sols import double_integrator_params
from unicycle import Unicycle

import signal
import time
import traceback

# allow Ctrl-C to work despite plotting
signal.signal(signal.SIGINT, signal.SIG_DFL)


# ==============================================================================================
# USER DEFINED PARAMETERS
# ==============================================================================================

# set the noise level
w = 0.2
enable_perturbations = True

gamma = 1.9

# choose the Q and R for the low level controller design
Q = np.eye(4)
Q[2,2] = 2 # increased weights on vel
Q[3,3] = 2
R = np.eye(2)

P, Vmax, d_tighten, K = double_integrator_params(w, gamma, Q, R)

print("USING D_tighten = ", d_tighten)

robot_params = {"d": d_tighten, "w": w, "P": P, "Vmax": Vmax, "K_feedback": K, "velMax" : 2.0}

# choose MPC timestep and horizon
dt_mpc = 0.5
N = 25

# how long to simulate for
SIMULATE_T = N*dt_mpc # seconds

# this is the timestep for simulation/integration
dt_uni = 0.005

# dimension of state and control inputs
nx =4
nu= 2

# choose the cost for the MPC solution
Q = 0.5*np.eye(nx)
R = 10*np.eye(nu)
Qf = 20*N*np.eye(nx)


# set the start and goal
x0 = np.array([2, 2.,0.0,0.5])   # Position and velocity
theta_0 = np.arctan2(x0[3],x0[2]) # initial heading angle
goal = np.array([13.0, 2.0, 0.0, 0.0])


# construct the environment,
recList = []
# Each rec = [ x, y, width, height ]
recList.append([ 0,  0,  5, 7.5]) 
recList.append([10,  0,  5, 7.5])
recList.append([ 0,7.5, 15, 7.5])

mapEnv = MAP(recList, printLevel = 0)

# construct a tightened environment 
recListTight = []
recListTight.append([ 0,  0,  5-d_tighten, 7.5+d_tighten]) # Each rec = [ x, y, width, height ]
recListTight.append([10+d_tighten,  0,  5-d_tighten, 7.5+d_tighten])
recListTight.append([ 0,7.5+d_tighten, 15, 7.5-d_tighten])

mapTight = MAP(recListTight, printLevel=0)

plt.figure()
mapEnv.plotMap()
mapTight.plotMap(linestyle='dashed')
plt.scatter(x0[0], x0[1], marker='o')
plt.scatter(goal[0], goal[1], marker='o')


plt.savefig("figs/regions.eps")
plt.savefig("figs/regions.png")


# ==============================================================================================

# discrete time double integrator
A = np.array([[1, 0, dt_mpc,  0], 
	          [0, 1,  0, dt_mpc],
	          [0, 0,  1,  0],
	          [0, 0,  0,  1]])

B = np.array([[ dt_mpc**2/2, 0], 
	          [ 0, dt_mpc**2/2],
	          [dt_mpc, 0],
	          [ 0, dt_mpc]])

## discrete matrices for unicycle interp of lin_sys
A_di = np.array([[1, 0, dt_uni,  0], 
          [0, 1,  0, dt_uni],
          [0, 0,  1,  0],
          [0, 0,  0,  1]])

B_di = np.array([[ dt_uni**2/2, 0], 
	          [ 0, dt_uni**2/2],
	          [dt_uni, 0],
	          [ 0, dt_uni]])


##### Initialize Unicycle
robot = Unicycle(x0[0],x0[1],theta_0, robot_params)
robot.controls_v.append( np.linalg.norm(x0[2:4]))
robot.controls_w.append( 0 )
robot.controls_v_perturbed.append( np.linalg.norm(x0[2:4]))
robot.controls_w_perturbed.append( 0 )


###### INTIALISE THE FTOCP PROBLEM 
ftocp = FTOCP(N, goal, A, B, Q, R, Qf, mapTight.bVecList, robot_params, printLevel = 0, mpcTimestep=dt_mpc, amax=0.5)
ftocp.build()


def replanMPC(robot, ftocp):

    s_time = time.time()
        
    # use the last applied velocity as the estimate for the double integrator state
    vx = robot.controls_v[-1]*np.cos(robot.X[2])
    vy = robot.controls_v[-1]*np.sin(robot.X[2])

    # create the estimated double integrator state
    x0 = np.array([robot.X[0], robot.X[1], vx, vy ])

    s_time = time.time()
    if not ftocp.solve(x0): 
        raise "FTOCP DID NOT SOLVE SUCCESSFULLY!"

    print("Replan Time: ", time.time() - s_time)

    return


###### SOLVE THE FTOCP PROBLEM ONCE FIRST

replanMPC(robot, ftocp)

# Init Variables for plotting
Xref_uni = []
Yref_uni = []

t_MPC = [i*dt_mpc for i in range(N+1)]
x_MPC = ftocp.xPred[0,:] 
y_MPC = ftocp.xPred[1,:]
vx_MPC = ftocp.xPred[2,:]
vy_MPC = ftocp.xPred[3,:]
ax_MPC = ftocp.uPred[0,:]
ay_MPC = ftocp.uPred[1,:]

###### Save initial solution

plt.figure()
mapEnv.plotMap()
mapTight.plotMap(linestyle='dashed')
plt.plot(x_MPC, y_MPC, '--rx', label='Double Integrator MPC trajectory')
plt.scatter(x0[0], x0[1], marker='o')
plt.scatter(goal[0], goal[1], marker='o')
plt.savefig("figs/INITIAL_PATH_tracking.eps")
plt.savefig("figs/INITIAL_PATH_tracking.png")

plt.figure()
plt.subplot(211)
plt.plot(t_MPC, x_MPC)
plt.ylabel('x')

plt.subplot(212)
plt.plot(t_MPC, y_MPC)
plt.ylabel('y')
plt.xlabel('time')

plt.savefig("figs/Initial_MPC_xy.eps")
plt.savefig("figs/Initial_MPC_xy.png")

plt.figure()
plt.subplot(211)
plt.plot(t_MPC, vx_MPC)
plt.ylabel('vx')

plt.subplot(212)
plt.plot(t_MPC, vy_MPC)
plt.ylabel('vy')
plt.xlabel('time')

plt.savefig("figs/Initial_MPC_velocity.eps")
plt.savefig("figs/Initial_MPC_velocity.png")

plt.figure()
plt.subplot(211)
plt.plot(t_MPC[:-1], ax_MPC)
plt.ylabel('ax')

plt.subplot(212)
plt.plot(t_MPC[:-1], ay_MPC)
plt.ylabel('ay')
plt.xlabel('time')

plt.savefig("figs/Initial_MPC_accelerations.eps")
plt.savefig("figs/Initial_MPC_accelerations.png")



# ==============================================================================================
# MAIN SIMULATION LOOP
# ==============================================================================================




reached = False

MPC_x = [x0[0]]
MPC_y = [x0[1]]
MPC_vx = [x0[2]]
MPC_vy = [x0[3]]
MPC_ux = [ftocp.uPred[0,0]]
MPC_uy = [ftocp.uPred[1,0]]
MPC_x_current = [x0[0]]
MPC_y_current = [x0[1]]
MPC_t = [robot.time]


ref_index = 0
MPC_index = 0

last_replan_time = robot.time

X_ref = np.array([ftocp.xPred[:,0]]).T
U_ref = np.array([ftocp.uPred[0,0], ftocp.uPred[1,0] ]).reshape(-1,1)

MPC_x.append(X_ref[0,0])
MPC_y.append(X_ref[1,0])
MPC_t.append(robot.time)
MPC_ux.append(U_ref[0,0])
MPC_uy.append(U_ref[1,0])
MPC_vx.append(X_ref[2,0])
MPC_vy.append(X_ref[3,0])
ref_index += 1

print('================== Starting time loop')

try:
    
    # simulate forward in time
    while robot.time <= SIMULATE_T and np.linalg.norm(robot.X[0:2] - goal[0:2])>0.05:

        # check if need to replan
        if (robot.time >= last_replan_time + dt_mpc):
            
            replanMPC(robot, ftocp)

            last_replan_time = robot.time
            ref_time = robot.time
            
            X_ref = np.array([ftocp.xPred[:,0]]).T
            U_ref = np.array([ftocp.uPred[0,0], ftocp.uPred[1,0] ]).reshape(-1,1)
            
            MPC_t.append(robot.time)
            MPC_x.append(X_ref[0,0])
            MPC_y.append(X_ref[1,0])
            MPC_vx.append(X_ref[2,0])
            MPC_vy.append(X_ref[3,0])
            MPC_ux.append(U_ref[0,0])
            MPC_uy.append(U_ref[1,0])
        
        
        # regardless, compute low-level
        u, reached = robot.low_level(dt_mpc, dt_uni, X_ref, U_ref, robot_params)
        
        if reached:
            robot.step([0,0], dt_uni)
            print("EXITING HERE!")
            break
        robot.step(u, dt_uni, robot_params["w"], enable_perturbations=enable_perturbations)

        # propagate X_ref in time for tracking
        X_ref = A_di @ X_ref + B_di @ U_ref
        Xref_uni.append(X_ref[0,0])
        Yref_uni.append(X_ref[1,0])
        # exit()

except Exception as e:
    traceback.print_exc()
    print(e)
    exit()

print("Simulation Completed!")

# ==============================================================================================
# PLOTTING
# ==============================================================================================


plt.figure()
mapEnv.plotMap()
mapTight.plotMap(linestyle='dashed')
plt.plot([x0[0]], [x0[1]], 'o')
plt.plot([goal[0]], [goal[1]], 'x')
plt.plot(Xref_uni,Yref_uni,'k.', label = "Reference")
plt.plot(MPC_x, MPC_y, 'rx', label="Waypoints")
plt.plot(robot.states[:,0], robot.states[:,1],'g', label="Path")

plt.savefig("figs/map_tracking.eps")
plt.savefig("figs/map_tracking.png")

plt.figure()
plt.subplot(311)
plt.plot(MPC_t,MPC_x,'rx',label='MPC Waypoints')
plt.plot(robot.times[:-1],Xref_uni, '.-', label='Tracking Waypoints')
plt.axhline([goal[0]])
plt.ylabel('X')
plt.legend()

plt.subplot(312)
plt.plot(MPC_t,MPC_y,'rx',label='MPC Waypoints')
plt.plot(robot.times[:-1],Yref_uni, '.-', label='Tracking Waypoints')
plt.ylabel('Y')
plt.axhline([goal[1]])
plt.xlabel('time')
plt.legend()

plt.subplot(313)
plt.plot(robot.times[1:], robot.lyapunov)
plt.axhline(Vmax, linestyle='--', label = 'Max Lyapunov')
plt.xlabel('time')
plt.ylabel("Lyapunov Function")

plt.savefig("figs/Tracking reference.eps")
plt.savefig("figs/Tracking reference.png")

plt.figure()
plt.plot(robot.times[1:], robot.lyapunov)
plt.axhline(Vmax, linestyle='--', label = 'Max Lyapunov')
plt.xlabel('time')
plt.ylabel("Lyapunov Function")

plt.savefig("figs/lyapunov.eps")
plt.savefig("figs/lyapunov.png")

robot.times = list(robot.times)

plt.figure()
plt.subplot(211)
plt.plot(robot.times, robot.controls_v_perturbed,'r')
plt.plot(robot.times, robot.controls_v,'g')
plt.ylabel("Linear Velocity")
plt.subplot(212)
plt.plot(robot.times[1:], robot.controls_w_perturbed[1:])
plt.plot(robot.times, robot.controls_w,'g')
plt.ylabel("Angular velocity")
# plt.subplot(313)
# plt.plot(robot.times[:-1], r_plot)
# plt.xlabel('time')
# plt.ylabel("radial dist")

plt.savefig("figs/controls_tracking.eps")
plt.savefig("figs/controls_tracking.png")

vx = []
vy = []
for index, value in enumerate(robot.controls_v):
    vx.append(robot.controls_v[index]*np.cos(robot.states[index][2]) )   # V*cos(theta)
    vy.append(robot.controls_v[index]*np.sin(robot.states[index][2]) )

plt.figure()
plt.subplot(211)
plt.plot(robot.times, vx,label='vx')
plt.step(MPC_t,MPC_vx,label='vx_ref',where='post')
plt.ylabel("X velocity")
plt.legend()
plt.subplot(212)
plt.plot(robot.times, vy,label='vy')
plt.step(MPC_t,MPC_vy,label='vy_ref',where='post')
plt.ylabel("Y velocity")
plt.xlabel('time')
plt.legend()
plt.savefig("figs/velocity_xy_tracking.eps")
plt.savefig("figs/velocity_xy_tracking.png")

plt.figure()
plt.subplot(211)
plt.plot(MPC_t,MPC_ux,label='ax')
plt.ylabel("X acceleration")

plt.subplot(212)
plt.plot(MPC_t,MPC_uy,label='ay')
plt.ylabel("Y acceleration")
plt.savefig("figs/acceleration MPC.eps")
plt.savefig("figs/acceleration MPC.png")
