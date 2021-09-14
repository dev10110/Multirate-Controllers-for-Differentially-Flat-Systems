
import numpy as np
from utils import wrap_angle
from low_level_controllers import *


class Unicycle:

  def __init__(self, x0, y0, theta0, params = {}):
    self.time = 0.0
    self.X = np.array([x0, y0, theta0])

    self.times = [0.0]
    self.states = np.copy(self.X)
    self.controls_v = []
    self.controls_w = []
    self.controls_v_perturbed = []
    self.controls_w_perturbed = []
    self.lyapunov = []

    self.params = params

  def step(self, U, dt, d, enable_perturbations=False):
    
    v = U[0]
    w = U[1]
    
    #dynamics update
    if (enable_perturbations):
          # v_perturbed = v + (2*np.random.rand()-1)*d/np.sqrt(2)
          # w_perturbed = w + (2*np.random.rand()-1)*d/np.sqrt(2)
          v_perturbed = v + d/np.sqrt(2)*np.sin(2*np.pi*self.time)
          w_perturbed = w + d/np.sqrt(2)*np.cos(2*np.pi*self.time)
    else:
          v_perturbed = v
          w_perturbed = w
    f = np.array([ v_perturbed*np.cos(self.X[2]), v_perturbed*np.sin(self.X[2]), w_perturbed ])
    self.X = self.X + f*dt
    self.X[2] = wrap_angle(self.X[2])

    # save results
    self.time = self.time + dt
    self.times = np.append(self.times, self.time)
    self.states = np.vstack((self.states,self.X))
    self.controls_v.append(v)
    self.controls_w.append(w)
    self.controls_v_perturbed.append(v_perturbed)
    self.controls_w_perturbed.append(w_perturbed)

  def get_r(self, MPC_X_ref):
    
    x, y, theta = self.X
    x_ref, y_ref = MPC_X_ref[0:2,0]
  
    # check if reached
    r = np.sqrt((x-x_ref)**2 + (y-y_ref)**2)
    
    return r


  def low_level(self, MPC_dt, uni_dt, MPC_X_ref, MPC_U_ref, params):

    x, y, theta = self.X


    # estimate the flat state, using the last applied velocity
    flat_X = np.array([ x, y, self.controls_v[-1]*np.cos(theta), self.controls_v[-1]*np.sin(theta) ]).reshape(-1,1)
        
    # v, w = v_ref, w_ref
    v, w, Lyapunov = flat_space_control(uni_dt, flat_X, MPC_X_ref, MPC_U_ref, params["P"],  params["K_feedback"], params["Vmax"])
    
    self.lyapunov.append(Lyapunov)
    
    reached = False

    if v == None:
          reached = True

    return [v, w], reached



