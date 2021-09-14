# @brief: provides a method to solve the modified Ricatti Equation

import numpy as np
from scipy.linalg import solve_continuous_are

def createLowLevelParams(A, B, Q, R, g, w):
  # A, B are control matrices
  # Q, R are cost weights
  # g = \gamma controls the sensitivity
  # w is the magnitude of the noise

  # determine size of the problem
  n = Q.shape[1]
    
  G = B @ np.linalg.inv(R) @ B.transpose() - (1.0/g**2) * np.eye(n)

  Bmod = np.eye(n)

  P = solve_continuous_are(A, Bmod, Q, np.linalg.inv(G))

  try:
    np.linalg.cholesky(P)
  except:
    raise "P is not positive definite! - choose a larger gamma?"
  
  K = 0.5 * np.linalg.inv(R) @ B.transpose() @ P
  # control input is u = - K @ x
  
  Vmax = 0.5 * g**2 * np.max(np.linalg.eigvals(P)) / np.min(np.linalg.eigvals(Q)) * w**2
  
  d_tighten = np.sqrt(2 * Vmax / np.min(np.linalg.eigvals(P)))
  
  return P, Vmax, d_tighten, K

def double_integrator_params(w, gamma, Q=np.eye(4), R = np.eye(2)):
  A = np.array([[0, 0, 1,  0], 
              [0, 0,  0, 1],
              [0, 0, 0,  0],
              [0, 0,  0,  0]])

  B = np.array([[ 0, 0], 
              [ 0, 0],
              [1, 0],
              [ 0, 1]])

  return createLowLevelParams(A, B, Q, R, gamma, w)