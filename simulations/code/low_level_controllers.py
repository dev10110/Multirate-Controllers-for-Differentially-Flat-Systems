import numpy as np
from utils import *

def flat_space_control(dt, X, X_ref, U_ref, P, K, Vmax):
    	
	e = X - X_ref

	U_e = - K @ e

	U = U_ref + U_e

	Lyapunov = 0.5 * e.T @ P @ e

	A = np.array([[1, 0, dt,  0], 
          [0, 1,  0, dt],
          [0, 0,  1,  0],
          [0, 0,  0,  1]])

	B = np.array([[ dt**2/2, 0], 
				[ 0, dt**2/2],
				[dt, 0],
				[ 0, dt]])

	X_next = A @ X + B @ U 

	v = np.linalg.norm(X_next[2:4,0])
	vx = X_next[2,0]
	vy = X_next[3,0]
	ax = U[0,0]
	ay = U[1,0]
	w = (-vy*ax + vx*ay)/v**2
	
	# limit the omega (handles singularities)
	if w > 3.0 : 
		print("FORCEFULLY LIMITING OMEGA from ", w)
		print(f"vx: {vx}, vy: {vy}, ax: {ax}, ay: {ay}")
		w = 3.0
		exit()
		
	if w < -3.0 :
		print("FORCEFULLY LIMITING OMEGA from ", w)
		w = -3.0
	
	return v, w, Lyapunov[0,0]
