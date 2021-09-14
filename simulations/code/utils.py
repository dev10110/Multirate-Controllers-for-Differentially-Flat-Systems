import numpy as np

def wrap_angle(angle):
	
	if angle>np.pi:
		angle -= 2*np.pi
		return wrap_angle(angle)
	if angle<=-np.pi:
		angle += 2*np.pi
		return wrap_angle(angle)
	return angle

# wrap an angle to between 0 and 2pi
def wrap_angle_2Pi(angle):
	if angle >= 2*np.pi:
		angle -= 2*np.pi
		return wrap_angle_2Pi(angle)
	if angle < 0 :
		angle += 2*np.pi
		return wrap_angle_2Pi(angle)
	return angle