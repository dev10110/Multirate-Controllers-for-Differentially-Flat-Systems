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


def convertToPolar(x, y, theta, x_ref, y_ref, theta_ref):

    r = np.sqrt( (x_ref - x)**2 + (y_ref - y)**2 )
    targetHeading = np.arctan2(y_ref - y, x_ref - x)
    angleToTarget = wrap_angle(theta - targetHeading)
    psi = wrap_angle_2Pi(np.arctan2(y - y_ref, x-x_ref) - theta_ref)

    return [r, psi, angleToTarget]