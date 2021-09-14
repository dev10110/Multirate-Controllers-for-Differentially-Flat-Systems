import cvxpy
import numpy as np
from cvxpy import *

class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
    Methods:
    - solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
    - model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

    """
    def __init__(self, N, goal, A, B, Q, R, Qf, bVecList, params, printLevel = 0,  mpcTimestep=1.0, amax=1.0):
        # Define variables
        self.N = N # Horizon Length

        # System Dynamics (x_{k+1} = A x_k + Bu_k)
        self.A = A 
        self.B = B 
        self.n = A.shape[1]
        self.d = B.shape[1]

        # Cost (h(x,u) = x^TQx +u^TRu)
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.goal = goal

        # Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []

        self.F = np.vstack((np.eye(2), -np.eye(2)))

        self.bMatrix = np.array(bVecList).T
        self.numReg  = self.bMatrix.shape[1]

        self.printLevel = printLevel

        self.mpcTimestep = mpcTimestep

        self.P = params["P"]
        self.Vmax = params["Vmax"]
        self.w = params["w"]
        self.velMax = params["velMax"]


    def build(self):#, regVar=None):
        
        # This method build the problems  

        # Initialize Variables
        self.x = Variable((self.n, self.N+1))
        self.u = Variable((self.d, self.N))
        self.regVar = Variable((self.numReg, self.N+1), boolean=True) # Initialize vector of variables
        self.x0 = Parameter(self.n)
        
        # Initial condition constraint
        constr = [ 0.5 * cvxpy.quad_form(self.x[:,0]-self.x0, self.P) <= self.Vmax ]

        # Final condition constraint
        constr += [self.x[:,-1] == self.goal]

        T = self.mpcTimestep
        contA = np.array([[0, 0, 1, 0],[0, 0 ,0 ,1 ], [0,0,0,0] , [0,0,0,0] ])
        contB = np.array([[0, 0],[0, 0], [1,0],[0,1]])

        for i in range(0, self.N):
            # dynamics constraints
            constr += [self.x[:,i+1] == self.A @ self.x[:,i] + self.B @ self.u[:,i]]

            # force the trajectory intrasample to be also within 
            bMax = self.bMatrix@self.regVar[:, i]

            state = self.x[:,i]
            control = self.u[:,i]

            # first boundary (x < bmax[0])
            a = np.array([[1,0,0,0]])
            b = bMax[0]

            h = a @ state - b
            hdot = a @ (contA @ state + contB @ control)
            hddot = a @ contA @ contB @ control
            constr += [h + hdot * T + 0.5 * cvxpy.maximum(hddot, 0) * T**2 <= 0.0 ]

            # second boundary (y < bmax[1])
            a = np.array([[0,1,0,0]])
            b = bMax[1]

            h = a @ state - b
            hdot = a @ (contA @ state + contB @ control)
            hddot = a @ contA @ contB @ control
            constr += [h + hdot * T + 0.5 * cvxpy.maximum(hddot, 0) * T**2 <= 0.0 ]

            # third boundary (x  >= -bMax[2]) = (-x <= bMax[2])
            a = np.array([[-1,0,0,0]])
            b = bMax[2]

            h = a @ state - b
            hdot = a @ (contA @ state + contB @ control)
            hddot = a @ contA @ contB @ control
            constr += [h + hdot * T + 0.5 * cvxpy.maximum(hddot, 0) * T**2 <= 0.0 ]

            # fourth boundary (y  >= -bMax[3]) = (-y <= bMax[3])
            a = np.array([[0,-1,0,0]])
            b = bMax[3]

            h = a @ state - b
            hdot = a @ (contA @ state + contB @ control)
            hddot = a @ contA @ contB @ control
            constr += [h + hdot * T + 0.5 * cvxpy.maximum(hddot, 0) * T**2 <= 0.0 ]

        for i in range(0, self.N+1):

            # force the state to lie in the chosen box
            bMax = self.bMatrix@self.regVar[:, i]
            constr+= [self.x[0,i]  <= bMax[0] ]
            constr+= [self.x[1,i]  <= bMax[1] ]
            constr+= [self.x[0,i]  >= -bMax[2] ]
            constr+= [self.x[1,i]  >= -bMax[3]]

            # force it to be in exactly one box
            constr += [sum(self.regVar[:, i]) == 1]
            
            # speed limits
            constr += [cvxpy.abs(self.x[2,i]) <= self.velMax]
            constr += [cvxpy.abs(self.x[3,i]) <= self.velMax]

        # Cost Function
        cost = 0
        for i in range(0, self.N):
            # Running cost h(x,u) = x^TQx + u^TRu
            cost += quad_form(self.x[:,i]-self.goal, self.Q) + quad_form(self.u[:,i], self.R)
        
        cost += quad_form(self.x[:,self.N]-self.goal, self.Qf)

        # Build the Finite Time Optimal Control Problem
        self.problem = Problem(Minimize(cost), constr)
        

    def solve(self, x0):
        # This method solves an FTOCP given: - x0: initial condition

        self.x0.value = x0
        
        self.problem.solve(solver=cvxpy.GUROBI,verbose=False, warmstart=True)

        # check optimality
        if not (self.problem.status == 'optimal'):
            print("****** MPC failed ******")
            return False
        
        self.xPred = self.x.value
        self.uPred = self.u.value

        # print("P IN FTOCP", self.P)
        # print("e in FTOCP", x0 - self.xPred[:,0])
        # print(self.xPred[:,0])
        # print(x0)
        if 0.5 * (self.xPred[:,0]-x0).T @ self.P @ (self.xPred[:,0]-x0) >= self.Vmax + 0.01: # the 0.01 is just for numerical tolerances
            print(0.5 * (self.xPred[:,0]-x0).T @ self.P @ (self.xPred[:,0]-x0))
            print(self.Vmax)
            raise "NOT CHOSEN CORRECTLY"

        if self.printLevel >= 2: 
            print("xPred.T")
            print(self.xPred.T)
            print("uPred.T")
            print(self.uPred.T)

        return True









	

