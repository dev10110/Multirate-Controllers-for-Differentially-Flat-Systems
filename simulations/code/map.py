import numpy as np
import pdb 
from polytope import polytope

class MAP(object):

	def __init__(self, recList, printLevel = 0):
		# Define variables
		self.printLevel = printLevel
		self.bVecList = []
		F = np.vstack((np.eye(2), -np.eye(2)))
		self.poliList = []
		for rec in recList:
			self.bVecList.append(self.rec2bVec(rec))
			self.poliList.append(polytope(F, self.bVecList[-1], printLevel = printLevel))


	def rec2bVec(self, rec):
		# The rectangles will be represented as Polytopes Fx <= b. Here computing the vector b
		xmin = rec[0]
		ymin = rec[1]

		xmax = rec[0]+rec[2]
		ymax = rec[1]+rec[3]
		
		bVec = [xmax, ymax, -xmin, -ymin]
		if (self.printLevel >= 1): print(bVec)
		return np.array(bVec).astype(float)


	def plotMap(self,linestyle='solid'):
		color =['b','r','k','b','r','k','b','r','k','b','r','k','b','r','k','b','r','k','b','r','k','b','r','k']
		color.append('k') 
		color = ['b' for p in self.poliList ]
		i = 0
		for poli in self.poliList:
			poli.plot2DPolytope(color[i],linestyle=linestyle)
			i+=1


	

