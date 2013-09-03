import numpy as np

class CliqueTree():
	
	def __init__(self, cliqueList=None):
		self.cliqueList = cliqueList

	def create_clique_tree(self,factors,evid=None):
		""" Given a list of factors and evidence, constructs a clique tree

		Parameters
		----------
		factorList:	list of pgmFactor objects
		evid:	[Optional] numpy array (double bracketed)
				Nx2 matrix, variable and value for columns 1 and 2 respectively

		Returns
		--------
		None

		Additional
		----------
		Uses min-neighbor variable elimination to construct the clique tree
		
		"""
		# get the unique set of random variables from all the factors
		rvars = np.array([])
		for factor in factors:
			self.rvars = np.union1d(factor.scope,self.rvars)

		# initialize and fill in the adjacency matrix
		numVars = np.size(rvars)
		edgeMat = np.array([numVars,numVars])

		for rvar in rvars:
			

		


		




	
