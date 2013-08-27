import numpy as np
import pgmFactor as pgmf
import copy

class PGM():

	def __init__(self,factors):
		"""	Initializes the probabilistic graphical models with a list 
			of factors, as well as a mapping from variables to factors
		"""
		self.factors = factors 		# list of factors
		self.rvarToFactor = {}		# dictionary mapping variables to factors
		self.rvars = [] 			# list of all the variables
		
		for factor in factors:
			self.rvars = np.union1d(factor.scope,self.rvars)
			for rvar in factor.scope:
				if not rvar in self.rvarToFactor:
					self.rvarToFactor[rvar] = [factor]
				else:
					(self.rvarToFactor[rvar]).append(factor)

	def add_factor(self,factor):
		""" Adds a factor to the graph and updates the rvarToFactor map

			Parameters
			----------
			factor:	pgmFactor
					factor to add the PGM

			Returns
			-------
			None

		"""
		self.factors.append(factor)
		self.rvars = np.union1d(factor.scope,self.rvars)
		for rvar in factor.scope:
			if not rvar in self.rvarToFactor:
				self.rvarToFactor[rvar] = factor
			else:
				(self.rvarToFactor[rvar]).append(factor)

	def compute_marginal(self,mvars,evid=None):
		""" Computes the marginal of the joint distribution induced by the graph, 
			given the variables to marginalize out, and evidence if there is any

			Parameters
			----------
			mvars:	int or numpy array
					The variables which we would like to keep in our marginal
			evid:	[Optional] numpy array (double bracketed)
					Nx2 matrix, variable and value for columns 1 and 2 respectively

			Returns
			-------
			The marginal factor: pgmFactor object

			Additional
			----------
			Current implementation eliminates variables in an arbitrary order, which 
			may or may not be the optimal ordering. Heuristics used in attempting to
			determine the optimal ordering will be implemented in the future.
		"""
		# reduce and marginalize out the evidence from all the factors before computing
		# the marginal
		
		elvars = np.array([])
		if evid==None:
			elvars = np.setdiff1d(self.rvars,mvars) 
		else:
			evvars = evid[:,0]
			elvars = np.setdiff1d(self.rvars,np.concatenate((mvars,evvars))) # W = {X_1,..X_n} - Y - E
			for idx,rvar in enumerate(evid[:,0]):
				for factor in self.rvarToFactor[rvar]:
					factor.reduce(np.array([evid[idx,:]]))
					factor.marginalize(rvar)

		for elvar in elvars:	# for each variable we wish to eliminate...

			# compute the factor product of all factors that contain this
			# variable in its scope

			currFactProd = self.compute_factorlist_product(self.rvarToFactor[elvar])
			currFactProd.marginalize(elvar)
			# update the rvar to factor map
			self.rvarToFactor.pop(elvar,None)	# remove variable from dictionary

			for rvar in currFactProd.scope:
				newList = []				
				for factor in self.rvarToFactor[rvar]:
					if not elvar in factor.scope:
						newList.append(factor)
				newList.append(currFactProd)
				self.rvarToFactor[rvar] = newList

		# now that all the variables have been eliminated, the remaining
		# factors can all be multiplied together
		finalList = []
		for mvar in mvars:
			finalList.extend(self.rvarToFactor[mvar])

		return self.compute_factorlist_product(finalList)

	def compute_factorlist_product(self,factors):
		""" Computes the factor product of the factors given in a list

			Parameters
			----------
			factors		list
						list of factor objects 
			Returns	
			-------
			factorProd 	pgmFactor
						product of all the factors in the provided list
		"""
		factorProd = pgmf.DiscreteFactor()
		
		for factor in set(factors):	# the 'set' is very important!
			factorProd.compute_factor_product(factor)

		return factorProd

	def compute_joint_distribution(self):
		return self.compute_factorlist_product(self.factors)

	def compute_marginal_bf(self,mvars,evid=None):
		margDist = self.compute_joint_distribution()
		if not evid==None:
			margDist.reduce(evid)
		margDist.marginalize(np.setdiff1d(self.rvars,mvars))
		return margDist





























