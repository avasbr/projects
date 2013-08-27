import numpy as np
import pgmUtils as util

class DiscreteFactor():

	def __init__(self,scope=None,card=None,val=None):
		self.scope = scope		# full scope of variables
 		self.card = card 		# cardinality of each of the variables
		self.val = val 			# values for each of assignments

	def copy(self,factor):
		""" Create a deep copy of factors"""

		self.scope = np.copy(factor.scope)
		self.card = np.copy(factor.card)
		self.val = np.copy(factor.val)

	def checkEmpty(self):
		""" Checks if this factor is empty"""

		if self.scope==None and self.card==None and self.val==None:
			return True
		return False

	def compute_factor_product(self,factor):
		"""	Computes the factor product of this factor with the input factor

			Parameter
			---------
			factor:	DiscreteFactor
					Factor which will be multiplied with this instance

			Returns
			-------
			None
				The attributes of this instance are updated

			Additional
			----------
			If the current instance is empty, the factor product simply returns
			the same factor which was provided as input 
		"""
		if self.checkEmpty():
			self.copy(factor)
		else:
			# set variable random indices to be the union of the two factors
			scope  = np.union1d(self.scope, factor.scope) 

			# set cardinality
			card = np.empty(len(scope) ,dtype=np.int)
			mapOrig,dummy = util.map_arrays(scope, self.scope)  # mapA: i | C(i) = A(j), j = 0,..len(A)	
			mapFact,dummy = util.map_arrays(scope,factor.scope) # mapB: i | C(i) = B(j), j = 0,..len(B)
			card[mapOrig] = self.card
			card[mapFact] = factor.card
			
			# set values from the factor product
			assgns = util.idx_to_assgn(np.arange(np.prod(card)), card)
			idxOrig = util.assgn_to_idx(assgns[:,mapOrig],self.card)
			idxFact = util.assgn_to_idx(assgns[:,mapFact],factor.card)
			val = self.val[idxOrig]*factor.val[idxFact] # perform the factor product

			# update factor
			self.scope = scope
			self.card = card
			self.val = val

	def reduce(self,evid):
		"""	Reduces the factor with provided evidence

			Parameters
			----------
			evid:	numpy array
					Nx2 matrix, where each row corresponds to a variable and 
					corresponding observed value pair

			Returns
			--------
			None
				Updates attributes of this instance
		"""
		mapE,memE = util.map_arrays(self.scope,evid[:,0]) # map variables in evid to factor scope

		if len(mapE)!=0:
			assgns = util.idx_to_assgn(np.arange(np.prod(self.card)),self.card)
			for row in assgns:
				if not (row[mapE]==evid[memE,1]).all():
					self.val[util.assgn_to_idx(row,self.card)] = 0	
		
	def marginalize(self,rvars):
		""" Sum out variables from the factor

			Parameters
			----------
			rvars:	int or numpy array
					variable(s) to sum out of the factor

			Returns
			-------
			None
				Updates attributes of this instance
		"""
		scope = np.setdiff1d(self.scope,rvars) # compute set difference - these are the variables that will stay

		# if the variable(s) that are to be marginalized out are not in
		# the factor, no changes are neceesary
		if len(scope)!=0:
			mapM,dummy = util.map_arrays(self.scope,scope) # get the indices that map into the original factor
			card = self.card[mapM] # set cardinalities of marginalized factor
			numVal = np.prod(card)
			assgn = util.idx_to_assgn(np.arange(np.prod(self.card)),self.card)
			idx = util.assgn_to_idx(assgn[:,mapM],card)

			# fill in the values
			val = np.empty(numVal)
			for i in np.arange(numVal):
				val[i] = np.sum(self.val[util.assgn_to_idx(assgn[idx==i,:],self.card)])
			
			# update factor
			self.scope = scope
			self.card = card
			self.val = val
		else:
			self.scope = np.array([])
			self.card = 0
			self.val = np.array([1])