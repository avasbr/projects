import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def idx_to_assgn(idx,card):
	""" Converts factor assignments to indices

		Parameters
		----------
		idx:	int or numpy array
			 	indices for which we wish to compute factor assignments 
		card: 	numpy array
				cardinalities of variables in the scope of the factor

		Returns
		-------
		numpy array
				factor assignments corresponding to indices

		See also
		--------
		assgn_to_idx 
	"""
	cprod = np.concatenate((np.array([1]),np.cumprod(card[0:-1])))
	return np.mod(np.array(idx)[:,np.newaxis]/cprod,card)

def assgn_to_idx(assgn,card):
	""" Converts indices to factor assignments

		Parameters
		----------
		assgn:	numpy array
				factor assignments
		card:	numpy array
				cardinalities of variables in the scope of the factor

		Returns
		-------
		int or numpy array
				indices corresponding

		See also
		--------
		idx_to_assgn
	"""
	cprod = np.concatenate((np.array([1]),np.cumprod(card[0:-1])))
	return np.dot(assgn,cprod)

def map_arrays(A,B):
	""" Returns the indices of A for elements 

		Parameters
		----------
		A:	numpy array
		B:	numpy array

		Returns
		-------
		idx:	numpy array
				indices of A for elements of B
		mem:	numpy array
				indices of B that actually exist in A
	"""
	idx = []
	mem = []
	for i,b in enumerate(B):
		idxTup = np.where(A==b)[0]
		if not len(idxTup)==0:
			idx.append(idxTup[0])
			mem.append(i)
	return idx,mem

