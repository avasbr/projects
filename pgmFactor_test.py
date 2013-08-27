import unittest
import numpy as np
import pgmFactor as pgmf
import pgmUtils as util
import copy

class testFactor(unittest.TestCase):
	
	def setUp(self):
		self.factA = pgmf.DiscreteFactor(scope=np.array([1]),card=np.array([2]),val=np.array([0.11,0.89]))
		self.factB = pgmf.DiscreteFactor(scope=np.array([2,1]),card=np.array([2,2]),val=np.array([0.59,0.41,0.22,0.78]))
		self.factC = pgmf.DiscreteFactor(scope=np.array([3,2]),card=np.array([2,2]),val=np.array([0.39,0.61,0.06,0.94]))
		self.factorList = [self.factA,self.factB,self.factC]

	def test_copy(self):
		factCopy = self.factA
		factCopy_ = pgmf.DiscreteFactor()
		factCopy_.copy(self.factA)

		#*
		np.testing.assert_equal(factCopy.scope,factCopy_.scope)
		np.testing.assert_equal(factCopy.card,factCopy_.card)
		np.testing.assert_almost_equal(factCopy.val,factCopy_.val,decimal=5)

	def test_compute_factor_product(self):
		factP = pgmf.DiscreteFactor(scope=np.array([1,2]),card=np.array([2,2]),val=np.array([0.0649, 0.1958, 0.0451, 0.6942]))
		factP_ = copy.deepcopy(self.factA)
		factP_.compute_factor_product(self.factB)
		
		#*
		np.testing.assert_equal(factP.scope,factP_.scope)
		np.testing.assert_equal(factP.card,factP_.card)
		np.testing.assert_almost_equal(factP.val,factP_.val,decimal=5)

	def test_reduce(self):
		evid = np.array([[2,0],[3,1]])
		factRed = np.array([
							pgmf.DiscreteFactor(scope=np.array([1]),card=np.array([2]),val=np.array([0.11,0.89])),
							pgmf.DiscreteFactor(scope=np.array([2,1]),card=np.array([2,2]),val=np.array([0.59,0,0.22,0])),
							pgmf.DiscreteFactor(scope=np.array([3,2]),card=np.array([2,2]),val=np.array([0,0.61,0,0]))
							])
		factRed_ = copy.deepcopy(self.factorList)

		for fact,fact_ in zip(factRed,factRed_):
			fact_.reduce(evid)
			np.testing.assert_equal(fact.scope,fact_.scope)
			np.testing.assert_equal(fact.card,fact_.card)
			np.testing.assert_almost_equal(fact.val,fact_.val,decimal=5)

	def test_marginalize(self): 
		var = np.array([2])
		factMarg = pgmf.DiscreteFactor(scope=np.array([1]),card=np.array([2]),val=np.array([1,1]))
		factMarg_ = copy.deepcopy(self.factB)
		factMarg_.marginalize(var)

		np.testing.assert_equal(factMarg.scope,factMarg_.scope)
		np.testing.assert_equal(factMarg.card,factMarg_.card)
		np.testing.assert_almost_equal(factMarg.val,factMarg_.val,decimal=5)
		
def main():
	unittest.main()

if __name__ == '__main__':
	main()

# *TODO: Roll this into one custom assertion