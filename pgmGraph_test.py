import numpy as np
import pgmGraph as pgmg
import pgmFactor as pgmf
import unittest
import copy

class testPGM(unittest.TestCase):

	def setUp(self):
		self.fact1 = pgmf.DiscreteFactor(scope=np.array([1]),card=np.array([2]),val=np.array([0.11,0.89]))
		self.fact2 = pgmf.DiscreteFactor(scope=np.array([2,1]),card=np.array([2,2]),val=np.array([0.59,0.41,0.22,0.78]))
		self.fact3 = pgmf.DiscreteFactor(scope=np.array([3,2]),card=np.array([2,2]),val=np.array([0.39,0.61,0.06,0.94]))
		self.factors = [self.fact1,self.fact2,self.fact3]
		self.pgm = pgmg.PGM(self.factors)
		self.evid = np.array([[1,1]])
	
	def test_init(self):
		
		rvarToFactor = {1:[self.fact1,self.fact2],2:[self.fact2,self.fact3],
						3:[self.fact3]}
		rvars = [1,2,3]

		factors_ = self.pgm.factors
		rvarToFactor_ = self.pgm.rvarToFactor
		rvars_ = self.pgm.rvars

		#TODO: Check the rvarToFactors as well
		self.assertItemsEqual(rvars,rvars_)

		for fact,fact_ in zip(self.factors,factors_):
			np.testing.assert_equal(fact.scope,fact_.scope)
			np.testing.assert_equal(fact.card,fact_.card)
			np.testing.assert_almost_equal(fact.val,fact_.val,decimal=5)

	
	def test_add_factor(self):
		pgmCopy = copy.deepcopy(self.pgm)
		newFactor = pgmf.DiscreteFactor(scope=np.array([4]),card=np.array([2]),val=np.array([0.5,0.5]))
		pgmCopy.add_factor(newFactor)

		factors = [self.fact1,self.fact2,self.fact3,newFactor]
		rvarTofIdx = {1:[self.fact1,self.fact2],2:[self.fact2,self.fact3],
					3:[self.fact3],4:[newFactor]}
		
		rvars = [1,2,3,4]

		factors_ = pgmCopy.factors
		rvarToFactor_ = pgmCopy.rvarToFactor
		rvars_ = pgmCopy.rvars

		self.assertItemsEqual(rvars,rvars_)

		for fact,fact_ in zip(factors,factors_):
			np.testing.assert_equal(fact.scope,fact_.scope)
			np.testing.assert_equal(fact.card,fact_.card)
			np.testing.assert_almost_equal(fact.val,fact_.val,decimal=5)
	
	def test_compute_joint_distribution(self):
		joint = pgmf.DiscreteFactor(scope=np.array([1,2,3]),card=np.array([2,2,2]),val=np.array([0.025311, 0.076362, 0.002706, 0.041652, 0.039589, 0.119438, 0.042394, 0.652548]))
		joint_ = self.pgm.compute_joint_distribution()

		np.testing.assert_equal(joint.scope,joint_.scope)
		np.testing.assert_equal(joint.card,joint_.card)
		np.testing.assert_almost_equal(joint.val,joint_.val,decimal=5)

	def test_compute_marginal(self):
		margFactor = pgmf.DiscreteFactor(scope=np.array([2,3]),card=np.array([2,2]),val=np.array([0.0858,0.0468,0.1342,0.7332]))
		margFactor_ = self.pgm.compute_marginal(np.array([2,3]),self.evid)
		margFactor_.val = margFactor_.val/np.sum(margFactor_.val)

		np.testing.assert_equal(margFactor.scope,margFactor_.scope)
		np.testing.assert_equal(margFactor.card,margFactor_.card)
		np.testing.assert_almost_equal(margFactor.val,margFactor_.val,decimal=5)

	
	def test_compute_marginal_bf(self):
		margFactor = pgmf.DiscreteFactor(scope=np.array([2,3]),card=np.array([2,2]),val=np.array([0.0858,0.0468,0.1342,0.7332]))
		margFactor_ = self.pgm.compute_marginal_bf(np.array([2,3]),self.evid)
		margFactor_.val = margFactor_.val/np.sum(margFactor_.val)

		np.testing.assert_equal(margFactor.scope,margFactor_.scope)
		np.testing.assert_equal(margFactor.card,margFactor_.card)
		np.testing.assert_almost_equal(margFactor.val,margFactor_.val,decimal=5)

def main():
	unittest.main()

if __name__ == '__main__':
	main()

