import unittest
import numpy as np
import pgmUtils as util

class testUtils(unittest.TestCase):

	def test_idx_to_assgn(self):		
		card = np.array([3,4])
		idx = np.array([5,11])
		asgn = np.array([[2,1],[2,3]])
		np.testing.assert_equal(util.idx_to_assgn(idx,card),asgn)

	def test_assgn_to_idx(self):
		card = np.array([3,4])
		asgn = np.array([[2,1],[2,3]])
		idx = np.array([5,11])
		np.testing.assert_equal(util.assgn_to_idx(asgn,card),idx)

	def test_map_arrays(self):
		A = np.array([2,1,5,3,6,7])
		B = np.array([3,2,8,7])
		idx = [3,0,5]
		mem = [0,1,3]
		idx_,mem_ = util.map_arrays(A,B)
		np.testing.assert_equal(idx,idx_)
		np.testing.assert_equal(mem,mem_)

def main():
	unittest.main()

if __name__ == '__main__':
	main()