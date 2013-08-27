import numpy as np

class Clique():
		def __init__(self,scope=None,card=None,val=None):
		self.scope = scope		# full scope of factors
 		self.card = card 		# cardinality of each of the variables
		self.val = val 			# values for each of assignments
