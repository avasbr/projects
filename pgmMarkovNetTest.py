import numpy as np
import pgmGraph as pgmg
import pgmFactor as pgmf
import pgmUtils as pgmu

# Example 1: Markov Net toy example from Coursera
#-------------------------------------------------

# A = 1
# B = 2
# C = 3
# D = 4

# factor1 = pgmf.DiscreteFactor(scope=np.array([1,2]),card=np.array([2,2]),val=np.array([30,1,5,10]))
# factor2 = pgmf.DiscreteFactor(scope=np.array([2,3]),card=np.array([2,2]),val=np.array([100,1,1,100]))
# factor3 = pgmf.DiscreteFactor(scope=np.array([3,4]),card=np.array([2,2]),val=np.array([1,100,100,1]))
# factor4 = pgmf.DiscreteFactor(scope=np.array([1,4]),card=np.array([2,2]),val=np.array([100,1,1,100]))

# factors = [factor1,factor2,factor3,factor4]
# markovNet = pgmg.PGM(factors)

# # Computing P(A,B) using variable elimination of individual factors
# markovMargDist_ve = markovNet.compute_marginal(np.array([1,2]))
# markovMargDist_ve.val = 1.0*markovMargDist_ve.val/np.sum(markovMargDist_ve.val)

# # Compute P(A,B) using brute force computation of the joint distribution
# markovMargDist_bf = markovNet.compute_joint_distribution()
# markovMargDist_bf = markovNet.compute_marginal_bf(np.array([1,2]))
# markovMargDist_bf.val = 1.0*markovMargDist_bf.val/np.sum(markovMargDist_bf.val)

# # Answer: [a0,b0 = 0.13, a1,b0 = 0.14, a0,b1 = 0.69, a1,b1 = 0.04]
# print markovMargDist_ve.val
# print markovMargDist_bf.val

# Example 2: Student example Bayes net from Coursera
#----------------------------------------------------

# difficulty = 1
# intelligence = 2
# grade = 3
# sat = 4
# letter = 5


difficulty = pgmf.DiscreteFactor(scope=np.array([1]),card=np.array([2]),val=np.array([0.6,0.4]))
intelligence = pgmf.DiscreteFactor(scope=np.array([2]),card=np.array([2]),val=np.array([0.7,0.3]))
grade = pgmf.DiscreteFactor(scope=np.array([1,2,3]),card=np.array([2,2,3]),val=np.array([0.3,0.05,0.9,0.5,0.4,0.25,0.08,0.3,0.3,0.7,0.02,0.2]))
sat = pgmf.DiscreteFactor(scope=np.array([2,4]),card=np.array([2,2]),val=np.array([0.95,0.2,0.05,0.8]))
letter = pgmf.DiscreteFactor(scope=np.array([3,5]),card=np.array([3,2]),val=np.array([0.1,0.4,0.99,0.9,0.6,0.01]))

factors = [difficulty,intelligence,grade,sat,letter]
bayesNet = pgmg.PGM(factors)

# Computing P(I=1|G=1,D=1); Answer = 0.339
bayesMargDist_bf = bayesNet.compute_joint_distribution()
bayesMargDist_bf = bayesNet.compute_marginal_bf(np.array([2]),np.array([[3,1],[1,1]]))
bayesMargDist_bf.val = bayesMargDist_bf.val/np.sum(bayesMargDist_bf.val)

bayesMargDist_ve = bayesNet.compute_marginal(np.array([2]),np.array([[3,1],[1,1]]))
bayesMargDist_ve.val = bayesMargDist_ve.val/np.sum(bayesMargDist_ve.val)

print bayesMargDist_bf.val[1]
print bayesMargDist_ve.val[1]
