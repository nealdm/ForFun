# Q-Learning Concept

# Require:
# States X = {1,...,n_X}
# Actions Act = {1,...,n_a}
#         A: X => Act
# Reward function R: X x Act -> R
# (probabalistic) Transition function T: X x A -> X
# Learning rate alph in [0,1], typically alph = 0.1
# Discount factor gamma in [0,1]
#
# Procedure QLEARNING(X,A,R,T,alph,gamma)
#     Initialize Q: X x Act -> R (arbitrarily)
#     while Q is not converged DO:
#         Start in state s in X
#         while s is not terminal DO:
#             Calculate PI according to Q and exploration strategy (e.g. PI(x)<-argmax_a(Q(x,a)))
#             a <- PI(s)
#             r <- R(s,a)
#             s'<- T(s,a)
#             Q(s',a)<- (1-alph)*Q(s,a) + alph*(r+gamma * max_a'(Q(s',a')) )
#     return Q

''' Let's imagine we have a time series evaluation of the stock market.
    To keep things simple, let's say that:
    The states X are simply represented by the
      span of vectors that are in R**n, where each index represents
      How much money you have in each stock and in your own bank (meaning that each
      index of the vector represents a different company, with (n-1) companies
      in total, with the first index as your own bank account.
    The Actions Act are represented by the R**n dimensional vectors that have
      the constraint that the sum of their elements must 1 (such that the money
      can be distributed from any location to any other location.


'''
