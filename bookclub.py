# Cameron's bookclub
import math as m
from math import factorial as f
from scipy import misc
import itertools as it
import numpy as np

def sorting(n,p):
    # same as combination of n choose p
    return sum(f(n-i)/(f(p-1)*f(n-(p+i-1))) for i in range(1,p+1))

# Question is this:  Suppose you have x people you want to make a bookclub
# with.  Suppose you have t total books and each person likes b of the books.
# What is the probability that a bookclub can be formed?  That is, what is
# the probability that there will be at least 1 book that everybody likes?

# This is a combinations problem disguised as a probability problem.
# We want to see if each person can like the same book or not.
# Essentially, we want to look at the number of nonempty intersections
# in the cartesian product of a set with itself x number of times,
# where the set is composed of all combinations of b elements
# chosen from a set of size t.


def books(x,t,b):
    '''paramaters:
           x: number of people
           t: number of books in total
           b: number of books each person will like

       returns:
           p: the probability there will be at least 1 book everyone likes.
    '''
    # the following represents all the books available (as numbers)
    total_books = np.arange(t)
    # the following creates a set of all possible combinations
    # of the books to be used
    sets = [set(i) for i in it.combinations(total_books,b)]

    # the total number of ways that people can have their
    # book choices destributed is given by (t choose b)^x.
    denominator = misc.comb(t,b)**x

    # represents all possible groups of people, who can like whichever
    # set of books that they like.
    peoples_choises = [groups for groups in it.product(sets,repeat = x)]

    for list in it.combinations(peoples_choises):
        

    return(len(peoples_choises))
