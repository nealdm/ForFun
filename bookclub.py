# Cameron's bookclub
import math as m
from math import factorial as f

def sorting(n,p):
    return sum(f(n-i)/(f(p-1)*f(n-(p+i-1))) for i in range(1,p+1))
