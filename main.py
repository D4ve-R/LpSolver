from lpsolver import *


A = [[2, -2], [0, 1]] 
b = [3, 3]
c = [4, -1]

prob = MaxLp(A,b,c)
obj,x,_ = prob.solve()
