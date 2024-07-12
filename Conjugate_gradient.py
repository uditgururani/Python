# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:20:45 2023

@author: Udit Gururani
"""


import numpy as np

def conjugate_gradient(a,b,iterations,x0):
    i = 0
    r0 = b
    p0 = r0
    
    # x0 = [[0],[0],[0]]
    cond = True
    while cond == True:
        alpha = np.matmul(np.matrix.transpose(r0),r0)/np.dot(np.dot(np.matrix.transpose(p0),a), p0)
        x1 = x0+alpha*p0
        print( "Iteration ",i+1,"\nRoots\n", x1)
        r1 = r0 - alpha*np.matmul(a,p0)
        beta = np.matmul(np.matrix.transpose(r1),r1)/(np.matmul(np.matrix.transpose(r0),r0))
        p1 = r1+beta*p0
        i = i+1
        if np.allclose(x0,x1,atol = 1e-12) == True:
            cond = False
            break
        x0 = x1
        p0 = p1
        r0 = r1
    

a = np.array([[10,5,2],[5,3,2],[2,2,3]])
x0 = np.zeros((3,1))
b = np.array([[12],[41],[11]])
print(np.linalg.cholesky(a))
conjugate_gradient(a, b, 20,x0)

print("Solution through standard library\n",np.linalg.solve(a,b))