# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 07:29:20 2023

@author: Udit Gururani
"""

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
from scipy.interpolate import barycentric_interpolate
#global parameters
u = 1.0
rho = 1
gamma = 0.02
l = 1
points = np.array([41,81,161,321]) ## number of grid points CVs+1
Pe = 50 #Peclet number
phi_0 = 0
phi_1 = 1
phi_p_CDS = np.zeros((len(points),1),dtype=float) # Array to store value at x = 0.9 for CDS scheme.
phi_p_UPWIND = np.zeros((len(points),1),dtype=float) # Array to store values at x = 0.9 for Upwind scheme.


## Exact solution to the problem
def exact_sol(x,Pe,phi_0):
    return phi_0 + (np.exp(x*Pe/l) - 1)/(np.exp(Pe) - 1)


for i in range(len(points)):
    #mesh
    n = points[i]
    dx = l/(n-1)
    phi = np.zeros((n,1),dtype=float)
    ## Boundary conditions on Phi
    phi[0] = phi_0
    phi[-1] = phi_1
    # print(np.shape(phi))

    F = rho*u   #Advection strength
    D = gamma/dx   #Diffusion strength

    def CDS(F,D,n):
        a_P = (D - F/2) + (D + F/2)
        a_W = D + F/2
        a_E = D - F/2

        # defining the A matrix
        A = np.zeros((n-2,n-2),dtype = float)
        # print(np.shape(A))
        m = len(A)
        A[0,0] = -a_P
        A[0,1] = a_E
        A[m-1,m-2] = a_W
        A[m-1,m-1] = -a_P
        # print(A)
        for i in range(1,m-1):
            for j in range(i-1,i+2):
                if j<i:
                    A[i,j] = a_W
                if i==j:
                    A[i,j] = -a_P
                
                elif j>i:
                    A[i,j] = a_E
                    
        # defining the rhs matrix
        B = np.zeros((n-2,1),dtype = float)
        # print(np.shape(B))
        B[0] = -a_W*phi[0]
        B[-1] = -a_E*phi[-1]

        phi[1:-1] = np.linalg.solve(A,B) # solution of matrices

        return phi

    def UPWIND(F,D,n):
        a_P = (D + F) + D + (F - F)
        a_W = D + F
        a_E = D 

        # defining the A matrix
        A = np.zeros((n-2,n-2),dtype = float)
        # print(np.shape(A))
        m = len(A)
        A[0,0] = -a_P
        A[0,1] = a_E
        A[m-1,m-2] = a_W
        A[m-1,m-1] = -a_P
        # print(A)
        for i in range(1,m-1):
            for j in range(i-1,i+2):
                if j<i:
                    A[i,j] = a_W
                if i==j:
                    A[i,j] = -a_P
                
                elif j>i:
                    A[i,j] = a_E
                    
        # defining the rhs matrix
        B = np.zeros((n-2,1),dtype = float)
        # print(np.shape(B))
        B[0] = -a_W*phi[0]
        B[-1] = -a_E*phi[-1]

        phi[1:-1] = np.linalg.solve(A,B) # solution of matrices

        return phi

    

    x = np.linspace(0,1,n)
    
    phi_p_CDS[i] = barycentric_interpolate(x,CDS(F,D,n),0.9)
    phi_p_UPWIND[i] = barycentric_interpolate(x,UPWIND(F,D,n),0.9)
    plt.plot(x,CDS(F,D,n),marker='o',markersize = 7,label = "CDS",alpha = 0.7,lw = 3)
    plt.plot(x,UPWIND(F,D,n),marker='*',markersize = 7,label = "UPWIND",alpha = 0.7,lw = 3)
    plt.plot(x,exact_sol(x,Pe,phi_0),label = "exact solution",alpha = 0.7,lw = 3)
    plt.xlabel("x")
    plt.ylabel("phi")
    plt.title("Solution for {} grid points".format(points[i]))
    plt.legend()
    plt.show()
    
# Plotting the values of Phi at x = 0.9 for CDS and Upwind, using log linear plotting
plt.plot(points,phi_p_UPWIND,marker='*',markersize = 7,label="Upwind scheme")
plt.xscale("log")
plt.xlabel("grid points")
plt.ylabel("phi at x = 0.9")
plt.plot(points,phi_p_CDS,marker='*',markersize = 7,label = "CDS")
plt.xlabel("grid points")
plt.ylabel("phi at x = 0.9 ")
plt.legend()

# order estimation using the finest 3 grids
# using CDS scheme
# h is 80 CVs, rh = 160 CVs, r^2h = 320 CVs.
# phi_p_CDS[1] refers to CVs 80 and likewise [2] refers to 160 CVs and [3] refers to 320 CVs.
# in our case r = 1/2 (refinement ratio)

r = 1/2
p_CDS = np.log((phi_p_CDS[3]-phi_p_CDS[2])/(phi_p_CDS[2]-phi_p_CDS[1]))/np.log(r)
print("Order estimation using CDS: ", p_CDS)
p_UPWIND = np.log((phi_p_UPWIND[3]-phi_p_UPWIND[2])/(phi_p_UPWIND[2]-phi_p_UPWIND[1]))/np.log(r)
print("Order estimation using Upwind scheme: ", p_UPWIND)

# solution using richardson extrapolation

sol_richardson_CDS = phi_p_CDS[3] + (phi_p_CDS[2] - phi_p_CDS[3])/(1 - 1/(r**p_CDS))
sol_richardson_upwind = phi_p_UPWIND[3] + (phi_p_UPWIND[2] - phi_p_UPWIND[3])/(1 - 1/(r**p_UPWIND))
print("Richardson extrapolation solution (CDS): ", sol_richardson_CDS)
print("Richardson extrapolation solution (Upwind): ", sol_richardson_upwind)

# difference between exact and richardson extrapolation

difference_CDS = exact_sol(0.9, Pe, phi_0) - sol_richardson_CDS
difference_Upwind = exact_sol(0.9, Pe, phi_0) - sol_richardson_upwind

print("difference between exact and richardson extrapolation (CDS): ",difference_CDS)
print("difference between exact and richardson extrapolation (Upwind): ",difference_Upwind)








