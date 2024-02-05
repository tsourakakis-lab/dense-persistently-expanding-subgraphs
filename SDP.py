import cvxpy as cp
from cvxpy import *
import numpy as np
from help_functions import eigs, Laplacian
import copy


#Induced Laplacian of Y by Y=xx^T, with x being an indicator vector. 
def induced_laplacian(A, Y):
    E = cp.multiply(Y, A) #induced adjacaceny
    return cp.multiply((cp.sum(E, axis = 1, keepdims=True) @ np.ones((1,len(A)))), np.identity(len(A))) - E 
 
 
#SDP algorithm for DPES
def SDP_DPES(As, sigma, ret_time = False):
    #Number of nodes
    n = len(As[0])
    #Number of timstamps
    T = len(As)
    
    #Define and solve the CVXPY problem.
    #--------------------------------------------------------------------------------------------------------------
    #Constraints
    #-------------------------------------------------------------------------------------------------------------
    #Y is a Symmetric and PSD  matrix 
    #with Yij = x_i*x_j, where x_i = 1 if node i is part of the solution and x_i = 0 otherwise
    Y = cp.Variable((n+1,n+1), symmetric = True)
    constraints = [Y>>0]

    #Yij is in the range [0, 1] and Y[0,0]=1
    constraints += [Y >= 0]  #we can ommit it
    constraints += [Y <= 1] #we can ommit it
    constraints += [Y[0,0] == 1]
    
    #diag(Y)=x
    constraints += [cp.diag(Y) == Y[0,:]]

    #Spectral constraint: L >> sigma*R, where L, R are the induced laplacians of A_t and A_{t-1} respectively.
    L = 0 
    R = 0
    for t in range(1,T):
            L = induced_laplacian(As[t], Y[1:,1:])
            R = induced_laplacian(As[t-1], Y[1:,1:])
            constraints += [L >> sigma*R]
    #-------------------------------------------------------------------------------------------------------------
    #Objective
    #-------------------------------------------------------
    #Objective function: sum of induced edge-weights.
    density_score = 0
    Aall = np.zeros((n+1,n+1))
    for A in As:
        Aall[1:,1:] += A
    density_score = cp.sum(cp.multiply(Y,Aall))    
    #-------------------------------------------------------
    #Solving the problem
    #-----------------------------------------------------------------
    #Set the problem 
    prob = cp.Problem(cp.Maximize(density_score), constraints)
    #Solve problem
    print('SDP solving...')
    _verbose = True
    if ret_time: _verbose = False
    prob.solve(verbose = _verbose, solver = cp.MOSEK, warm_start=True)
    #------------------------------------------------------------------
    print(f'Compilation time {prob._compilation_time}')
    print(f'Solver Time {prob._solve_time}')
    #--------------------------------------------------------------------------------------------------------------
    if ret_time: return prob._compilation_time, prob._solve_time
    Y = Y.value
    return Y


#Iteratively execute the SDP algorithm until finding a feasible solution
def run_SDP_DPES(As, sigma):
    #Run SDP_DPES and sorted diagonal elements
    Yret = SDP_DPES(As, sigma = sigma)
    Y = np.diag(Yret[1:,1:])
    sorted_ind = np.argsort(-1*Y)
    iterations = [range(len(Y))]
    #Max obective (sum of induced edge-weights) and optimal solution
    max_objective = -1
    best_X = None
    #Iterate over permutations
    for sort_L in [sorted_ind]:
        for iters in iterations:
            flag = True
            X = []
            for i in iters:
                X.append(sort_L[i]) #append the next element
                objective_t = 0     #current density
                eigenvalues_t_minus_1 = [0 for _ in range(len(X))] #eigenvalues
                flag = False
                for A in As:
                    A_t = A[X,:]
                    A_t = A_t[:,X]
                    objective_t += np.sum(A_t)
                    eigenvalues_t = eigs(Laplacian(A_t))
                    eigenvalues_t.sort()
                    flag = False
                    if(np.min(eigenvalues_t-eigenvalues_t_minus_1) < -0.000001):
                        flag = True
                        eigenvalues_t_minus_1 = [0 for _ in range(len(eigenvalues_t))]
                        break
                    else:
                        eigenvalues_t_minus_1 = [eigenvalues_t[j]*sigma for j in range(len(eigenvalues_t))]
                if flag: continue
                elif objective_t >= max_objective:
                    max_objective = objective_t
                    best_X = copy.deepcopy(X)
    if len(best_X)>1: best_X = [x for x in best_X if np.sum([As[i][x,best_X] for i in range(len(As))])>0]
    
    #Induced Adjacencies for all A_t
    for i in range(len(As)):
        As[i] = As[i][best_X,:]
        As[i] = As[i][:,best_X] 
    return Yret, As, best_X