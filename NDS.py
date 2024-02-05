import numpy as np
import copy

#Peeling algorithm; iteratively removing the node with the minimum sum of edge-weights and returns nodes
#with the maximum sum of induced edge-weight found.
def peeling(A):
    n = len(A)
    deg_list = [np.sum(A[i,:]) for i in range(n)]
    x_all = [i for i in range(n)]
    x_max = np.argsort(deg_list)[-1:][0]
    x = {x_max, }
    x_all.remove(x_max)
    deg_list[x_max] = 1000000000
    c = 0
    An = copy.deepcopy(A)
    while (np.sum(An))<0 :
        c+=1
        for neigh in range(n):
            if A[x_max,neigh]==0: continue
            if neigh not in x:
                deg_list[neigh] = deg_list[neigh] - A[x_max, neigh]
        x_max = np.argsort(deg_list)[0]
        x.add(x_max)
        deg_list[x_max] = 1000000000
        x_all.remove(x_max)
        An = copy.deepcopy(A)[list(x_all),:]
        An = An[:,list(x_all)]
    return list(x_all)

#Negative Densest Subgraph
def NDS(As, alpha = 1000000):
    print('Running NDS...')
    #number of nodes
    n = len(As[0])
    #init nxn weighted adjacency matrix
    Anew = np.zeros((n, n))
    for t in range(len(As)-1,-1,-1): #from t in {T, T-1,...,1}
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if As[t][i,j] == 0: continue
                if (Anew[i,j] == len(As)-t-1): #if edge is appeared in all timestamts from t to T
                    Anew[i,j] += As[t][i,j]    #increase the edge-weight of the new matrix
                else:                                
                    Anew[i,j] -= alpha #else decrease by alpha
    best_X = peeling(Anew)
    if len(best_X)>1: 
        best_X2 = [x for x in best_X if np.sum([As[i][x,best_X] for i in range(len(As))])>0]
        if len(best_X2)>0: best_X = best_X2
            
    #Induced Adjacencies for all A_t
    for i in range(len(As)):
        As[i] = As[i][best_X,:]
        As[i] = As[i][:,best_X] 
    return As, best_X