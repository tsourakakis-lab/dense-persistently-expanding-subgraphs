import networkx as nx
import numpy as np
from help_functions import read_graphs


#Load Tree dataset
def load_treegraph(n = 100, T = 20, weighted = None):
    name = 'treegraph'
    return read_graphs(name, n, T, weighted)


#Load Random dataset
def load_randomgraph(n = 100, T = 20, weighted = None):
    name = 'randomgraph'
    return read_graphs(name, n, T, weighted)


#Load Animal dataset
def load_animal(n = 100, T = range(10), weighted = None):
    name = 'animal'
    return read_graphs(name, n, T, weighted)


#Load Clique_Animal dataset
def load_clique_plus_animal(n = 100, T = range(10), weighted = None):
    name = 'animal'
    A = np.ones((n,n))
    for i in range(n): A[i,i] = 0
    return read_graphs(name, n, T, weighted, As=[A])


#Load Email dataset
def load_email(n = 100, T = range(10), weighted = None):
    name = 'email'
    return read_graphs(name, n, T, weighted)


#Load Flights dataset
def load_usflights(n = 100, T = range(10), weighted = None):
    name = 'usflights'
    return read_graphs(name, n, T, weighted)


#Load School dataset
def load_school(n = 100, T = range(10), weighted = None):
    name = 'school'
    return read_graphs(name, n, T, weighted)