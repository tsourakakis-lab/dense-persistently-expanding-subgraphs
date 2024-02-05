import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

#read nodes from file
def read_nodes(As, filename, read_all = False):
    X = []
    flag = False
    with open(filename) as file:
        for line in file:
            if line=='======Nodes======\n':
                flag=True
            elif read_all or flag: 
                #print(line[:-1])
                X.append(int(line[:-1]))
    for i in range(len(As)):
        As[i] = As[i][X,:]
        As[i] = As[i][:,X] 
    return As, X


#read graph from edgelist
def readGraph(filename, N = None, weighted = None):
        file1 = open(filename)
        Lines = file1.readlines()
        count = 0
        for line in Lines:
            if (weighted is not None) and (weighted!=False):
                edge = (list(map(float, list(line[0:-1].split(" ")[0:3]))))
            else: edge = (list(map(float, list(line[0:-1].split(" ")[0:2]))))
            if count == 0:
                if N is None:
                    N = int(edge[0])
                g = nx.Graph()
                for i in range(N):
                    g.add_node(i)
            else:
                if (edge[0]<=(N-1)) and (edge[1]<=(N-1)):
                    if (weighted is not None) and (weighted!=False): g.add_edge(edge[0], edge[1], weight = edge[2])
                    else: g.add_edge(edge[0], edge[1])
            count+=1
        return g


#read graphs
def read_graphs(name, n = 100, range_T=[0], weighted = None, As = None, extra_folder=''):
    if As is None: As = []
    for i in range_T:     
        G = readGraph(filename = f'./Graphs/{extra_folder}{name}_{i}.txt', weighted=weighted)
        A_G = np.array(nx.adjacency_matrix(G).todense())
        A_G = A_G[0:n,0:n]
        As.append(A_G)
    return As


#return edges of networkx graph as a list
def nxgetEdges(g):
    edges = []
    for u,v in g.edges():
        if 'weight' in g[u][v]:
            edges.append([u,v,g[u][v]['weight']])
        else: edges.append([u,v])
    return edges


#save weighted graph as triplets
def saveWeightedGraph(N, edges, filename, saveall = True):
    ma_mm_file = './BestFriendsForever-BFF--master/java/experiments'
    filename_all = ma_mm_file+'/'+filename.split('_')[0].split('/')[-1]+'.txt'
    f2 = None
    t = filename.split('_')[-1].split('.')[0]
    if saveall and os.path.exists(ma_mm_file):
        if t=='0': f2 = open(filename_all, 'w')
        else: f2 = open(filename_all, 'a')
    f = open(filename, 'w')
    f.write(str(N))
    f.write('\n')
    for edge in edges:
        f.write(str(list(edge)[0])+" "+str(list(edge)[1])+" "+str(list(edge)[2]))
        f.write('\n')
        if saveall and os.path.exists(ma_mm_file):
            f2.write(str(list(edge)[0])+"\t"+str(list(edge)[1])+"\t"+str(t))
            f2.write('\n')
        

#write text at the end of file
def write_at_end(new_text, folder, filename):
    file_object = open(f'./{folder}/{filename}.txt', 'a')
    file_object.write(new_text)
    file_object.close()
   
   
#write text at the begin of file   
def write_at_begin(new_text, folder, filename):
    file_object = open(f'./{folder}/{filename}.txt', 'w')
    file_object.write(new_text)
    file_object.close()


#finding stats for the graph with (weighted) adjacency A
#--------------------------------------------------------
def stats(A):
    import networkx as nx
    num_of_nodes = len(A)
    if num_of_nodes == 0: return f'0 0 0 0 0 0'
    num_of_edges = np.sum(A)/2
    min_deg = np.min(np.sum(A, axis = 0))
    max_deg = np.max(np.sum(A, axis = 0))
    G = nx.from_numpy_array(A)
    number_of_triangles = sum(nx.triangles(G).values()) / 3
    L_A = Laplacian(A)
    Eigs_L_A = np.sort(np.linalg.eigvals(L_A))
    lambda2 = 0
    if len(Eigs_L_A)>1: lambda2 = Eigs_L_A[1].real
    return f'{num_of_nodes} {num_of_edges} {min_deg} {max_deg} {number_of_triangles} {lambda2}'


def stats_ar(As):
    Aminus = As[0]
    A = As[1]
    import networkx as nx
    num_of_nodes = len(A)
    num_of_edges = np.sum(A)/2
    min_deg = np.min(np.sum(A, axis = 0))
    max_deg = np.max(np.sum(A, axis = 0))
    G = nx.from_numpy_array(A)
    number_of_triangles = sum(nx.triangles(G).values()) / 3
    number_plus = 0
    number_minus = 0
    Adiff = A-Aminus
    edges_in = 0
    edges_out = 0
    for i in range(len(Adiff)):
        for j in range(i,len(Adiff)):
            if i==j: continue
            if Adiff[i,j]<0:
                edges_out += 1
                number_minus += np.sum(A[i,:]*A[:,j])
            if Adiff[i,j]>0:
                edges_in +=1
                number_plus += np.sum((A[i,:]*A[:,j]))
    L_A = Laplacian(A)
    Eigs_L_A = np.sort(np.linalg.eigvals(L_A))
    lambda2 = 0
    if len(Eigs_L_A)>1: lambda2 = Eigs_L_A[1].real
    return f'{num_of_nodes} {num_of_edges} {min_deg} {max_deg} {number_of_triangles} {lambda2} {number_plus} {number_minus} {edges_in} {edges_out}'


def stats_extra(A):
    import networkx as nx
    G = nx.from_numpy_array(A)
    edgecon = nx.edge_connectivity(G)
    nodecon = nx.node_connectivity(G)
    diameter = nx.diameter(G)
    avgpath = nx.average_shortest_path_length(G)
    return f'{edgecon} {nodecon} {diameter} {avgpath}'


def stats_for_G(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deglist = [G.degree[node] for node in G.nodes()]
    min_deg = np.min(deglist)
    max_deg = np.max(deglist)
    triangles = sum(nx.triangles(G).values()) / 3
    X = np.array(nx.adjacency_matrix(G, weight=None).todense()).reshape(n,n)
    X = np.diag(np.sum(X, axis = 0)) - X
    eigsall = eigs(X)
    eigsall.sort()
    l2 = eigsall[1].real
    lmax = eigsall[-1].real
    return np.array([n, m, min_deg, max_deg, triangles, l2, lmax])
#--------------------------------------------------------


#eigenvalues of X
def eigs(X):
    return np.linalg.eigvals(X)


#Laplacian of X
def Laplacian(X):
    return (np.sum(X, axis=0) * np.identity(len(X)) - X)


#drawing a netwokr
def draw_networks(As, graphmame, showall = False, titlename = None): 
    fig, axes = plt.subplots(ncols=len(As))
    ax = axes.flatten()
    counter = 0
    for A in As:
        G = nx.from_numpy_array(A)
        nx.draw_networkx(G, pos=nx.circular_layout(G), node_size = 200, width = 2,with_labels = False, ax=ax[counter])
        ax[counter].set_axis_off()
        counter += 1
    if titlename is not None: plt.title(titlename)
    plt.tight_layout()
    fig.savefig(f'./Plots_/{graphmame}.png', format='png', dpi=300)
    if showall: plt.show()
    

#plot and save spectrum of SDP and NDS algorithms
def spectrum_plots(As1, As2, graphname, showall=False, labels = ['SDP', 'NDS'], Ts = None, folder = 'Plots_'):
    fig, axes = plt.subplots(ncols=2)
    ax = axes.flatten()
    colors = ['red', 'blue', 'green', 'orange', 'brown',
              'olive', 'black', 'purple', 'yellow', 'grey']
    counter = 0
    for As in [As1, As2]:
        i = 0
        for A in As:
            L_A = Laplacian(A)
            Eigs_L_A = np.sort(np.linalg.eigvals(L_A))
            #print(Eigs_L_A)
            file = None
            if(counter==0): 
                if i==0: 
                    file = open(f'./{folder}/{graphname}_{labels[0]}.txt','w')
                    file.write('eig k t\n')
                else: file = open(f'./{folder}/{graphname}_{labels[0]}.txt','a')
            else: 
                if i==0: 
                    file = open(f'./{folder}/{graphname}_{labels[1]}.txt','w')
                    file.write('eig k t\n')
                else: file = open(f'./{folder}/{graphname}_{labels[1]}.txt','a')
            for k in range(len(Eigs_L_A)):
                file.write(f'{Eigs_L_A[k].real} {k} {i}\n')
            ax[counter].plot(Eigs_L_A, color = colors[i])
            i+=1
        if Ts is None: Ts = range(len(As))
        ax[counter].legend([f't={i}' for i in Ts], loc ="upper left")
        ax[counter].set_ylabel('Eigenvalues')
        if counter==0:
            ax[counter].set_xlabel(labels[0])
        else: 
            ax[counter].set_xlabel(labels[1])
        counter+=1
    plt.tight_layout()
    fig.savefig(f'./{folder}/{graphname}_connectivity_{labels[0]}_vs_{labels[1]}.png', format='png', dpi=300)
    if showall: plt.show()
    

#saving results
def saving_results(As, X, filename, algoname, SDStxt, T, weighted=False, write_extra = False, folder = 'Results_'):
    if write_extra:
        write_at_begin('n m min_deg max_deg triangles lambda2 trianglesin trianglesout edgesin edgesout edgecon nodecon diam avgpath\n', 
                        folder, f'{filename}_data_{algoname}{SDStxt}_{T}')
        for i in range(len(As)): 
            if ((weighted) or (i == 0)): write_at_end(stats(As[i])+' 0 0 0 0 0 0\n', 
                folder, f'{filename}_data_{algoname}{SDStxt}_{T}')
            else: write_at_end(stats_ar([As[i-1], As[i]])+' '+stats_extra(As[i])+'\n', 
                folder, f'{filename}_data_{algoname}{SDStxt}_{T}')
        
    else:
        write_at_begin('n m min_deg max_deg triangles lambda2 trianglesin trianglesout edgesin edgesout\n', 
                        folder, f'{filename}_data_{algoname}{SDStxt}_{T}')
        for i in range(len(As)): 
            if ((weighted) or (i == 0)): write_at_end(stats(As[i])+' 0 0\n', 
                folder, f'{filename}_data_{algoname}{SDStxt}_{T}')
            else: write_at_end(stats_ar([As[i-1], As[i]])+'\n', 
                folder, f'{filename}_data_{algoname}{SDStxt}_{T}')

    data = ""
    for x in X: data += str(x)+'\n'
    write_at_begin(data, folder, f'{filename}_X_{algoname}{SDStxt}_{T}')
    draw_networks(As, f'{filename}_{algoname}_{T}', showall=False, titlename=algoname)