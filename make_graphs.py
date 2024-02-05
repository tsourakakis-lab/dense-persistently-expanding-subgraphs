import pandas as pd
import networkx as nx
import numpy as np
from help_functions import saveWeightedGraph, stats_for_G, nxgetEdges
import random
random.seed(10)


#Erdos Renyi (Random) Graph
def make_randomgraph(n = 242, T = 10, p=0.5, q=0.1):
    print('Making Random Graph...')
    G = nx.fast_gnp_random_graph(int(n), p = p, directed=False, seed = 10)
    A = nx.adjacency_matrix(G).todense()
    m = np.sum(A)
    to_shuffle = int(m*q)
    gedges = set()
    for i1 in range(len(A)):
            for i2 in range(i1, len(A)):
                if A[i1,i2]>0: gedges.add(f'{i1}_{i2}')
    avg_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for t in range(T):
        savename = 'randomgraph_'+str(t)+'.txt'
        gedges2 = []
        for i1 in range(len(A)):
            for i2 in range(len(A)):
                if A[i1,i2]>0: gedges2.append([i1, i2, 1])
        G = nx.from_numpy_matrix(A).to_undirected()
        avg_stats += stats_for_G(G)
        saveWeightedGraph(n, gedges2, './Graphs/'+savename) 
        list_to_shuffle = random.sample(gedges, to_shuffle)
        for el in list_to_shuffle: 
            gedges.remove(el)
            i1 = int(el.split('_')[0])
            i2 = int(el.split('_')[1])
            A[i1,i2] = 0
            A[i2,i1] = 0
        counter = to_shuffle
        while counter>0:
            i1 = random.randint(0, n-1)
            i2 = random.randint(0, n-1)
            mini = min(i1,i2)
            maxi = max(i1,i2)
            if f'{mini}_{maxi}' in gedges: continue
            if (A[mini,maxi] == 1): continue
            if(mini == maxi): continue
            counter-=1
            A[mini,maxi] = 1
            A[maxi,mini] = 1
            gedges.add(f'{mini}_{maxi}')
    avg_stats = np.array(avg_stats)/T
    print('Stats for Random graph:')
    print(f'n: {avg_stats[0]}\n<m>: {avg_stats[1]}\n<min deg>: {avg_stats[2]}\n<max deg>: {avg_stats[3]}\n<triangles>: {avg_stats[4]}\n<l2>: {avg_stats[5]}\n<lmax>: {avg_stats[6]}\nTotal number of timestamps: {T}\n\n')


#Tree Graph     
def make_treegraph(n = 242, T = 10, q=0.1):
    print('Making Tree Graph...')
    G = nx.random_tree(int(n), seed=10)
    A = nx.adjacency_matrix(G).todense()
    m = np.sum(A)
    to_shuffle = int(m*q)
    gedges = set()
    for i1 in range(len(A)):
            for i2 in range(i1, len(A)):
                if A[i1,i2]>0: gedges.add(f'{i1}_{i2}')
    avg_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for t in range(T):
        savename = 'treegraph_'+str(t)+'.txt'
        gedges2 = []
        for i1 in range(len(A)):
            for i2 in range(len(A)):
                if A[i1,i2]>0: gedges2.append([i1, i2, 1])
        G = nx.from_numpy_matrix(A).to_undirected()
        avg_stats += stats_for_G(G)
        saveWeightedGraph(n, gedges2, './Graphs/'+savename) 
        list_to_shuffle = random.sample(gedges, to_shuffle)
        for el in list_to_shuffle: 
            gedges.remove(el)
            i1 = int(el.split('_')[0])
            i2 = int(el.split('_')[1])
            A[i1,i2] = 0
            A[i2,i1] = 0
        counter = to_shuffle
        while counter>0:
            i1 = random.randint(0, n-1)
            i2 = random.randint(0, n-1)
            mini = min(i1,i2)
            maxi = max(i1,i2)
            if f'{mini}_{maxi}' in gedges: continue
            if (A[mini,maxi] == 1): continue
            if(mini == maxi): continue
            counter-=1
            A[mini,maxi] = 1
            A[maxi,mini] = 1
            gedges.add(f'{mini}_{maxi}')
    avg_stats = np.array(avg_stats)/T
    print('Stats for Tree graph:')
    print(f'n: {avg_stats[0]}\n<m>: {avg_stats[1]}\n<min deg>: {avg_stats[2]}\n<max deg>: {avg_stats[3]}\n<triangles>: {avg_stats[4]}\n<l2>: {avg_stats[5]}\n<lmax>: {avg_stats[6]}\nTotal number of timestamps: {T}\n\n')
        

#Animal Graph
def make_animal(n=202, T=6):
    print('Making Animal Graph...')
    df = pd.read_csv("./Graphs/animal/aves-wildbird-network.edges", 
                     header=None, names=['# source', ' target', ' weight', ' time'], sep=' ')
    Gs = [nx.Graph() for _ in range(T)]
    for i in range(T):
        for j in range(n):
            Gs[i].add_node(j)
    Ts = set()
    for index, row in df.iterrows():
        u = int(row['# source'])
        v = int(row[' target'])
        if u>=(n-1): continue
        if v>=(n-1): continue
        t = int(row[' time'])-1
        Ts.add(t)
        w = float(row[' weight'])
        if Gs[t].has_edge(u, v): Gs[t][u][v]['weight'] += w
        else: Gs[t].add_edge(u, v, weight = w)
    to_save = []
    for i in range(T):
        if Gs[i].number_of_edges() > 0: to_save.append(i)
    j = 0
    avg_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in to_save:
        avg_stats += stats_for_G(Gs[i])
        savename = 'animal_'+str(j)+'.txt'
        j+=1
        saveWeightedGraph(n, nxgetEdges(Gs[i]), './Graphs/'+savename) 
    avg_stats = np.array(avg_stats)/len(to_save)
    print('Stats for Animal graph:')
    print(f'n: {avg_stats[0]}\n<m>: {avg_stats[1]}\n<min deg>: {avg_stats[2]}\n<max deg>: {avg_stats[3]}\n<triangles>: {avg_stats[4]}\n<l2>: {avg_stats[5]}\n<lmax>: {avg_stats[6]}\nTotal number of timestamps: {len(Ts)}\n\n')


#Email Graph
def make_email(n = 167, T = 10, min_ = 1285884492, max_ = 1262454010):
    print('Making Email Graph...')
    df = pd.read_csv("./Graphs/email/ia-radoslaw-email.edges", header=None,comment='%',
                     names=['# source', ' target', ' w', ' time'], sep=r'[ ]{1,}')
    Gs = [nx.Graph() for _ in range(T)]
    for i in range(T):
        for j in range(n):
            Gs[i].add_node(j)
    interv = (max_ - min_)/T
    Ts = set()
    for index, row in df.iterrows():
        u = int(row['# source'])
        v = int(row[' target'])
        if u>=(n-1): continue
        if v>=(n-1): continue
        Ts.add(row[' time'])
        t = int((row[' time']-min_)/interv) 
        if t == T: t=T-1
        if Gs[t].has_edge(u, v): Gs[t][u][v]['weight'] += 1
        else: Gs[t].add_edge(u, v, weight = 1)
    to_save = []
    for i in range(T):
        if Gs[i].number_of_edges() > 0: to_save.append(i)
    j = 0
    avg_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in to_save:
        avg_stats += stats_for_G(Gs[i])
        savename = 'email_'+str(j)+'.txt'
        j+=1
        saveWeightedGraph(n, nxgetEdges(Gs[i]), './Graphs/'+savename) 
    avg_stats = np.array(avg_stats)/len(to_save)
    print('Stats for Email graph:')
    print(f'n: {avg_stats[0]}\n<m>: {avg_stats[1]}\n<min deg>: {avg_stats[2]}\n<max deg>: {avg_stats[3]}\n<triangles>: {avg_stats[4]}\n<l2>: {avg_stats[5]}\n<lmax>: {avg_stats[6]}\nTotal number of timestamps: {len(Ts)}\n\n')


#School Graph
def make_school(n = 242, T = 10, min_ = 31220, max_ = 148120):
    print('Making School Graph...')
    df = pd.read_csv("./Graphs/school/edges.csv")
    Gs = [nx.Graph() for _ in range(T)]
    for i in range(T):
        for j in range(n):
            Gs[i].add_node(j)
    interv = (max_ - min_)/T
    Ts = set()
    for index, row in df.iterrows():
        u = row['# source']
        v = row[' target']
        Ts.add(row[' time'])
        t = int((row[' time']-min_)/interv) 
        if t == T: t=T-1
        if Gs[t].has_edge(u, v): Gs[t][u][v]['weight'] += 1
        else: Gs[t].add_edge(u, v, weight = 1)
    to_save = []
    for i in range(T):
        if Gs[i].number_of_edges() > 0: to_save.append(i)
    j = 0
    avg_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for i in to_save:
        avg_stats += stats_for_G(Gs[i])
        savename = 'school_'+str(j)+'.txt'
        j+=1
        saveWeightedGraph(n, nxgetEdges(Gs[i]), './Graphs/'+savename) 
    avg_stats = np.array(avg_stats)/len(to_save)
    print('Stats for School graph:')
    print(f'n: {avg_stats[0]}\n<m>: {avg_stats[1]}\n<min deg>: {avg_stats[2]}\n<max deg>: {avg_stats[3]}\n<triangles>: {avg_stats[4]}\n<l2>: {avg_stats[5]}\n<lmax>: {avg_stats[6]}\nTotal number of timestamps: {len(Ts)}\n\n')


#Airport Graph
def make_airport(n = 242, T=10):
    print('Making Airport Graph...')
    edges = pd.read_csv('./Graphs/airport/edges.csv')
    nodes = pd.read_csv('./Graphs/airport/nodes.csv')
    airpordid_to_state = {}
    state_to_airportsids = {}

    for index, row in nodes.iterrows():
        airpordid_to_state[index] = row[' state_abr']
        
        if row[' state_abr'] not in state_to_airportsids: 
            state_to_airportsids[row[' state_abr']] = [row[' airport_id']]
        else:
            state_to_airportsids[row[' state_abr']].append(row[' airport_id'])
    eastern_states = {"FL", "GA", "SC", "NC", "VA", "DE", "NJ", "CT", "MA", "NH", "ME",
                      "VT", "NY", "PA", "MD", "WV", "OH", "KY", "TN", "AL", "MS", "IN", 
                      "MI"}
    western_states = {"WA", "OR", "CA", "MT", "ID", "NV", "AZ", "WY", "CO", "NM"}

    nodeid_counter = [0 for _ in range(len(nodes))]
    for _, row in edges.iterrows():
        u = row['# source']
        v = row[' target']
        nodeid_counter[u] += 1
        nodeid_counter[v] += 1
    ret_ids = sorted(range(len(nodeid_counter)), key=lambda k: -1*nodeid_counter[k])

    ids = {}
    i = 0
    for id in ret_ids[0:n]:   
        ids[id] = i
        i += 1
    
    file_object = open(f'./Graphs/airport/ids_airport.txt', 'a')
    for index, row in nodes.iterrows():
        cname = row[' city_name']
        if index in ids:
            file_object.write(f'{ids[index]} | {cname}\n')
    file_object.close()

        
    state_to_id = {}
    i = 0
    for state in state_to_airportsids:
        state_to_id[state] = i
        i += 1
        
    Gs = [nx.Graph() for _ in range(T)]
    Ts = np.unique(edges[' year'].values)
    for t in range(T):
        for j in range(len(ids)):
            Gs[t].add_node(j)
        edges_t = edges[(1990+t*int(20/T) <= edges[' year'])  
                        & (edges[' year'] < 1990+t*int(20/T)+int(20/T))]
        for _, row in edges_t.iterrows():
            u = row['# source']
            v = row[' target']
            if (u not in ids) or (v not in ids): continue
            if airpordid_to_state[u] == airpordid_to_state[v]: continue
            if (airpordid_to_state[u] not in eastern_states) and (airpordid_to_state[u] not in western_states): continue
            if (airpordid_to_state[v] not in eastern_states) and (airpordid_to_state[v] not in western_states): continue
            if (airpordid_to_state[u] in eastern_states) and (airpordid_to_state[v] in eastern_states): continue
            if (airpordid_to_state[u] in western_states) and (airpordid_to_state[v] in western_states): continue
            else:
                u = ids[u]
                v = ids[v]
                if Gs[t].has_edge(u, v): Gs[t][u][v]['weight'] += 1
                else: Gs[t].add_edge(u,v, weight = 1)
    avg_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for t in range(T):  
        avg_stats += stats_for_G(Gs[t])   
        edges_all = []
        for u,v in Gs[t].edges:
            edges_all.append([u,v,Gs[t][u][v]['weight']])
        if ids is not None: saveWeightedGraph(len(ids), edges_all, f'./Graphs/usflights_{t}.txt')
        else: saveWeightedGraph(len(nodes), edges_all, f'./Graphs/usflights_{t}.txt')
    avg_stats = np.array(avg_stats)/T
    print('Stats for Flights graph:')
    print(f'n: {avg_stats[0]}\n<m>: {avg_stats[1]}\n<min deg>: {avg_stats[2]}\n<max deg>: {avg_stats[3]}\n<triangles>: {avg_stats[4]}\n<l2>: {avg_stats[5]}\n<lmax>: {avg_stats[6]}\nTotal number of timestamps: {len(Ts)}\n\n')


#Generate all 6 Graphs
def generate_all_graphs():
    make_randomgraph(n = 250, T = 6)
    make_treegraph(n = 250, T = 6)
    make_animal(T = 6)
    make_email(T = 10)
    make_school(T = 10)
    make_airport(n = 250, T = 10)


if __name__ == "__main__": 
    generate_all_graphs()

