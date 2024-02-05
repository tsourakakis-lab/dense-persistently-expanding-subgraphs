from graph_loading import *
from help_functions import saving_results, spectrum_plots
from NDS import NDS
from SDP import run_SDP_DPES


#Function for executing the experiments
# - fname: folder/file name
# - load_function: function for loading data
# - sigmas: list of sigma parameters used in DPES algorithm
# - n: number of nodes (if graph has more than n nodes then take the induced subgraph spanned by the first n nodes)
# - T: number of timestamps
# - weighted: False(True resp.) for using the Adjacency (weighted Adjacecny resp.)
def run_experiments(fname, load_function, sigmas = [1], n = 100, 
                    T = range(5), weighted = True, write_extra = False):
    SDStxt=""
    Ys = []
    for sigma in sigmas:
        #filanems to save data
        filename = f'{fname}_{sigma}'
        #NDS
        if write_extra == False: 
            As = load_function(n, T, weighted)
            As_NDS, X_NDS = NDS(As)
            saving_results(As_NDS, X_NDS, filename, 'NDS', SDStxt, len(T), weighted)
        #SDP
        As = load_function(n, T, weighted)
        Yret, As_SDP, X_SDP = run_SDP_DPES(As, sigma)
        Ys.append(Yret)
        saving_results(As_SDP, X_SDP, filename, 'SDP', SDStxt, len(T), weighted, write_extra)
        #Saving spectrum plots
        if write_extra == False:  spectrum_plots(As_SDP, As_NDS, f'{filename}{SDStxt}_{len(T)}', showall=False, folder = 'Plots_')
    return Ys


#Tuning sigma parameter experiments. 
def run_experiments_SDP_thetas():
    sigmas = [0.1, 0.3, 0.5, 0.7, 0.9]
    run_experiments(f'animal_sigma', load_clique_plus_animal, sigmas, n=100, T = range(6), weighted = False, write_extra = True)




if __name__ == "__main__":  
    folder = 'Results_'  
    run_experiments(f'randomgraph', load_randomgraph, sigmas = [1], n=250, T = range(6), weighted = False)
    run_experiments(f'treegraph', load_treegraph, sigmas = [1], n=250, T = range(6), weighted = False)
    run_experiments(f'animal', load_animal, sigmas = [1], n=202, T = range(6), weighted = False)
    run_experiments(f'email', load_email, sigmas = [1], n=167, T = range(6), weighted = False)
    run_experiments(f'school', load_school, sigmas = [1], n=242, T = range(6), weighted = False)
    run_experiments(f'usflights', load_usflights, sigmas = [1], n=250, T = range(6), weighted = True) #use edge-weights