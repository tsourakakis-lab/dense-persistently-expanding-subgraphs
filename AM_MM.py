from graph_loading import *
from help_functions import saving_results, spectrum_plots, read_nodes
import os

def run_AM_MM(fname, load_function, sigmas = [1], n = 100, T = range(5), weighted = True, foldername = './BestFriendsForever-BFF--master/java/experiments'):
    filenameMM = f'{foldername}/{fname}.txt_1.txt'
    filenameAM = f'./{foldername}/{fname}.txt_3.txt'
    for sigma1 in sigmas:
        filename = f'{fname}_{sigma1}'
        
        As = load_function(n, T, weighted)
        As_MM, X_MM = read_nodes(As, filenameMM)
        saving_results(As_MM, X_MM, filename, 'MM', '', len(T), weighted)
        
        As = load_function(n, T, weighted)
        As_AM, X_AM = read_nodes(As, filenameAM)
        saving_results(As_AM, X_AM, filename, 'AM', '', len(T), weighted)
        
        spectrum_plots(As_MM, As_AM, f'{filename}_{len(T)}', showall=False, labels = ['MM', 'AM'])
        
if __name__ == "__main__":  
    ma_mm_file = './BestFriendsForever-BFF--master/java/experiments'
    if os.path.exists(ma_mm_file): 
        run_AM_MM(f'randomgraph', load_randomgraph, sigmas = [1], n=250, T = range(6), weighted = False)
        run_AM_MM(f'treegraph', load_treegraph, sigmas = [1], n=250, T = range(6), weighted = False)
        run_AM_MM(f'animal', load_animal, sigmas = [1], n=202, T = range(6), weighted = False)
        run_AM_MM(f'email', load_email, sigmas = [1], n=167, T = range(6), weighted = False)
        run_AM_MM(f'school', load_school, sigmas = [1], n=242, T = range(6), weighted = False)
        run_AM_MM(f'usflights', load_usflights, sigmas = [1], n=250, T = range(6), weighted = True) #use edge-weights

