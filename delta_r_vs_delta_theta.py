import os
import re
import time
import warnings
import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DeltaRVsDeltaTheta:
    """
    For all galaxies create: an angular separation between the densest pixel and the angular momentum versus the distance between the centre of
    mass and the centre of potential normalised wrt the half-mass plot.
    
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        delta_rs = np.load(data_path + 'glx_delta_rs.npy')
        delta_thetas = np.load(data_path + 'glx_delta_thetas.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(delta_rs, delta_thetas)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(delta_rs, delta_thetas):
        """
        Plot the angular separation between the densest pixel and the angular momentum versus the distance between the centre of mass and
        the centre of potential normalised wrt the half-mass.
        :param stellar_data_tmp: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        cmap = matplotlib.cm.get_cmap('nipy_spectral_r')
        axis.set_facecolor(cmap(0))
        plt.xlim(0, 2)
        plt.ylim(0, 180)
        plt.xlabel(r'$\mathrm{\delta_{r}/R_{hm}}$', size=16)
        plt.ylabel(r'$\mathrm{\delta_{\theta}/\degree}$', size=16)
        
        plt.hexbin(delta_rs, delta_thetas[:, 0], bins='log', gridsize=100, cmap='nipy_spectral_r')
        # plot_tools.create_colorbar(cbar_axis, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
        
        plt.savefig(plots_path + 'DRDT' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = DeltaRVsDeltaTheta(simulation_path, tag)
