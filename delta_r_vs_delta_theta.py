import os
import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

from matplotlib import gridspec

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
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(delta_rs, delta_thetas, disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(delta_rs, delta_thetas, disc_fractions_IT20):
        """
        Plot the angular separation between the densest pixel and the angular momentum versus the distance between the centre of mass and
        the centre of potential normalised wrt the half-mass.
        :param delta_rs: from read_add_attributes.py.
        :param delta_thetas: from read_add_attributes.py.
        :param disc_fractions_IT20: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(1, 2, wspace=0.0, width_ratios=[1, 0.05])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[0, 1])
        plot_tools.set_axis(axis00, xlim=[0, 2], ylim=[0, 180], xlabel=r'$\mathrm{\delta_{r}/R_{hm}}$', ylabel=r'$\mathrm{\delta_{\theta}/\degree}$')
        
        # sc = axis00.scatter(delta_rs, delta_thetas[:, 0], c=disc_fractions_IT20, cmap='coolwarm_r', vmin=0, vmax=1, s=5)
        hb = axis00.hexbin(delta_rs, delta_thetas[:, 0], gridsize=100, cmap='nipy_spectral_r')
        plot_tools.create_colorbar(axis10, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'vertical')
        # plot_tools.create_colorbar(axis10, sc, r'$\mathrm{D/T_{30\degree}}$', 'vertical')
        
        plt.savefig(plots_path + 'DRDT' + '-' + date + '.png', bbox_inches='tight')  # Save the figure.
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = DeltaRVsDeltaTheta(simulation_path, tag)
