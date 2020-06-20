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

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class TullyFisher:
    """
    For all galaxies create: a Tully-Fisher relation plot.
    
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        disc_rotationals = np.load(data_path + 'disc_rotationals.npy')
        stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        bulge_rotationals = np.load(data_path + 'bulge_rotationals.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(stellar_masses, disc_rotationals, bulge_rotationals)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_masses, disc_rotationals, bulge_rotationals):
        """
        Plot the Tully-Fisher relation.
        :param stellar_masses: from read_add_attributes.py.
        :param disc_rotationals: from read_add_attributes.py.
        :param bulge_rotationals: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0.5, 3], ylim=[9.1, 11.9], xlabel=r'$\mathrm{log_{10}(V_{rot}/(km\;s^{-1}))}$',
                            ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$')
        
        axis.scatter(np.log10(disc_rotationals), np.log10(stellar_masses), c='tab:blue', s=5, label=r'$\mathrm{Disc}$')
        axis.scatter(np.log10(bulge_rotationals), np.log10(stellar_masses), c='tab:red', s=5, label=r'$\mathrm{Bulge}$')
        
        # Create the legend and save the figure #
        plt.legend(loc='upper right', fontsize=16, frameon=False, numpoints=1)
        plt.savefig(plots_path + 'TF' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = TullyFisher(simulation_path, tag)
