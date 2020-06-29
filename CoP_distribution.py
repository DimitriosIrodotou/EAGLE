import os
import re
import time
import warnings
import matplotlib
import plot_tools
import matplotlib.cbook

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class CoPDistribution:
    """
    For all galaxies create: a distribution of the centres of potential plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        CoPs = np.load(data_path + 'CoPs.npy')
        box_data = np.load(data_path + 'box_data.npy')
        box_data = box_data.item()
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(CoPs, box_data)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished MultipleDecomposition for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(CoPs, box_data):
        """
        Plot the distribution of the centres of potential.
        :param CoPs: defined as the coordinates of the most bound particle (i.e., most negative binding energy).
        :param box_data: data extracted from the header of SUBFIND.
        :return: None
        """
        box_side = box_data['BoxSize'] * 1e3 / box_data['HubbleParam']
        
        # Declare arrays to store the data.
        distances = np.zeros([len(CoPs), len(CoPs)])
        distances_flags = np.zeros(len(CoPs))
        
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 10))
        # plot_tools.set_axis(axis10, xlim=[0.5, 3.1], ylim=[9.5, 12.1], ylabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', aspect=None)
        
        # Calculate the distances between all centres of potential #
        CoPs = np.linalg.norm(CoPs, axis=1)  # Reduce the dimension from 3D to 2D.
        CoPs = np.linalg.norm(CoPs, axis=1)
        
        for i in np.arange(0, len(CoPs), 1):
            for j in np.arange(i + 1, len(CoPs), 1):
                # Periodic wrap coordinates around centre before calculating distances #
                CoPs[j] = np.mod(CoPs[j] - CoPs[i] + 0.5 * box_side, box_side) + CoPs[i] - 0.5 * box_side
                distances[i, j] = np.abs(CoPs[i] - CoPs[j])
                distances[j, i] = distances[i, j]
        
        for i in np.arange(0, len(CoPs), 1):
            mask, = np.where((distances[i, :] > 0) & (distances[i, :] < 30))
            if len(mask) == 0:
                distances_flags[i] = 0
            else:
                distances_flags[i] = 1
        
        mask, = np.where(distances_flags == 0)
        print(mask)
        print(len(distances_flags[mask]))
        mask, = np.where(distances_flags == 1)
        print(len(distances_flags[mask]))
        # sc = axis10.scatter(np.log10(glx_rotationals), np.log10(glx_stellar_masses), c=disc_fractions_IT20, s=10, cmap='seismic_r')
        # axis20.scatter(np.log10(disc_rotationals), np.log10(glx_stellar_masses), c='tab:blue', s=10, label=r'$\mathrm{Disc}$')
        # plot_tools.create_colorbar(axiscbar, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        plt.savefig(plots_path + 'CoPD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/tmp/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = CoPDistribution(simulation_path, tag)
