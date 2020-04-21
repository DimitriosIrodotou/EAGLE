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

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DTTVsAngularMomentumAlignment:
    """
    For all galaxies create: angle between angular momentum of gaseous versus stellar discs.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        """
        start_local_time = time.time()  # Start the local time.
        
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        gaseous_angular_momenta = np.load(data_path + 'glx_gaseous_angular_momenta.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions_IT20, stellar_angular_momenta, gaseous_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished DTT_AMA for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions_IT20, stellar_angular_momenta, gaseous_angular_momenta):
        """
        Plot the angle between angular momentum of gaseous versus stellar discs.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param stellar_angular_momenta: defined as the sum of each particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        ax00 = figure.add_subplot(gs[0, 0])
        ax10 = figure.add_subplot(gs[1, 0])
        
        ax10.grid(True, which='both', axis='both')
        # ax10.set_xscale('log')
        # ax10.set_yscale('log')
        # ax10.set_ylim(1e0, 1e5)
        # ax10.set_xlim(1e9, 1e12)
        ax10.set_ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        ax10.set_xlabel(r'$\mathrm{(\vec{J}_{\bigstar}\cdot\vec{J}_{gas})/(|\vec{J}_{\bigstar}||\vec{J}_{gas}|)}$', size=16)
        ax10.tick_params(direction='out', which='both', top='on', right='on', left='on', labelsize=16)
        
        angle = np.divide(np.sum(stellar_angular_momenta * gaseous_angular_momenta, axis=1),
                          np.linalg.norm(stellar_angular_momenta, axis=1) * np.linalg.norm(gaseous_angular_momenta, axis=1))
        sc = ax10.scatter(angle, disc_fractions_IT20, s=8)  # , cmap='RdYlBu_r', marker='h')
        # plot_tools.create_colorbar(ax00, sc, r'$\mathrm{B/T_{30\degree}}$', 'horizontal')
        
        # Save the plot #
        plt.savefig(plots_path + 'DTT_AMA' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DTTVsAngularMomentumAlignment(simulation_path, tag)
