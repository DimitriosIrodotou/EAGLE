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
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SFRVsMass:
    """
    For all galaxies create: star formation rate versus stellar mass colour-coded by disc to total ratio.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        star_formation_rates = np.load(data_path + 'glx_star_formation_rates.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(stellar_masses, disc_fractions_IT20, star_formation_rates)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished SFRVsMass for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, stellar_masses, disc_fractions_IT20, glx_star_formation_rates):
        """
        Plot star formation rate versus stellar mass colour-coded by disc to total ratio
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param glx_star_formation_rates: defined as the star formation rate of all gaseous particles within 30kpc from the most bound particle.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        
        axis10.grid(True, which='both', axis='both')
        axis10.set_xlim(9, 12)
        axis10.set_ylim(-3.5, 1.5)
        axis10.set_xlabel(r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$', size=16)
        axis10.set_ylabel(r'$\mathrm{log_{10}(SFR/(M_{\odot}\;yr^{-1}))}$', size=16)
        axis10.tick_params(direction='out', which='both', top='on', right='on',  labelsize=16)
        
        sc = axis10.scatter(np.log10(stellar_masses[glx_star_formation_rates > 0]), np.log10(glx_star_formation_rates[glx_star_formation_rates > 0]),
                          c=disc_fractions_IT20[glx_star_formation_rates > 0], s=8, cmap='seismic_r', vmin=0, vmax=1)
        plot_tools.create_colorbar(axis00, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(np.log10(stellar_masses[glx_star_formation_rates > 0]),
                                                                np.log10(glx_star_formation_rates[glx_star_formation_rates > 0]), 0.1, log=False)
        axis10.plot(x_value, median, color='black', linewidth=5, zorder=5)
        axis10.fill_between(x_value, shigh, slow, color='black', alpha='0.5', zorder=5)
        
        # Save the figure #
        plt.savefig(plots_path + 'SFR_M' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = SFRVsMass(simulation_path, tag)
