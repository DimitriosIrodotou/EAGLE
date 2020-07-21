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


class SFRVsAngularMomentum:
    """
    For all galaxies create: a star formation rate as a function of stellar angular momentum colour-coded by disc to total ratio.
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
        stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(stellar_masses, disc_fractions_IT20, star_formation_rates, stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished SFRVsAngularMomentum for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    @staticmethod
    def plot(stellar_masses, disc_fractions_IT20, star_formation_rates, stellar_angular_momenta):
        """
        Plot star formation rate as a function of stellar angular momentum colour-coded by disc to total ratio
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param star_formation_rates: defined as the star formation rate of all gaseous particles within 30kpc from the most bound particle.
        :param stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        
        axis10.grid(True, which='both', axis='both')
        axis10.set_ylabel(r'$\mathrm{log_{10}((SFR/M_{\bigstar})/yr^{-1})}$', size=16)
        axis10.set_xlabel(r'$\mathrm{(|\vec{J}_{\bigstar}|/M_{\bigstar})/(kpc\;km\;s^{-1})}$', size=16)
        axis10.tick_params(direction='out', which='major', top='on', right='on',  labelsize=16)
        
        spc_angular_momenta = np.linalg.norm(stellar_angular_momenta, axis=1) / stellar_masses
        spc_star_formation_rates = star_formation_rates / stellar_masses
        sc = axis10.scatter(np.log10(spc_angular_momenta[spc_star_formation_rates > 0]),
                          np.log10(spc_star_formation_rates[spc_star_formation_rates > 0]), c=disc_fractions_IT20[spc_star_formation_rates > 0], s=8,
                          cmap='seismic_r', vmin=0, vmax=1)
        plot_tools.create_colorbar(axis00, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        # Plot median and 1-sigma lines #
        x_value, median, shigh, slow = plot_tools.binned_median_1sigma(np.log10(spc_angular_momenta[spc_star_formation_rates > 0]),
                                                                np.log10(spc_star_formation_rates[spc_star_formation_rates > 0]), 0.1, log=False)
        axis10.plot(x_value, median, color='black', linewidth=5)
        axis10.fill_between(x_value, shigh, slow, color='black', alpha='03')
        
        # Save the figure #
        plt.savefig(plots_path + 'SFR_AM' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = SFRVsAngularMomentum(simulation_path, tag)
