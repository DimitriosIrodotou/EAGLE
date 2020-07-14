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


class DiscToTotalVsGalacticAttributes:
    """
    For all galaxies create: a disc to total ratio as a function of mass, angular momentum.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        gaseous_masses = np.load(data_path + 'glx_gaseous_masses.npy')
        stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        star_formation_rates = np.load(data_path + 'glx_star_formation_rates.npy')
        stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        
        # Normalise disc fractions #
        epsilon = 0.5 * np.subtract(1, np.cos((np.pi / 6)))
        disc_fractions_IT20 = np.divide(1, np.subtract(1, epsilon)) * np.subtract(disc_fractions_IT20, epsilon)
        
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(gaseous_masses, stellar_masses, disc_fractions_IT20, star_formation_rates, stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print(
            'Finished DTT_GP for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, gaseous_masses, stellar_masses, disc_fractions_IT20, star_formation_rates, stellar_angular_momenta):
        """
        Plot the disc to total ratio as a function of mass, angular momentum.
        :param gaseous_masses: defined as the mass of all gaseous particles within 30kpc from the most bound particle.
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param star_formation_rates: defined as the star formation rate of all gaseous particles within 30kpc from the most bound particle.
        :param stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00, axis01, axis02, axis03 = figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]), figure.add_subplot(gs[0, 2]), figure.add_subplot(
            gs[0, 3])
        axis10, axis11, axis12, axis13 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]), figure.add_subplot(gs[1, 2]), figure.add_subplot(
            gs[1, 3])
        
        axis10.set_xlim(5e9, 5e11)
        axis11.set_xlim(1e1, 2e3)
        axis12.set_xlim(1e-3, 1e0)
        axis10.set_ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        for axis in [axis10, axis11, axis12, axis13]:
            axis.set_ylim(0, 1)
            axis.set_xscale('log')
            axis.grid(True, which='major', axis='both')
            axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=16)
        for axis in [axis11, axis12, axis13]:
            axis.set_yticklabels([])
        
        fgas = np.divide(gaseous_masses, gaseous_masses + stellar_masses)
        spc_stellar_angular_momenta = np.linalg.norm(stellar_angular_momenta, axis=1) / stellar_masses
        axes = [axis10, axis11, axis12, axis13]
        axescbar = [axis00, axis01, axis02, axis03]
        x_attributes = [stellar_masses, spc_stellar_angular_momenta, fgas[fgas > 0], star_formation_rates[star_formation_rates > 0]]
        y_attributes = [disc_fractions_IT20, disc_fractions_IT20, disc_fractions_IT20[fgas > 0], disc_fractions_IT20[star_formation_rates > 0]]
        labels = [r'$\mathrm{M_{\bigstar}/M_{\odot}}$', r'$\mathrm{(|\vec{J}_{\bigstar}|/M_{\bigstar})/(kpc\;km\;s^{-1})}$', r'$\mathrm{f_{gas}}$',
                  r'$\mathrm{SFR/(M_{\odot}\;yr^{-1})}$']
        for axis, axiscbar, x_attribute, y_attribute, label in zip(axes, axescbar, x_attributes, y_attributes, labels):
            # Plot attributes #
            hb = axis.hexbin(x_attribute, y_attribute, xscale='log', gridsize=50, cmap='CMRmap_r')
            plot_tools.create_colorbar(axiscbar, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
            
            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(x_attribute, y_attributes, bin_type='equal_width', n_bins=10, log=True)
            axis.plot(x_value, median, color='black', linewidth=3, zorder=5)
            axis.fill_between(x_value, shigh, slow, color='black', alpha='0.3', zorder=5)
            
            axis.set_xlabel(label, size=16)
        # Save the figure #
        plt.savefig(plots_path + 'DTT_GP' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalVsGalacticAttributes(simulation_path, tag)
