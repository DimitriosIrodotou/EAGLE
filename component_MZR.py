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


class ComponentMZR:
    """
    For all galaxies create: a disc and bulge metallicity-mass relation colour-coded by DTT plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_metallicities = np.load(data_path + 'disc_metallicities.npy')
        bulge_metallicities = np.load(data_path + 'bulge_metallicities.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions_IT20, stellar_masses, disc_metallicities, bulge_metallicities)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished CMZR for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions_IT20, stellar_masses, disc_metallicities, bulge_metallicities):
        """
        Plot the disc and bulge metallicity-mass relation colour-coded by DTT.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param disc_metallicities: defined as the mass-weighted sum of each disc particle's metallicity.
        :param bulge_metallicities: defined as the mass-weighted sum of each bulge particle's metallicity.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        ax00 = figure.add_subplot(gs[0, :])
        ax10 = figure.add_subplot(gs[1, 0])
        ax11 = figure.add_subplot(gs[1, 1])
        
        for axis in [ax10, ax11]:
            axis.grid(True, which='both', axis='both')
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_ylim(1e-1, 1e1)
            axis.set_xlim(1e9, 6e11)
            axis.set_xlabel(r'$\mathrm{M_{\bigstar}/M_{\odot}}$', size=16)
            axis.tick_params(direction='out', which='both', top='on', right='on', left='on', labelsize=16)
        ax11.yaxis.tick_right()
        ax11.yaxis.set_label_position("right")
        ax10.set_ylabel(r'$\mathrm{Z_{bulge}/Z_{\odot}}$', size=16)
        ax11.set_ylabel(r'$\mathrm{Z_{disc}/Z_{\odot}}$', size=16)
        
        ax10.scatter(stellar_masses, bulge_metallicities, c=disc_fractions_IT20, s=8, cmap='seismic_r', vmin=0, vmax=1, marker='h')
        sc = ax11.scatter(stellar_masses, disc_metallicities, c=disc_fractions_IT20, s=8, cmap='seismic_r', vmin=0, vmax=1, marker='h')
        plot_tools.create_colorbar(ax00, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        # Save the plot #
        plt.savefig(plots_path + 'CMZR' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentMZR(simulation_path, tag)
