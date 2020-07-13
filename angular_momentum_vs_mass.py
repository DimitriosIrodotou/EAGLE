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


class AngularMomentumVsMass:
    """
    For all galaxies create: a galactic angular momentum versus stellar mass colour-coded by disc to total ratio plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.
        
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        
        # Normalise disc fractions #
        # epsilon = 0.5 * (1 - np.cos(np.pi / 6))
        # disc_fractions_IT20 = np.divide(1, 1 - epsilon) * np.abs(disc_fractions_IT20 - epsilon)
        
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(glx_stellar_masses, disc_fractions_IT20, glx_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished AngularMomentumVsMass for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(glx_stellar_masses, disc_fractions_IT20, glx_stellar_angular_momenta):
        """
        Plot galactic angular momentum versus stellar mass colour-coded by disc to total ratio.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param glx_stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(10, 7.5))
        gs = gridspec.GridSpec(2, 1, wspace=0.0, hspace=0.05, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis10 = figure.add_subplot(gs[1, 0])
        plot_tools.set_axis(axis10, xlim=[5e9, 1e12], ylim=[1e0, 1e5], xscale='log', yscale='log',
                            xlabel=r'$\mathrm{log_{10}(M_{\bigstar}/M_{\odot})}$',
                            ylabel=r'$\mathrm{(|\vec{J}_{\bigstar}|/M_{\bigstar})/(kpc\;km\;s^{-1})}$', aspect=None, which='major')
        
        # Plot the specific galactic angular momentum versus stellar mass colour-coded by disc to total ratio #
        spc_stellar_angular_momenta = np.linalg.norm(glx_stellar_angular_momenta, axis=1) / glx_stellar_masses
        sc = axis10.scatter(glx_stellar_masses, spc_stellar_angular_momenta, c=disc_fractions_IT20, s=8, cmap='seismic_r')
        plot_tools.create_colorbar(axis00, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        # Read observational data from OG13 and FR18 #
        OG13 = np.genfromtxt('./observational_data/OG_1312.4543/Figure7_stars.csv', delimiter=',', names=['Mstar', 'jstar'])
        FR18 = np.genfromtxt('./observational_data/FR_1808.02525/Figure1_FR.csv', delimiter=',', names=['Mstar', 'jstar'])
        
        # Plot observational data from OG13 and FR18 #
        axis10.scatter(np.power(10, OG13['Mstar']), np.power(10, OG13['jstar']), edgecolor='black', color='lime', s=50, marker='^',
                       label=r'$\mathrm{Obreschkow\;&\;Glazebrook\;13}$', zorder=4)
        axis10.plot(np.power(10, FR18['Mstar'][0:2]), np.power(10, FR18['jstar'][0:2]), color='blue', lw=3, linestyle='dashed',
                    label=r'$\mathrm{Fall\;&\;Romanowsky\;18:Discs}$', zorder=4)
        axis10.plot(np.power(10, FR18['Mstar'][2:4]), np.power(10, FR18['jstar'][2:4]), color='red', lw=3, linestyle='dashed',
                    label=r'$\mathrm{Fall\;&\;Romanowsky\;18:Bulges}$', zorder=4)
        
        # Create the legend, save and close the figure #
        plt.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        plt.savefig(plots_path + 'AM_M' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = AngularMomentumVsMass(simulation_path, tag)
