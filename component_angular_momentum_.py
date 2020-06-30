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
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class ComponentAngularMomentum:
    """
    For all galaxies create: a angular momentum versus stellar mass colour-coded by disc to total ratio plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        disc_stellar_angular_momenta = np.load(data_path + 'disc_stellar_angular_momenta.npy')
        bulge_stellar_angular_momenta = np.load(data_path + 'bulge_stellar_angular_momenta.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(glx_stellar_masses, disc_fractions_IT20, glx_stellar_angular_momenta, disc_stellar_angular_momenta, bulge_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished ComponentAngularMomentum for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, glx_stellar_masses, disc_fractions_IT20, glx_stellar_angular_momenta, disc_stellar_angular_momenta, bulge_stellar_angular_momenta):
        """
        Plot galactic angular momentum versus stellar mass colour-coded by disc to total ratio
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param glx_stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        
        plot_tools.set_axis(axis, xlim=[5e9, 1e12], ylim=[1e0, 1e6], xscale='log', yscale='log',
                            xlabel=r'$\mathrm{log_{10}(M_{\bigstar,comp}/M_{\odot})}$',
                            ylabel=r'$\mathrm{(|\vec{J}_{\bigstar,comp}|/M_{\bigstar, comp})/(kpc\;km\;s^{-1})}$', aspect=None, which='major')
        
        spc_disc_angular_momenta = np.divide(np.linalg.norm(disc_stellar_angular_momenta, axis=1), disc_fractions_IT20 * glx_stellar_masses)
        spc_bulge_angular_momenta = np.divide(np.linalg.norm(bulge_stellar_angular_momenta, axis=1), (1 - disc_fractions_IT20) * glx_stellar_masses)
        plt.scatter(disc_fractions_IT20 * glx_stellar_masses, spc_disc_angular_momenta, c='tab:blue', s=8)
        plt.scatter((1 - disc_fractions_IT20) * glx_stellar_masses, spc_bulge_angular_momenta, c='tab:red', s=8)
        
        # Read observational data from FR13 and OG14 #
        FR13_D = np.genfromtxt('./observational_data/FR_1305.1626/Figure2_D.csv', delimiter=',', names=['Md', 'jd'])
        FR13_E = np.genfromtxt('./observational_data/FR_1305.1626/Figure2_E.csv', delimiter=',', names=['Mb', 'jb'])
        
        # Plot observational data from FR13 and OG14 #
        plt.scatter(np.power(10, FR13_D['Md']), np.power(10, FR13_D['jd']), edgecolor='black', color='blue', s=50, marker='s',
                    label=r'$\mathrm{Fall\; &\; Romanowsky\, 13:Discs}$', zorder=4)
        plt.scatter(np.power(10, FR13_E['Mb']), np.power(10, FR13_E['jb']), edgecolor='black', color='red', s=50, marker='s',
                    label=r'$\mathrm{Fall\; &\; Romanowsky\, 13:Bulges}$', zorder=4)
        
        # Create the legend and save the figure #
        plt.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        plt.savefig(plots_path + 'CAM' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentAngularMomentum(simulation_path, tag)
