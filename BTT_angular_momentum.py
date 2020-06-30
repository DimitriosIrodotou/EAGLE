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


class BulgeToTotalAngularMomentum:
    """
    For all galaxies create: a bulge to total ratio versus normalised specific galactic angular momentum plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        glx_stellar_angular_momenta = np.load(data_path + 'glx_stellar_angular_momenta.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(glx_stellar_masses, glx_disc_fractions_IT20, glx_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished BulgeToTotalAngularMomentum for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, glx_stellar_masses, glx_disc_fractions_IT20, glx_stellar_angular_momenta):
        """
        Plot bulge to total ratio versus normalised specific galactic angular momentum.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        pixel.
        :param glx_stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        
        plot_tools.set_axis(axis, xlim=[-6, -3], ylim=[0, 1],
                            xlabel=r'$\mathrm{(|\vec{J}_{\bigstar}|/M_{\bigstar}/M^{\alpha})/(kpc\;km\;s^{-1}\;M_{\odot}^{-\alpha})}$',
                            ylabel=r'$\mathrm{B/T_{30\degree}}$', aspect=None, which='major')
        
        spc_stellar_angular_momenta = np.linalg.norm(glx_stellar_angular_momenta, axis=1) / glx_stellar_masses
        spc_stellar_angular_momenta /= np.power(glx_stellar_masses, 0.67)
        mask, = np.where(spc_stellar_angular_momenta>-5)
        axis.hexbin(np.log10(spc_stellar_angular_momenta[mask]), 1 - glx_disc_fractions_IT20[mask], cmap='cool_r', bins='log',gridsize=50)
        
        # Read observational data from FR18 #
        FR18 = np.genfromtxt('./observational_data/FR_1808.02525/Figure3_classical.csv', delimiter=',', names=['BtT', 'jstarMalpha'])
        FR18_model = np.genfromtxt('./observational_data/FR_1808.02525/Figure3_model.csv', delimiter=',', names=['BtT', 'jstarMalpha'])
        
        # Plot observational data from OG14 and FR18 #
        axis.scatter(FR18['BtT'], FR18['jstarMalpha'], color='black', s=15, label=r'$\mathrm{Fall\; &\; Romanowsky}$', zorder=4)
        axis.scatter(FR18_model['BtT'], FR18_model['jstarMalpha'], color='orange', s=15, label=r'$\mathrm{Fall\; &\; Romanowsky}$', zorder=4)
        
        # Create the legend and save the figure #
        plt.legend(loc='bottom left', fontsize=12, frameon=False, numpoints=1)
        plt.savefig(plots_path + 'BTT_AM' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = BulgeToTotalAngularMomentum(simulation_path, tag)
