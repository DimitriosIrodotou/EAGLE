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


class ComponentAge:
    """
    For all components create: an average birth scale factor versus galaxy mass colour-coded by disc to total ratio plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.
        
        disc_as = np.load(data_path + 'disc_as.npy')
        bulge_as = np.load(data_path + 'bulge_as.npy')
        glx_stellar_masses = np.load(data_path + 'glx_stellar_masses.npy')
        glx_disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_as, bulge_as, glx_stellar_masses, glx_disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished ComponentAge for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    @staticmethod
    def plot(disc_as, bulge_as, glx_stellar_masses, glx_disc_fractions_IT20):
        """
        Plot the average component birth scale factor versus galaxy mass colour-coded by disc to total ratio.
        :param disc_as: defined as the mean birth scale factor for the disc component.
        :param bulge_as: defined as the mean birth scale factor for the bulge component.
        :param glx_stellar_masses: defined as the mass of all stellar particles within 30kpc from the most bound particle.
        :param glx_disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest
        pixel.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 2, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axiscbar = figure.add_subplot(gs[0, :])
        axis10, axis11 = figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])
        for axis in [axis10, axis11]:
            plot_tools.set_axis(axis, xlim=[1e9, 6e11], ylim=[1e-1, 1e0], xscale='log', yscale='log', xlabel=r'$\mathrm{M_{\bigstar}/M_{\odot}}$',
                                aspect=None)
        axis11.yaxis.tick_right()
        axis11.yaxis.set_label_position("right")
        axis10.set_ylabel(r'$\mathrm{\overline{\alpha_{bulge}}}$', size=16)
        axis11.set_ylabel(r'$\mathrm{\overline{\alpha_{disc}}}$', size=16)
        
        # Plot the average birth scale factor versus galaxy mass colour-coded by disc to total ratio #
        axis10.scatter(glx_stellar_masses, bulge_as, c=glx_disc_fractions_IT20, s=8, cmap='seismic_r')
        sc = axis11.scatter(glx_stellar_masses, disc_as, c=glx_disc_fractions_IT20, s=8, cmap='seismic_r')
        plot_tools.create_colorbar(axiscbar, sc, r'$\mathrm{D/T_{30\degree}}$', 'horizontal')
        
        # Plot median and 1-sigma lines #
        for component, axis in zip([bulge_as, disc_as], [axis10, axis11]):
            x_value, median, shigh, slow = plot_tools.median_1sigma(glx_stellar_masses, component, 0.1, log=True)
            axis.plot(x_value, median, color='black', linewidth=3, zorder=5)
            axis.fill_between(x_value, shigh, slow, color='black', alpha='0.5', zorder=5)
        
        # Save and close the figure #
        plt.savefig(plots_path + 'CA' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ComponentAge(simulation_path, tag)
