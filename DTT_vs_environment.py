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


class DTTVsEnvironment:
    """
    For all galaxies create: angle between angular momentum of gaseous versus stellar discs.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
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
        
        print('Finished DTT_E for ' + re.split('Planck1/|/PE', simulation_path)[1] + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions_IT20, stellar_angular_momenta, gaseous_angular_momenta):
        """
        Plot the angle between angular momentum of gaseous versus stellar discs.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :param gaseous_angular_momenta: defined as the sum of each gaseous particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        ax00 = figure.add_subplot(gs[0, 0])
        ax01 = figure.add_subplot(gs[0, 1])
        ax02 = figure.add_subplot(gs[0, 2])
        ax03 = figure.add_subplot(gs[0, 3])
        ax10 = figure.add_subplot(gs[1, 0])
        ax11 = figure.add_subplot(gs[1, 1])
        ax12 = figure.add_subplot(gs[1, 2])
        ax13 = figure.add_subplot(gs[1, 3])
        
        ax10.set_ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        cmap = matplotlib.cm.get_cmap('copper')
        
        for axis in [ax10, ax11, ax12, ax13]:
            axis.set_ylim(0, 1)
            # axis.set_xscale('log')
            axis.set_facecolor(cmap(0))
            axis.grid(True, which='major', axis='both')
            axis.tick_params(direction='out', which='both', top='on', right='on', left='on', labelsize=16)
        for axis in [ax11, ax12, ax13]:
            axis.set_yticklabels([])
        
        angle = np.divide(np.sum(stellar_angular_momenta * gaseous_angular_momenta, axis=1),
                          np.linalg.norm(stellar_angular_momenta, axis=1) * np.linalg.norm(gaseous_angular_momenta, axis=1))
        
        axes = [ax10, ax11, ax12, ax13]
        cbar_axes = [ax00, ax01, ax02, ax03]
        x_attributes = [angle]
        y_attributes = [disc_fractions_IT20]
        labels = [r'$\mathrm{(\vec{J}_{\bigstar}\cdot\vec{J}_{gas})/(|\vec{J}_{\bigstar}||\vec{J}_{gas}|)}$']
        for axis, cbar_axis, x_attribute, y_attribute, label in zip(axes, cbar_axes, x_attributes, y_attributes, labels):
            hb = axis.hexbin(x_attribute, y_attribute, bins='log', gridsize=100, label=r'$D/T_{\vec{J}_{b} = 0}$', cmap=cmap)
            plot_tools.create_colorbar(cbar_axis, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
            
            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.median_1sigma(x_attribute, disc_fractions_IT20, 0.05, log=False)
            axis.plot(x_value, median, color='silver', linewidth=5, zorder=5)
            axis.fill_between(x_value, shigh, slow, color='silver', alpha='0.5', zorder=5)
            
            axis.set_xlabel(label, size=16)
        
        # Save the plot #
        plt.savefig(plots_path + 'DTT_E' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DTTVsEnvironment(simulation_path, tag)