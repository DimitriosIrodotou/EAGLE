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
        disc_stellar_angular_momenta = np.load(data_path + 'disc_stellar_angular_momenta.npy')
        bulge_stellar_angular_momenta = np.load(data_path + 'bulge_stellar_angular_momenta.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions_IT20, stellar_angular_momenta, gaseous_angular_momenta, disc_stellar_angular_momenta, bulge_stellar_angular_momenta)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished DTTVsEnvironment for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions_IT20, stellar_angular_momenta, gaseous_angular_momenta, disc_stellar_angular_momenta,
             bulge_stellar_angular_momenta):
        """
        Plot the angle between angular momentum of gaseous versus stellar discs.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param stellar_angular_momenta: defined as the sum of each stellar particle's angular momentum.
        :param gaseous_angular_momenta: defined as the sum of each gaseous particle's angular momentum.
        :param disc_stellar_angular_momenta: defined as the sum of each disc particle's angular momentum.
        :param bulge_stellar_angular_momenta: defined as the sum of each bulge particle's angular momentum.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(20, 7.5))
        gs = gridspec.GridSpec(2, 4, wspace=0.0, hspace=0.0, height_ratios=[0.05, 1])
        axis00 = figure.add_subplot(gs[0, 0])
        axis01 = figure.add_subplot(gs[0, 1])
        axis02 = figure.add_subplot(gs[0, 2])
        axis03 = figure.add_subplot(gs[0, 3])
        axis10 = figure.add_subplot(gs[1, 0])
        axis11 = figure.add_subplot(gs[1, 1])
        axis12 = figure.add_subplot(gs[1, 2])
        axis13 = figure.add_subplot(gs[1, 3])
        
        axis10.set_ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        
        for axis in [axis10, axis11, axis12, axis13]:
            axis.set_ylim(0, 1)
            axis.grid(True, which='major', axis='both')
            axis.tick_params(direction='out', which='both', top='on', right='on', labelsize=16)
        for axis in [axis11, axis12, axis13]:
            axis.set_yticklabels([])
        
        angle = np.divide(np.sum(stellar_angular_momenta * gaseous_angular_momenta, axis=1),
                          np.linalg.norm(stellar_angular_momenta, axis=1) * np.linalg.norm(gaseous_angular_momenta, axis=1))
        
        angle_components = np.divide(np.sum(disc_stellar_angular_momenta * bulge_stellar_angular_momenta, axis=1),
                                     np.linalg.norm(disc_stellar_angular_momenta, axis=1) * np.linalg.norm(bulge_stellar_angular_momenta, axis=1))
        axes = [axis10, axis11]  # , axis12, axis13]
        axescbar = [axis00, axis01]  # , axis02, axis03]
        x_attributes = [angle, angle_components]
        labels = [r'$\mathrm{(\vec{J}_{\bigstar}\cdot\vec{J}_{gas})/(|\vec{J}_{\bigstar}||\vec{J}_{gas}|)}$',
                  r'$\mathrm{(\vec{J}_{disc}\cdot\vec{J}_{bulge})/(|\vec{J}_{disc}||\vec{J}_{bulge}|)}$']
        for axis, axiscbar, x_attribute, label in zip(axes, axescbar, x_attributes, labels):
            hb = axis.hexbin(x_attribute, disc_fractions_IT20, bins='log', label=r'$D/T_{\vec{J}_{b} = 0}$', cmap='CMRmap_r')
            plot_tools.create_colorbar(axiscbar, hb, r'$\mathrm{Counts\;per\;hexbin}$', 'horizontal')
            
            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = plot_tools.binned_median_1sigma(x_attribute, disc_fractions_IT20, 0.05, log=False)
            axis.plot(x_value, median, color='black', linewidth=3, zorder=5)
            axis.fill_between(x_value, shigh, slow, color='black', alpha='0.3', zorder=5)
            
            axis.set_xlabel(label, size=16)
        
        # Save the figure #
        plt.savefig(plots_path + 'DTT_E' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DTTVsEnvironment(simulation_path, tag)
