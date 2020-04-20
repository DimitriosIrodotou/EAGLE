import os
import re
import time
import warnings
import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

from matplotlib import gridspec

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DiscToTotalVsMorphologicalParameters:
    """
    For all galaxies create: a disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over
    dispersion plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        """
        start_local_time = time.time()  # Start the local time.
        
        disc_fractions = np.load(data_path + 'glx_disc_fractions.npy')
        kappas_corotation = np.load(data_path + 'glx_kappas_corotation.npy')
        disc_fractions_IT20 = np.load(data_path + 'glx_disc_fractions_IT20.npy')
        concentration_indices = np.load(data_path + 'glx_concentration_indices.npy')
        rotationals_over_dispersions = np.load(data_path + 'glx_rotationals_over_dispersions.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions_IT20, concentration_indices, kappas_corotation, disc_fractions, rotationals_over_dispersions)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished DTTMP for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def plot(self, disc_fractions_IT20, concentration_indices, kappas_corotation, disc_fractions, rotationals_over_dispersions):
        """
        Plot the disc to total ratio as a function of concentration index, kappa corotation, disc fraction and rotational over dispersion.
        :param disc_fractions_IT20: where the disc consists of particles whose angular momentum angular separation is 30deg from the densest pixel.
        :param concentration_indices: defined as R90/R50
        :param kappas_corotation: defined as angular kinetic energy over kinetic energy.
        :param disc_fractions: where the bulge is assumed to have zero net angualr momentum.
        :param rotationals_over_dispersions: defined as vrot/sigam
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
        
        ax10.set_xlim(0, 9)
        ax10.set_ylabel(r'$\mathrm{D/T_{30\degree}}$', size=16)
        cmap = matplotlib.cm.get_cmap('copper')
        for a in [ax10, ax11, ax12, ax13]:
            a.set_ylim(0, 1)
            a.set_facecolor(cmap(0))
            a.grid(True, which='both', axis='both')
            a.tick_params(direction='out', which='both', top='on', right='on', left='on', labelsize=16)
        for a in [ax11, ax12, ax13]:
            a.set_yticklabels([])
        
        axes = [ax10, ax11, ax12, ax13]
        cbar_axes = [ax00, ax01, ax02, ax03]
        x_attributes = [concentration_indices, kappas_corotation, disc_fractions, rotationals_over_dispersions]
        labels = [r'$\mathrm{Concentration\;index}$', r'$\mathrm{\kappa_{co}}$', r'$\mathrm{D/T_{\vec{J}_{b}=0}}$', r'$\mathrm{V_{rot}/\sigma}$']
        for axis, cbar_axis, x_attribute, label in zip(axes, cbar_axes, x_attributes, labels):
            # Plot attributes #
            hb = axis.hexbin(x_attribute, disc_fractions_IT20, bins='log', gridsize=100, label=r'$D/T_{\vec{J}_{b} = 0}$', cmap=cmap)
            
            cbar = plt.colorbar(hb, cax=cbar_axis, orientation='horizontal')
            cbar.set_label(r'$\mathrm{log_{10}(Counts\;per\;hexbin) }$', size=16)
            cbar_axis.xaxis.tick_top()
            cbar_axis.xaxis.set_label_position("top")
            
            # Plot median and 1-sigma lines #
            x_value, median, shigh, slow = self.median_1sigma(x_attribute, disc_fractions_IT20, 0.03)
            axis.plot(x_value, median, color='silver', linewidth=5, zorder=5)
            axis.fill_between(x_value, shigh, slow, color='silver', alpha='0.5', zorder=5)
            
            axis.set_xlabel(label, size=16)
        # Save the plot #
        plt.savefig(plots_path + 'DTTMP' + '-' + date + '.png', bbox_inches='tight')
        return None
    
    
    @staticmethod
    def median_1sigma(x, y, delta):
        """
        Calculate the median and 1-sigma lines.
        :param x:
        :param y:
        :param delta:
        :return:
        """
        # Initialise arrays #
        nbin = int((max(np.log10(x)) - min(np.log10(x))) / delta)
        x_value = np.empty(nbin)
        median = np.empty(nbin)
        slow = np.empty(nbin)
        shigh = np.empty(nbin)
        x_low = min(np.log10(x))
        
        # Loop over all bins and calculate the median and 1-sigma lines #
        for i in range(nbin):
            index, = np.where((np.log10(x) >= x_low) & (np.log10(x) < x_low + delta))
            x_value[i] = np.mean(x[index])
            if len(index) > 0:
                median[i] = np.nanmedian(y[index])
            slow[i] = np.nanpercentile(y[index], 15.87)
            shigh[i] = np.nanpercentile(y[index], 84.13)
            x_low += delta
        
        return x_value, median, shigh, slow


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/DTTMP/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = DiscToTotalVsMorphologicalParameters(simulation_path, tag)
