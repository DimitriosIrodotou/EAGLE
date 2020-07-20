import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import matplotlib.pyplot as plt

from plot_tools import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class BarStrength:
    """
    For each galaxies create: a bar strength plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.
        
        group_number, subgroup_number = 23, 0
        stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                   allow_pickle=True)
        stellar_data_tmp = stellar_data_tmp.item()
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(stellar_data_tmp, group_number, subgroup_number)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished BarStrength for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_data_tmp, group_number, subgroup_number):
        """
        Plot the bar strength radial profile.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: unique halo number.
        :param subgroup_number: unique subhalo number.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 10], ylim=[-0.1, 1.1], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{A_{2}}$', aspect=None, which='major')
        
        # Rotate coordinates and velocities of stellar particles so the galactic angular momentum points along the x axis #
        stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_Jz(
            stellar_data_tmp)
        
        # Calculate and plot the bar strength from Fourier modes of surface density as a function of radius plot #
        n_bins = 40  # Number of radial bins.
        r = np.sqrt(stellar_data_tmp['Coordinates'][:, 0] ** 2 + stellar_data_tmp['Coordinates'][:, 1] ** 2)  # Radius of each particle.
        
        # Split up galaxy in radius bins and calculate Fourier components #
        r_m, beta_2, alpha_0, alpha_2 = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
        for i in range(0, n_bins):
            r_s = float(i) * 0.25
            r_b = float(i) * 0.25 + 0.25
            r_m[i] = float(i) * 0.25 + 0.125
            xfit = stellar_data_tmp['Coordinates'][:, 0][(r < r_b) & (r > r_s)]
            yfit = stellar_data_tmp['Coordinates'][:, 1][(r < r_b) & (r > r_s)]
            for k in range(0, len(xfit)):
                th_i = np.arctan2(yfit[k], xfit[k])
                alpha_0[i] = alpha_0[i] + 1
                alpha_2[i] = alpha_2[i] + np.cos(2 * th_i)
                beta_2[i] = beta_2[i] + np.sin(2 * th_i)
        
        a2 = np.divide(np.sqrt(alpha_2[:] ** 2 + beta_2[:] ** 2), alpha_0[:])
        
        plt.plot(r_m, a2, label='Bar strength: %.2f' % max(a2))  # Plot the bar strength radial profile.
        
        # Create the legends and save and close the figure #
        plt.legend(loc='upper left', fontsize=12, frameon=False, scatterpoints=3)
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'BS' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/BS/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = BarStrength(simulation_path, tag)
