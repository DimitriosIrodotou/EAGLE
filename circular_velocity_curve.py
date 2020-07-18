import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.constants import G

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class CircularVelocityCurve:
    """
    For each galaxies create: a circular velocity curve plot.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        # Load the data #
        start_local_time = time.time()  # Start the local time.
        
        group_number, subgroup_number = 25, 0
        stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                   allow_pickle=True)
        stellar_data_tmp = stellar_data_tmp.item()
        gaseous_data_tmp = np.load(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                   allow_pickle=True)
        gaseous_data_tmp = gaseous_data_tmp.item()
        
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(stellar_data_tmp, gaseous_data_tmp, group_number, subgroup_number)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished CircularVelocityCurve for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def plot(stellar_data_tmp, gaseous_data_tmp, group_number, subgroup_number):
        """
        Plot surface density profiles.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param gaseous_data_tmp: from read_add_attributes.py.
        :param group_number: unique halo number.
        :param subgroup_number: unique subhalo number.
        :return: None
        """
        # Generate the figure and define its parameters #
        figure, axis = plt.subplots(1, figsize=(10, 7.5))
        plot_tools.set_axis(axis, xlim=[0, 30], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{V_{c}/(km\;s^{-1})}$', aspect=None, which='major')
        
        # Calculate the spherical distance of each galaxy particle and sort their masses and velocities based on that #
        data = stellar_data_tmp, gaseous_data_tmp
        labels = r'$\mathrm{Stars}$', r'$\mathrm{Gas}$', r'$\mathrm{Dark\;matter}$', r'$\mathrm{Disc}$', r'$\mathrm{Bulge}$'
        for data, label in zip(data, labels):
            prc_spherical_radius = np.sqrt(np.sum(data['Coordinates'] ** 2, axis=1))
            sort = np.argsort(prc_spherical_radius)
            sorted_prc_spherical_radius = prc_spherical_radius[sort]
            cumulative_mass = np.cumsum(data['Mass'][sort])
            astronomical_G = G.to(u.km ** 2 * u.kpc * u.Msun ** -1 * u.s ** -2).value
            circular_velocity = np.sqrt(np.divide(astronomical_G * cumulative_mass, sorted_prc_spherical_radius))
            
            plt.plot(sorted_prc_spherical_radius, circular_velocity, label=label)  # Plot the circular velocity curve.
        
        # Calculate the spherical distance of each component particle and sort their masses and velocities based on that #
        data = stellar_data_tmp, stellar_data_tmp
        labels = r'$\mathrm{Disc}$', r'$\mathrm{Bulge}$'
        masks = stellar_data_tmp['disc_mask_IT20'], stellar_data_tmp['bulge_mask_IT20']
        for data, label, mask in zip(data, labels, masks):
            prc_spherical_radius = np.sqrt(np.sum(data['Coordinates'][mask] ** 2, axis=1))
            sort = np.argsort(prc_spherical_radius)
            sorted_prc_spherical_radius = prc_spherical_radius[sort]
            cumulative_mass = np.cumsum(data['Mass'][mask][sort])
            astronomical_G = G.to(u.km ** 2 * u.kpc * u.Msun ** -1 * u.s ** -2).value
            circular_velocity = np.sqrt(np.divide(astronomical_G * cumulative_mass, sorted_prc_spherical_radius))
            
            plt.plot(sorted_prc_spherical_radius, circular_velocity, label=label)  # Plot the circular velocity curve.
        
        # Create the legend, save and close the figure #
        plt.legend(loc='upper right', fontsize=12, frameon=False, numpoints=1)
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'CVC' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/CVC/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = CircularVelocityCurve(simulation_path, tag)
