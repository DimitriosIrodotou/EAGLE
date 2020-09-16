import os
import re
import time
import warnings
import matplotlib
import plot_tools

matplotlib.use('Agg')

import numpy as np
import healpy as hlp
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt

from scipy.special import gamma
from astropy_healpix import HEALPix
from scipy.optimize import curve_fit
from plot_tools import RotateCoordinates

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class SampleSurfaceDensityProfiles:
    """
    For a sample of galaxies create: surface density profiles plot.
    """


    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        group_numbers = [10, 12, 17, 23, 25, 34, 39, 42, 53, 60, 62, 66, 82, 91, 93, 100]
        group_numbers = [5, 2, 20, 14, 3, 18, 39, 25]

        # Generate the figure and define its parameters #
        figure, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.31)

        for group_number, axis in zip(group_numbers, axes.flatten()):
            for subgroup_number in range(0, 1):  # Get centrals only.
                start_local_time = time.time()  # Start the local time.

                # Load the data #
                stellar_data_tmp = np.load(
                    data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy', allow_pickle=True)
                stellar_data_tmp = stellar_data_tmp.item()
                print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

                # Plot the data #
                start_local_time = time.time()  # Start the local time.

                self.plot(axis, stellar_data_tmp, group_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')

        # Save and close the figure #
        plt.savefig(plots_path + 'SSDP' + '-' + date + '.png', bbox_inches='tight')
        plt.close()
        print('Finished SampleSurfaceDensityProfiles for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')


    @staticmethod
    def plot(axis, stellar_data_tmp, group_number):
        """
        Plot a sample of HEALPix histograms.
        :param axis: from __init__.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param group_number: from read_add_attributes.py.
        :return: None
        """


        # Declare the Sersic, exponential and total profiles and Sersic parameter b #
        def sersic_profile(r, I_0b, b, n):
            """
            Calculate a Sersic profile.
            :param r: radius.
            :param I_0b: Spheroid central intensity.
            :param b: Sersic b parameter
            :param n: Sersic index
            :return: I_0b * np.exp(-(r / b) ** (1 / n))
            """
            return I_0b * np.exp(-(r / b) ** (1 / n))  # b = R_eff / b_n ^ n


        def exponential_profile(r, I_0d, R_d):
            """
            Calculate an exponential profile.
            :param r: radius
            :param I_0d: Disc central intensity.
            :param R_d: Disc scale length.
            :return: I_0d * np.exp(-r / R_d)
            """
            return I_0d * np.exp(-r / R_d)


        def total_profile(r, I_0d, R_d, I_0b, b, n):
            """
            Calculate a total (Sersic + exponential) profile.
            :param r: radius.
            :param I_0d: Disc central intensity.
            :param R_d: Disc scale length.
            :param I_0b: Spheroid central intensity.
            :param b: Sersic b parameter.
            :param n: Sersic index.
            :return: exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
            """
            y = exponential_profile(r, I_0d, R_d) + sersic_profile(r, I_0b, b, n)
            return y


        def sersic_b_n(n):
            """
            Calculate the Sersic b parameter.
            :param n: Sersic index.
            :return: b_n
            """
            if n <= 0.36:
                b_n = 0.01945 + n * (- 0.8902 + n * (10.95 + n * (- 19.67 + n * 13.43)))
            else:
                x = 1.0 / n
                b_n = -1.0 / 3.0 + 2. * n + x * (4.0 / 405. + x * (46. / 25515. + x * (131. / 1148175 - x * 2194697. / 30690717750.)))
            return b_n


        # Generate the figure and define its parameters #
        plot_tools.set_axis(axis, xlim=[0.0, 30.0], ylim=[1e6, 1e10], xlabel=r'$\mathrm{R/kpc}$', ylabel=r'$\mathrm{\Sigma/(M_{\odot}\;kpc^{-2})}$',
                            yscale='log', aspect=None, which='major')

        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        coordinates, velocities, prc_angular_momentum, glx_angular_momentum = RotateCoordinates.rotate_Jz(stellar_data_tmp)

        # Calculate the ra and el of the (unit vector of) angular momentum for each particle #
        prc_unit_vector = prc_angular_momentum / np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis]
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        el = np.degrees(np.arcsin(prc_unit_vector[:, 2]))

        # Plot a HEALPix histogram #
        nside = 2 ** 4  # Define the resolution of the grid (number of divisions along the side of a base-resolution grid cell).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, el * u.deg)  # Create list of HEALPix indices from particles' ra and el.
        densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix grid cell.

        # Perform a top-hat smoothing on the densities #
        smoothed_densities = np.zeros(hp.npix)
        # Loop over all grid cells #
        for i in range(hp.npix):
            mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30degree cone search around each grid cell.
            smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.

        # Find location of density maximum #
        index_densest = np.argmax(smoothed_densities)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2

        # Calculate and plot the disc (spheroid) mass surface density as the mass within (outside) 30 degrees from the densest grid cell #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask, = np.where(angular_theta_from_densest < (np.pi / 6.0))
        spheroid_mask, = np.where(angular_theta_from_densest > (np.pi / 6.0))

        colors = ['tab:blue', 'tab:red']
        labels = ['Disc', 'Spheroid']
        labels2 = ['Exponential', 'Sersic']
        masks = [disc_mask, spheroid_mask]
        profiles = [exponential_profile, sersic_profile]
        for mask, color, profile, label, label2 in zip(masks, colors, profiles, labels, labels2):
            component_coordinates, component_velocities, component_data = RotateCoordinates.rotate_component(stellar_data_tmp, mask)
            cylindrical_distance = np.sqrt(component_coordinates[:, 0] ** 2 + component_coordinates[:, 1] ** 2)  # Radius of each particle.

            mass, edges = np.histogram(cylindrical_distance, bins=50, range=(0, 30), weights=component_data['Mass'])
            centers = 0.5 * (edges[1:] + edges[:-1])
            surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
            sden = mass / surface

            axis.scatter(centers, sden, c=color, marker='.', linestyle="None", label=label)

            try:
                if mask is disc_mask:
                    popt, pcov = curve_fit(profile, centers, sden, sigma=0.1 * sden, p0=[sden[0], 2])  # p0 = [I_0d, R_d]
                    axis.plot(centers, profile(centers, popt[0], popt[1]), c=color, label=label2)
                    sden_tmp = sden

                    # Calculate disc attributes #
                    I_0d, R_d = popt[0], popt[1]
                    disk_mass = 2.0 * np.pi * I_0d * R_d ** 2

                elif mask is spheroid_mask:
                    popt, pcov = curve_fit(profile, centers, sden, sigma=0.1 * sden, p0=[sden[0], 2, 4])  # p0 = [I_0b, b, n]
                    axis.plot(centers, profile(centers, popt[0], popt[1], popt[2]), c=color, label=label2)

                    # Calculate spheroid attributes #
                    I_0b, b, n = popt[0], popt[1], popt[2]
                    R_eff = b * sersic_b_n(n) ** n
                    spheroid_mass = np.pi * I_0b * R_eff ** 2 * gamma(2.0 / n + 1)

            except RuntimeError:
                print('Could not fit a Sersic or exponential profile')

        cylindrical_distance = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)  # Radius of each particle.

        mass, edges = np.histogram(cylindrical_distance, bins=50, range=(0, 30), weights=stellar_data_tmp['Mass'])
        centers = 0.5 * (edges[1:] + edges[:-1])
        surface = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        sden = mass / surface

        axis.scatter(centers, sden, c='k', marker='.', linestyle="None", label='Total')

        try:
            popt, pcov = curve_fit(total_profile, centers, sden, sigma=0.1 * sden, p0=[sden_tmp[0], 2, sden[0], 2, 4])  # p0 = [I_0d, R_d, I_0b, b, n]
            axis.plot(centers, total_profile(centers, popt[0], popt[1], popt[2], popt[3], popt[4]), c='k', label=r'$\mathrm{Total}$')

            # Calculate galactic attributes #
            I_0d, R_d, I_0b, b, n = popt[0], popt[1], popt[2], popt[3], popt[4]
            R_eff = b * sersic_b_n(n) ** n
            disk_mass = 2.0 * np.pi * I_0d * R_d ** 2
            spheroid_mass = np.pi * I_0b * R_eff ** 2 * gamma(2.0 / n + 1)
            disk_fraction = disk_mass / (spheroid_mass + disk_mass)

        except RuntimeError:
            print('Could not fit a Sersic+exponential profile')

        # plt.text(0.2, 0.7, '\n' r'$\mathrm{n}=%.2f$' '\n' r'$\mathrm{R_{d}}=%.2f$ kpc' '\n' r'$\mathrm{R_{eff}}=%.2f$ kpc' '\n' % (n, R_d, R_eff),
        #          transform=axis.transAxes, size=14)

        # Create the legend and save the figure #
        axis.legend(loc='upper centre', fontsize=12, frameon=False, numpoints=1, ncol=2)
        plt.text(0.0, 1.05, str(group_number), color='red', fontsize=14, transform=axis.transAxes)
        plt.text(0.3, 0.7, r'$\mathrm{D/T_{\Delta \theta<30\degree}=%.2f}$' % stellar_data_tmp['disc_fraction_IT20'], fontsize=14,
                 transform=axis.transAxes)
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = SampleSurfaceDensityProfiles(simulation_path, tag)
