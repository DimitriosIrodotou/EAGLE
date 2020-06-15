import re
import time
import warnings
import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec
from astropy_healpix import HEALPix

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class DiscToTotalProbabilityDensityFunction:
    """
    Create a PDF of the disc to total ratio.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        start_local_time = time.time()  # Start the local time.
        
        masses = np.load(data_path + 'glx_masses.npy')
        subgroup_numbers = np.load(data_path + 'subgroup_numbers.npy')
        disc_fractions_IT20 = np.load(data_path + 'disc_fractions_IT20.npy')
        print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(masses, subgroup_numbers, disc_fractions_IT20)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print(
            'Finished BTTPDF for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        :return: stellar_data, subhalo_data.
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'ParticleBindingEnergy', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e9)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: stellar_data_tmp, prc_unit_vector
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        halo_mask, = np.where((self.subhalo_data_tmp['GroupNumber'] == group_number) & (self.subhalo_data_tmp['SubGroupNumber'] == subgroup_number))
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        galaxy_mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(self.stellar_data['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask], axis=1) <= 30.0))  # In kpc.
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[galaxy_mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = stellar_data_tmp['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask]
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass'], axis=0))  # In km s^-1.
        stellar_data_tmp['Velocity'] = stellar_data_tmp['Velocity'] - CoM_velocity
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        prc_unit_vector = prc_angular_momentum / np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis]
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.
        
        # Find location of density maximum and plot its positions and the ra (lon) and dec (lat) of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        glx_mass = np.sum(stellar_data_tmp['Mass'])
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask = np.where(angular_theta_from_densest < (np.pi / 6.0))
        disc_fraction_IT20 = np.sum(stellar_data_tmp['Mass'][disc_mask]) / np.sum(stellar_data_tmp['Mass'])
        
        return disc_fraction_IT20, glx_mass
    
    
    @staticmethod
    def plot(masses, subgroup_numbers, disc_fractions_IT20):
        """
        A method to plot a HEALPix histogram.
        :param disc_fractions_IT20: from mask_galaxies.
        :return: None
        """
        
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(0, figsize=(20, 15))
        
        gs = gridspec.GridSpec(2, 2)
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax11 = plt.subplot(gs[1, 1])
        
        for axis in [ax00, ax10, ax01, ax11]:
            axis.grid(True)
            axis.set_ylim(0.0, 1.2)
        
        ax10.set_xscale('log')
        ax00.set_xlabel(r'$\mathrm{(B/T)_{\bigstar}}$', size=16)
        ax00.set_ylabel(r'$\mathrm{f(B/T)_{\bigstar}}$', size=16)
        ax10.set_ylabel(r'$\mathrm{f(B/T>0.5)}$', size=16)
        ax10.set_xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$', size=16)
        
        mass_mask = np.where(masses > 1e10)
        bulge_fraction = 1 - disc_fractions_IT20
        
        # Plots BBT19 bar's midpoints #
        BBT19 = np.genfromtxt('./Obs_Data/BBT19.csv', delimiter=',', names=['BT', 'f'])
        ax00.scatter(BBT19['BT'], BBT19['f'], color='red', s=3, marker='_', zorder=2, label="$\mathrm{Bluck+19}$")
        
        # Weight each bin by its contribution to the total number of values and create a histogram #
        weights = np.ones_like(bulge_fraction[mass_mask]) / float(len(bulge_fraction[mass_mask]))
        ax00.hist(bulge_fraction[mass_mask], align='left', weights=weights, histtype='step', edgecolor='black', bins=20)
        figure.text(0.0, 0.95, r'$\mathrm{M_{\bigstar}>10^{10}M_{\odot}}$', fontsize=16, transform=ax00.transAxes)
        
        # Put galaxies into bins #
        bins = np.logspace(9, 11.5 + 0.1, 20)
        bin_index = np.digitize(masses, bins)
        yBulge = np.empty(len(bins) - 1)
        glx_mass_bins = np.empty(len(bins) - 1)
        
        # Loop over bins and count fractions in each class #
        for iBin in range(len(bins) - 1):
            glx_mass_bins[iBin] = 0.5 * (bins[iBin] + bins[iBin + 1])
            indThisBin, = np.where(bin_index == iBin + 1)
            allBin = len(indThisBin)
            yBulge[iBin] = len(np.where(bulge_fraction[indThisBin] > 0.5)[0]) / float(allBin)
        
        # Plot data #
        ax10.plot(glx_mass_bins, yBulge, color='blue', lw=2, label="$\mathrm{Irodotou+18}$")
        
        satellites, = np.where(subgroup_numbers != 0)
        # Put galaxies into bins #
        bins = np.logspace(9, 11.5 + 0.1, 20)
        bin_index = np.digitize(masses[satellites], bins)
        yBulge = np.empty(len(bins) - 1)
        glx_mass_bins = np.empty(len(bins) - 1)
        
        # Loop over bins and count fractions in each class #
        for iBin in range(len(bins) - 1):
            glx_mass_bins[iBin] = 0.5 * (bins[iBin] + bins[iBin + 1])
            indThisBin, = np.where(bin_index == iBin + 1)
            allBin = len(indThisBin)
            yBulge[iBin] = len(np.where(bulge_fraction[indThisBin] > 0.5)[0]) / float(allBin)
        
        ax01.plot(glx_mass_bins, yBulge, color='blue', lw=2, label="$\mathrm{Irodotou+18}$")
        
        centrals, = np.where(subgroup_numbers == 0)
        # Put galaxies into bins #
        bins = np.logspace(9, 11.5 + 0.1, 20)
        bin_index = np.digitize(masses[centrals], bins)
        yBulge = np.empty(len(bins) - 1)
        glx_mass_bins = np.empty(len(bins) - 1)
        
        # Loop over bins and count fractions in each class #
        for iBin in range(len(bins) - 1):
            glx_mass_bins[iBin] = 0.5 * (bins[iBin] + bins[iBin + 1])
            indThisBin, = np.where(bin_index == iBin + 1)
            allBin = len(indThisBin)
            yBulge[iBin] = len(np.where(bulge_fraction[indThisBin] > 0.5)[0]) / float(allBin)
        
        ax11.plot(glx_mass_bins, yBulge, color='blue', lw=2, label="$\mathrm{Irodotou+18}$")
        
        # Save the figure #
        plt.savefig(plots_path + 'BTT_PDF' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = DiscToTotalProbabilityDensityFunction(simulation_path, tag)
