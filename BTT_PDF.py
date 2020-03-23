import os
import re
import time
import warnings
import argparse
import matplotlib

matplotlib.use('Agg')

import numpy as np
import seaborn as sns
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec
from astropy_healpix import HEALPix

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create D/T PDF plot.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class BulgeToTotalProbabilityDensityFunction:
    """
    Create a PDF of the disc to total ratio.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise empty arrays to hold the data #
        glx_masses, disc_fractions_IT20 = [], []
        
        if not args.l:
            # Extract particle and subhalo attributes and convert them to astronomical units #
            self.stellar_data, self.subhalo_data = self.read_galaxies(simulation_path, tag)
            print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
            
            for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all the accepted haloes
                for subgroup_number in range(0, 1):
                    if args.rs:  # Read and save data.
                        start_local_time = time.time()  # Start the local time.
                        
                        disc_fraction_IT20, glx_mass = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                        
                        # Save data in numpy arrays #
                        np.save(data_path + 'glx_masses/' + 'glx_mass_' + str(group_number) + '_' + str(subgroup_number), glx_mass)
                        np.save(data_path + 'disc_fraction_IT20_' + str(group_number) + '_' + str(subgroup_number), disc_fraction_IT20)
                        print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                            round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                        print('–––––––––––––––––––––––––––––––––––––––––––––')
                        p += 1
                    elif args.r:  # Read data.
                        start_local_time = time.time()  # Start the local time.
                        
                        disc_fraction_IT20, glx_mass = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                        glx_masses.append(glx_mass)
                        disc_fractions_IT20.append(disc_fraction_IT20)
                        print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                            round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                        print('–––––––––––––––––––––––––––––––––––––––––––––')
                        p += 1  # Increase the count by one.
                    
                    if args.l or args.rs:  # Load data.
                        start_local_time = time.time()  # Start the local time.
                        
                        glx_mass = np.load(data_path + 'glx_masses/' + 'glx_mass_' + str(group_number) + '_' + str(subgroup_number) + '.npy')
                        glx_masses.append(glx_mass.item())
                        print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                        print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            if args.l or args.rs:  # Load data.
                np.save(data_path + 'glx_masses/' + 'glx_masses', glx_masses)
                np.save(data_path + 'disc_fractions_IT20', disc_fractions_IT20)
        else:
            start_local_time = time.time()  # Start the local time.
            
            glx_masses = np.load(data_path + 'glx_masses/' + 'glx_masses.npy')
            disc_fractions_IT20 = np.load(data_path + 'disc_fractions_IT20/' + 'disc_fractions_IT20.npy')
            print('Loaded data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        
        self.plot(disc_fractions_IT20, glx_masses)
        print('Plotted data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished BTTPDF for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        :return: stellar_data, subhalo_data
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
        Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e8 Msun.
        :return: subhalo_data_tmp
        """
        
        # Mask the halo data #
        halo_mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e8)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[halo_mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        Mask galaxies and normalise data.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, prc_unit_vector
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        halo_mask = np.where((self.subhalo_data_tmp['GroupNumber'] == group_number) & (self.subhalo_data_tmp['SubGroupNumber'] == subgroup_number))[0]
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        galaxy_mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(np.subtract(self.stellar_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][halo_mask]),
                           axis=1) <= 30.0))  # kpc
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[galaxy_mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][halo_mask])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass'], axis=0))  # km s-1
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # Msun kpc km s-1
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of points in each HEALPix pixel.
        
        # Find location of density maximum and plot its positions and the ra and dec of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate disc mass fraction as the mass within 30 degrees from the densest pixel #
        glx_mass = np.sum(stellar_data_tmp['Mass'])
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask = np.where(angular_theta_from_densest < np.divide(np.pi, 6.0))
        disc_fraction_IT20 = np.divide(np.sum(stellar_data_tmp['Mass'][disc_mask]), np.sum(stellar_data_tmp['Mass']))
        
        return disc_fraction_IT20, glx_mass
    
    
    @staticmethod
    def plot(disc_fractions_IT20, glx_masses):
        """
        A method to plot a HEALPix histogram.
        :param disc_fractions_IT20: from mask_galaxies
        :return: None
        """
        
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)
        
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(0, figsize=(20, 15))
        
        gs = gridspec.GridSpec(2, 2)
        ax00 = plt.subplot(gs[0, 0])
        ax10 = plt.subplot(gs[1, 0])
        ax01 = plt.subplot(gs[0, 1])
        ax11 = plt.subplot(gs[1, 1])
        
        for a in [ax00, ax10, ax01, ax11]:
            a.grid(True)
            a.set_ylim(0.0, 1.0)
        
        ax10.set_xscale('log')
        ax00.set_xlabel(r'$\mathrm{(B/T)_{\bigstar}}$')
        ax00.set_ylabel(r'$\mathrm{f(B/T)_{\bigstar}}$')
        ax10.set_ylabel(r'$\mathrm{f(B/T>0.5)}$')
        ax10.set_xlabel(r'$\mathrm{M_{\bigstar} / M_{\odot}}$')
        
        mass_mask = np.where(glx_masses > 1e10)
        disc_fractions_IT20 = disc_fractions_IT20
        bulge_fraction = 1 - disc_fractions_IT20
        
        # Plots BBT19 bar's midpoints #
        BBT19 = np.genfromtxt('./Obs_Data/BBT19.csv', delimiter=',', names=['BT', 'f'])
        ax00.scatter(BBT19['BT'], BBT19['f'], color='red', s=3, marker='_', zorder=2, label="$\mathrm{Bluck+19}$")
        
        # Weight each bin by its contribution to the total number of values and make a histogram #
        weights = np.divide(np.ones_like(bulge_fraction[mass_mask]), float(len(bulge_fraction[mass_mask])))
        ax00.hist(bulge_fraction[mass_mask], align='left', weights=weights, histtype='step', edgecolor='black', bins=20)
        figure.text(0.0, 0.95, r'$\mathrm{M_{\bigstar}>10^{10}M_{\odot}}$', fontsize=16, transform=ax00.transAxes)
        
        # Put galaxies into bins #
        bins = np.logspace(9, 11.5 + 0.1, 20)
        bin_index = np.digitize(glx_masses, bins)
        yBulge = np.empty(len(bins) - 1)
        glx_mass_bins = np.empty(len(bins) - 1)
        
        # Loop over bins and count fractions in each class #
        for iBin in range(len(bins) - 1):
            glx_mass_bins[iBin] = 0.5 * (bins[iBin] + bins[iBin + 1])
            indThisBin = np.where(bin_index == iBin + 1)[0]
            allBin = len(indThisBin)
            yBulge[iBin] = len(np.where(bulge_fraction[indThisBin] > 0.5)[0]) / float(allBin)
        
        # Plot L-Galaxies data #
        ax10.plot(glx_mass_bins, yBulge, color='blue', lw=2, label="$\mathrm{Irodotou+18}$")
        
        # Save the plot #
        plt.savefig(plots_path + 'BTT_PDF' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/BTT_PDF/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = BulgeToTotalProbabilityDensityFunction(simulation_path, tag)
