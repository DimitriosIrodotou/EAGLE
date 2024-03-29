import os
import re
import time
import warnings
import argparse
import matplotlib

matplotlib.use('Agg')

import numpy as np
import healpy as hlp
import matplotlib.cbook
import astropy.units as u
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
import eagle_IO.eagle_IO.eagle_IO as E

from matplotlib import gridspec
from plot_tools import RotateCoordinates

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create RA and Dec with attributes.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date.
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADecAttributes:
    """
    For each galaxy create: an RA and Dec plots from the angular momentum of particles colour-coded by different attributes.
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
        """
        
        p = 1  # Counter.
        # Initialise arrays and a dictionary to store the data #
        glx_unit_vector, stellar_data_tmp = [], {}
        
        if not args.l:
            # Extract particle and subhalo attributes and convert them to astronomical units #
            self.stellar_data, self.subhalo_data = self.read_attributes(simulation_path, tag)
            print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all masked haloes.
        for group_number in range(5, 6):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    
                    # Save data in numpy arrays #
                    np.save(data_path + 'glx_unit_vectors/' + 'glx_unit_vector_' + str(group_number) + '_' + str(subgroup_number), glx_unit_vector)
                    np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
                    print('Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                        time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask galaxies and normalise data.
                    print('Masked data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                        time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    glx_unit_vector = np.load(
                        data_path + 'glx_unit_vectors/' + 'glx_unit_vector_' + str(group_number) + '_' + str(subgroup_number) + '.npy')
                    stellar_data_tmp = np.load(
                        data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                        allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, glx_unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished RADecAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + '_' + str(tag) + ' in %.4s s' % (
            time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory.
        :param tag: redshift directory.
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
        for attribute in ['BirthDensity', 'Coordinates', 'GroupNumber', 'Mass', 'StellarFormationTime', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        stellar_data['BirthDensity'] *= u.g.to(u.Msun) / u.cm.to(u.kpc) ** 3
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
        :return: stellar_data_tmp
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
        
        # Normalise the coordinates and velocities wrt the centre of mass of the subhalo #
        stellar_data_tmp['Coordinates'] = stellar_data_tmp['Coordinates'] - self.subhalo_data_tmp['CentreOfPotential'][halo_mask]
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass']))  # In km s^-1.
        stellar_data_tmp['Velocity'] = stellar_data_tmp['Velocity'] - CoM_velocity
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s^-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s^-1.
        glx_unit_vector = glx_angular_momentum / np.linalg.norm(glx_angular_momentum)
        
        return stellar_data_tmp, glx_unit_vector
    
    
    @staticmethod
    def plot(stellar_data_tmp, glx_unit_vector, group_number, subgroup_number):
        """
        Plot RA and Dec from the angular momentum of particles colour-coded by different attributes.
        :param stellar_data_tmp: from read_add_attributes.py.
        :param glx_unit_vector: from mask_galaxies.
        :param group_number: from read_add_attributes.py.
        :param subgroup_number: from read_add_attributes.py.
        :return: None
        """
        # Generate the figure and define its parameters #
        plt.close()
        figure = plt.figure(figsize=(20, 15))
        
        gs = gridspec.GridSpec(2, 2)
        axis00 = figure.add_subplot(gs[0, 0], projection="mollweide")
        axis10 = figure.add_subplot(gs[1, 0], projection="mollweide")
        axis01 = figure.add_subplot(gs[0, 1], projection="mollweide")
        axis11 = figure.add_subplot(gs[1, 1], projection="mollweide")
        
        for axis in [axis00, axis10, axis01, axis11]:
            axis.set_xlabel('RA ($\degree$)')
            axis.set_ylabel('Dec ($\degree$)')
            axis.set_xticklabels([])
        
        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(stellar_data_tmp,
                                                                                                                                     glx_unit_vector)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixelisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create a list of HEALPix indices from particles' ra and dec.
        densities = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.

        # Perform a top-hat smoothing on the densities #
        smoothed_densities = np.zeros(hp.npix)
        # Loop over all grid cells #
        for i in range(hp.npix):
            mask = hlp.query_disc(nside, hlp.pix2vec(nside, i), np.pi / 6.0)  # Do a 30degree cone search around each grid cell.
            smoothed_densities[i] = np.mean(densities[mask])  # Average the densities of the ones inside and assign this value to the grid cell.


        # Find the location of the density maximum and plot its positions and the ra (lon) and dec (lat) of the galactic angular momentum #
        index_densest = np.argmax(smoothed_densities)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        axis00.annotate(r'Density maximum', xy=(lon_densest, lat_densest), xycoords='data', xytext=(0.78, 1.00), textcoords='axes fraction',
                      arrowprops=dict(arrowstyle="-", color='black', connectionstyle="arc3,rad=0"))  # Position of the denset pixel.
        axis00.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                     facecolors='none', zorder=5)  # Position of the galactic angular momentum.
        
        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)
        
        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = densities[coordinate_index]
        
        # Display data on a 2D regular raster and create a pseudo-color plot #
        im = axis00.imshow(density_map, cmap='nipy_spectral_r', aspect='auto', norm=matplotlib.colors.LogNorm(vmin=1))
        cbar = plt.colorbar(im, ax=axis00, orientation='horizontal')
        cbar.set_label('$\mathrm{Particles\;per\;grid\;cell}$', size=16)
        axis00.pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='nipy_spectral_r')
        
        # Plot the RA and Dec projection colour-coded by StellarFormationTime #
        scatter = axis01.scatter(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]), np.arcsin(prc_unit_vector[:, 2]),
                               c=stellar_data_tmp['StellarFormationTime'], cmap='jet_r', s=1, zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axis01, orientation='horizontal')
        cbar.set_label('$\mathrm{StellarFormationTime}$', size=16)
        
        # Plot the RA and Dec projection colour-coded by e^beta-1 #
        velocity_r_sqred = np.divide(np.sum(stellar_data_tmp['Velocity'] * stellar_data_tmp['Coordinates'], axis=1) ** 2,
                                     np.sum(stellar_data_tmp['Coordinates'] * stellar_data_tmp['Coordinates'], axis=1))
        beta = 1 - np.divide(np.sum(stellar_data_tmp['Velocity'] * stellar_data_tmp['Velocity'], axis=1) - velocity_r_sqred, 2 * velocity_r_sqred)
        
        scatter = axis10.scatter(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]), np.arcsin(prc_unit_vector[:, 2]), c=np.exp(beta - 1),
                               cmap='magma', vmax=1, s=1, zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axis10, orientation='horizontal')
        cbar.set_label(r'$\mathrm{exp(\beta-1)}$', size=16)
        
        # Plot the RA and Dec projection colour-coded by BirthDensity #
        scatter = axis11.scatter(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]), np.arcsin(prc_unit_vector[:, 2]),
                               c=np.log10(stellar_data_tmp['BirthDensity']), cmap='jet', s=1, zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axis11, orientation='horizontal')
        cbar.set_label('$\mathrm{log_{10}(Birth\;density)}$', size=16)
        
        # Save the figure #
        # plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(plots_path + str(group_number) + '_' + str(subgroup_number) + '-' + 'RDP' + '-' + date + '.png', bbox_inches='tight')
        
        return None


if __name__ == '__main__':
    # tag = '010_z005p000'
    # simulation_path = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'  # Path to G-EAGLE data.
    # plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RDP/G-EAGLE/'  # Path to save plots.
    # data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/RDP/G-EAGLE/'  # Path to save/load data.
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RDP/'  # Path to save plots.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    x = RADecAttributes(simulation_path, tag)
