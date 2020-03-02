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
from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematics

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create ra and dec plot.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADecDistribution:
    """
    Create a ra and dec plot with the angular momentum of particles for each galaxy.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        # Initialise arrays and a dictionary to store the data #
        glx_unit_vectors = []
        if not args.l:
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
            
            for group_number in np.sort(list(set(self.subhalo_data_tmp['GroupNumber']))):  # Loop over all masked haloes.
                for subgroup_number in range(0, 1):
                    start_local_time = time.time()  # Start the local time.
                    
                    glx_unit_vector = np.load(SavePath + 'glx_unit_vectors/' + 'glx_unit_vector_' + str(group_number) + '.npy')
                    glx_unit_vectors.append(glx_unit_vector)
                    
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
            
            np.save(SavePath + 'glx_unit_vectors/' + 'glx_unit_vectors', glx_unit_vectors)
        
        # Plot the data #
        start_local_time = time.time()  # Start the local time.
        glx_unit_vectors = np.load(SavePath + 'glx_unit_vectors/' + 'glx_unit_vectors.npy')
        self.plot(glx_unit_vectors)
        print('Plotted data for halo ' + ' in %.4s s' % (time.time() - start_local_time))
        print('–––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished RADecDistribution for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (
            time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(sim, tag):
        """
        Extract particle and subhalo attributes.
        :param sim: simulation directory
        :param tag: redshift folder
        :return: stellar_data, subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, sim, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'ParticleBindingEnergy', 'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return stellar_data, subhalo_data
    
    
    def mask_haloes(self):
        """
        A method to mask haloes.
        :return: subhalo_data_tmp
        """
        
        # Mask the data to select haloes more #
        mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 2.5e8)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[mask]
        
        return subhalo_data_tmp
    
    
    @staticmethod
    def plot(glx_unit_vectors):
        """
        A method to plot a HEALPix histogram.
        :param glx_unit_vectors: from mask_galaxies
        :return: None
        """
        
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)
        
        # Generate the figure #
        plt.close()
        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(projection='mollweide')
        
        plt.xlabel('RA ($\degree$)')
        plt.ylabel('Dec ($\degree$)')
        
        # Set manually the values of the ra axis #
        plt.annotate(r'0', xy=(0 - np.pi / 40, - np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'30', xy=(np.pi / 6 - np.pi / 25, - np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'-30', xy=(-np.pi / 6 - np.pi / 15, - np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'60', xy=(np.pi / 3 - np.pi / 25, - np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'-60', xy=(-np.pi / 3 - np.pi / 15, - np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'90', xy=(np.pi / 2 - np.pi / 25, - np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'-90', xy=(-np.pi / 2 - np.pi / 15, -np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'120', xy=(2 * np.pi / 3 - np.pi / 15, -np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'-120', xy=(-2 * np.pi / 3 - np.pi / 10, -np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'150', xy=(2.5 * np.pi / 3 - np.pi / 15, -np.pi / 65), xycoords='data', size=18)
        plt.annotate(r'-150', xy=(-2.5 * np.pi / 3 - np.pi / 10, -np.pi / 65), xycoords='data', size=18)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        glx_unit_vectors = glx_unit_vectors[~np.isnan(glx_unit_vectors).any(axis=1)]
        ra = np.degrees(np.arctan2(glx_unit_vectors[:, 1], glx_unit_vectors[:, 0]))
        dec = np.degrees(np.arcsin(glx_unit_vectors[:, 2]))
        
        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of points in each HEALPix pixel.
        
        # Sample a 360x180 grid in ra/dec #
        ra = np.linspace(-180.0, 180.0, num=360) * u.deg
        dec = np.linspace(-90.0, 90.0, num=180) * u.deg
        ra_grid, dec_grid = np.meshgrid(ra, dec)
        
        # Find density at each coordinate position #
        coordinate_index = hp.lonlat_to_healpix(ra_grid, dec_grid)
        density_map = density[coordinate_index]
        
        # Display data on a 2D regular raster and create a pseudo-color plot #
        im = plt.imshow(density_map, cmap='magma_r', aspect='auto',vmin=1)
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal')
        cbar.set_label('$\mathrm{Number\; of\; galaxies\; per\; grid\; cell}$')
        plt.pcolormesh(np.radians(ra), np.radians(dec), density_map, cmap='magma_r')
        
        # Save the plot #
        plt.savefig(outdir + 'RDD' + '-' + date + '.png', bbox_inches='tight')
        return None


if __name__ == '__main__':
    tag = '027_z000p101'
    sim = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RDD/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = RADecDistribution(sim, tag)