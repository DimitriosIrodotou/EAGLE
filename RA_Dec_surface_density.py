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
from morpho_kinematics import MorphoKinematics

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create RA and Dec.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADecSurfaceDensity:
    """
    Create a RA and Dec plot with the angular momemntum of particles for each galaxy.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise arrays and a dictionary to store the data #
        prc_unit_vector = []
        stellar_data_tmp = {}
        glx_unit_vector = []
        
        if not args.l:
            self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all the accepted haloes
        for group_number in range(1, 21):  # Loop over all the accepted haloes
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, prc_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    
                    # Save data in nampy arrays #
                    np.save(SavePath + 'unit_vector_' + str(group_number), prc_unit_vector)
                    np.save(SavePath + 'group_number_' + str(group_number), group_number)
                    np.save(SavePath + 'subgroup_number_' + str(group_number), subgroup_number)
                    np.save(SavePath + 'stellar_data_tmp_' + str(group_number), stellar_data_tmp)
                    np.save(SavePath + 'glx_unit_vector_' + str(group_number), glx_unit_vector)
                    print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.r:  # Read data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, prc_unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    group_number = np.load(SavePath + 'group_number_' + str(group_number) + '.npy')
                    subgroup_number = np.load(SavePath + 'subgroup_number_' + str(group_number) + '.npy')
                    prc_unit_vector = np.load(SavePath + 'unit_vector_' + str(group_number) + '.npy')
                    glx_unit_vector = np.load(SavePath + 'glx_unit_vector_' + str(group_number) + '.npy')
                    stellar_data_tmp = np.load(SavePath + 'stellar_data_tmp_' + str(group_number) + '.npy', allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, glx_unit_vector, prc_unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished RADecSurfaceDensity for ' + re.split('G-EAGLE/|/data', sim)[2] + ' in %.4s s' % (
            time.time() - start_global_time))  # Print total time.
        print('–––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_galaxies(sim, tag):
        """
         A static method to extract particle and subhalo attributes.
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
        mask = np.where(self.subhalo_data['ApertureMeasurements/Mass/030kpc'][:, 4] > 1e8)
        
        # Mask the temporary dictionary for each galaxy #
        subhalo_data_tmp = {}
        for attribute in self.subhalo_data.keys():
            subhalo_data_tmp[attribute] = np.copy(self.subhalo_data[attribute])[mask]
        
        return subhalo_data_tmp
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        A method to mask galaxies.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, prc_unit_vector
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        index = np.where(self.subhalo_data_tmp['GroupNumber'] == group_number)[0][subgroup_number]
        
        # Mask the data to select galaxies with a given GroupNumber and SubGroupNumber and particles inside a 30kpc sphere #
        mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(np.subtract(self.stellar_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index]), axis=1) <= 30.0))  # kpc
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp = {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[mask]
        
        # Normalise the coordinates and velocities wrt the centre of mass of the subhalo #
        stellar_data_tmp['Coordinates'] = np.subtract(stellar_data_tmp['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][index])
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass']))  # km s-1
        stellar_data_tmp['Velocity'] = np.subtract(stellar_data_tmp['Velocity'], CoM_velocity)
        
        # Compute the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # Msun kpc km s-1
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # Msun kpc km s-1
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        return stellar_data_tmp, glx_unit_vector, prc_unit_vector
    
    
    @staticmethod
    def plot(stellar_data_tmp, glx_unit_vector, prc_unit_vector, group_number, subgroup_number):
        """
        A method to plot a hexbin histogram.
        :param stellar_data_tmp: from mask_galaxies
        :param glx_unit_vector: from mask_galaxies
        :param prc_unit_vector: from mask_galaxies
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: None
        """
        lon = []
        lat = []
        
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)
        
        # Generate the figure #
        plt.close()
        figure = plt.figure(0, figsize=(20, 15))
        
        gs = gridspec.GridSpec(2, 2)
        axupperleft = plt.subplot(gs[0, 0], projection="mollweide")
        axupperright = plt.subplot(gs[0, 1])
        axlowerleft = plt.subplot(gs[1, 0])
        axlowerright = plt.subplot(gs[1, 1])
        
        axupperleft.grid(True, color='black')
        axlowerleft.grid(True, color='black')
        axupperright.grid(True, color='black')
        axupperleft.set_xlabel('RA ($\degree$)')
        axupperleft.set_ylabel('Dec ($\degree$)')
        axlowerleft.set_ylabel('Particles per hexbin')
        axupperright.set_ylabel('Particles per hexbin')
        axlowerleft.set_xlabel('Angular distance from X ($\degree$)')
        axupperright.set_xlabel('Angular distance from densest hexbin ($\degree$)')
        
        axupperright.set_xlim(-10, 190)
        axlowerleft.set_xlim(-10, 190)
        
        y_tick_labels = np.array(['', '-60', '', '-30', '', '0', '', '30', '', 60])
        x_tick_labels = np.array(['', '-120', '', '-60', '', '0', '', '60', '', 120])
        axupperleft.set_xticklabels(x_tick_labels)
        axupperleft.set_yticklabels(y_tick_labels)
        
        axupperright.set_xticks(np.arange(0, 181, 20))
        axlowerleft.set_xticks(np.arange(0, 181, 20))
        
        # Generate the RA and Dec projection #
        hexbin = axupperleft.hexbin(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]), np.arcsin(prc_unit_vector[:, 2]), bins='log',
                                    cmap='PuRd', gridsize=30, edgecolor='none', mincnt=1, zorder=-1)  # Element-wise arctan of x1/x2.
        axupperleft.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                            zorder=-1)
        
        # Get the values of each hexbin, convert their coordinates and plot them #
        counts = hexbin.get_array()
        verts = hexbin.get_offsets()
        for offc in range(verts.shape[0]):
            binx, biny = verts[offc][0], verts[offc][1]
            if counts[offc]:
                # Inverse transformation from x/y to lat/long #
                theta = np.arcsin(np.divide(biny, np.sqrt(2)))  # In radians.
                latitude = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)  # In radians.
                longitude = np.divide(np.pi * binx, (2 * np.sqrt(2) * np.cos(theta)))  # In radians.
                lat.append(latitude)
                lon.append(longitude)  # axupperleft.plot(longitude, latitude, 'k.')
        
        # Generate the color bar #
        cbar = plt.colorbar(hexbin, ax=axupperleft, orientation='horizontal')
        cbar.set_label('$\mathrm{Particles\; per\; hexbin}$')
        
        # Generate the RA and Dec  #
        # Get the positions of all hexbins and of the most dense one #
        position_other = np.vstack([lon, lat]).T  # In radians.
        index = np.where(counts == max(counts))[0]  # In radians.
        position_densest = np.vstack([lon[index[0]], lat[index[0]]]).T  # In radians.
        
        angular_theta_from_densest = np.arccos(
            np.cos(position_densest[0, 1]) * np.cos(position_other[:, 1]) * np.cos(position_densest[0, 0] - position_other[:, 0]) + np.sin(
                position_densest[0, 1]) * np.sin(position_other[:, 1]))  # In radians.
        
        axupperright.scatter(angular_theta_from_densest * np.divide(180.0, np.pi), counts, c='blue', s=10)
        
        ####################################################################################################
        index = np.where(angular_theta_from_densest < np.divide(np.pi, 6.0))
        axupperleft.scatter(position_other[index, 0], position_other[index, 1], s=40, c='green')
        axupperleft.scatter(position_densest[0, 0], position_densest[0, 1], s=100, c='black', marker='x')
        
        axupperright.axvline(x=30, c='green', lw=5)
        phi = np.linspace(0, 2.0 * np.pi, 50)
        r = np.radians(30)
        x = position_densest[0, 0] + r * np.cos(phi)
        y = position_densest[0, 1] + r * np.sin(phi)
        axupperleft.plot(x, y, color="red")
        ####################################################################################################
        
        # Generate the RA and Dec  #
        position_X = np.vstack([np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2])]).T
        angular_theta_from_X = np.arccos(
            np.cos(position_X[:, 0]) * np.cos(position_other[:, 0]) * np.cos(position_X[:, 1] - position_other[:, 1]) + np.sin(
                position_X[:, 0]) * np.sin(position_other[:, 0]))  # In radians.
        axlowerleft.scatter(angular_theta_from_X * np.divide(180.0, np.pi), counts, c='red', s=10)
        
        # Save the plot #
        # plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + str(group_number) + str(subgroup_number) + '-' + 'RDSD' + '-' + date + '.png', bbox_inches='tight')
        
        rows = ('discfrac', 'Wind', 'Flood', 'Quake', 'Hail')
        # axlowerright.table()
        # Calcuate kinematic diagnostics #
        # kappa, discfrac, orbi, vrotsig, vrots, delta, zaxis, Momentum = MorphoKinematics.kinematics_diagnostics(stellar_data_tmp['Coordinates'],
        #                                                                                                         stellar_data_tmp['Mass'],
        #                                                                                                         stellar_data_tmp['Velocity'],
        #                                                                                                         stellar_data_tmp[
        #                                                                                                             'ParticleBindingEnergy'])
        # print(discfrac)
        return None


if __name__ == '__main__':
    tag = '010_z005p000'
    sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RDSD/G-EAGLE/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/RDSD/G-EAGLE/'  # Path to save/load data.
    # tag = '027_z000p101'
    # sim = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
    # outdir = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/plots/RDSD/EAGLE/'  # Path to save plots.
    # SavePath = '/cosma7/data/dp004/dc-irod1/G-EAGLE/python/data/RDSD/EAGLE/'  # Path to save/load data.
    x = RADecSurfaceDensity(sim, tag)