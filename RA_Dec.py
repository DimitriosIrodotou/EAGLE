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

# Create a parser and add argument to read data #
parser = argparse.ArgumentParser(description='Create RA and Dec.')
parser.add_argument('-r', action='store_true', help='Read data')
parser.add_argument('-l', action='store_true', help='Load data')
parser.add_argument('-rs', action='store_true', help='Read data and save to numpy arrays')
args = parser.parse_args()

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)  # Ignore some plt warnings.


class RADec:
    """
    Create a RA and Dec plot with the angular momentum of particles for each galaxy.
    """
    
    
    def __init__(self, sim, tag):
        """
        A constructor method for the class.
        :param sim: simulation directory
        :param tag: redshift folder
        """
        
        p = 1  # Counter.
        # Initialise arrays and a dictionary to store the data #
        unit_vector = []
        stellar_data_tmp = {}
        glx_unit_vector = []
        
        if not args.l:
            self.Ngroups = E.read_header('SUBFIND', sim, tag, 'TotNgroups')
            self.stellar_data, self.subhalo_data = self.read_galaxies(sim, tag)
            print('Read data for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))
            print('–––––––––––––––––––––––––––––––––––––––––')
            
            self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes to select only those with stellar mass > 10^8Msun.
        
        # for group_number in list(set(self.subhalo_data_tmp['GroupNumber'])):  # Loop over all masked haloes.
        for group_number in range(8, 9):  # Loop over all masked haloes.
            for subgroup_number in range(0, 1):
                if args.rs:  # Read and save data.
                    start_local_time = time.time()  # Start the local time.
                    
                    stellar_data_tmp, glx_unit_vector, unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    
                    # Save data in nampy arrays #
                    np.save(SavePath + 'unit_vector_' + str(group_number), unit_vector)
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
                    
                    stellar_data_tmp, glx_unit_vector, unit_vector = self.mask_galaxies(group_number, subgroup_number)  # Mask the data.
                    print('Masked data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                        round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                    p += 1
                
                elif args.l:  # Load data.
                    start_local_time = time.time()  # Start the local time.
                    
                    group_number = np.load(SavePath + 'group_number_' + str(group_number) + '.npy')
                    subgroup_number = np.load(SavePath + 'subgroup_number_' + str(group_number) + '.npy')
                    unit_vector = np.load(SavePath + 'unit_vector_' + str(group_number) + '.npy')
                    glx_unit_vector = np.load(SavePath + 'glx_unit_vector_' + str(group_number) + '.npy')
                    stellar_data_tmp = np.load(SavePath + 'stellar_data_tmp_' + str(group_number) + '.npy', allow_pickle=True)
                    stellar_data_tmp = stellar_data_tmp.item()
                    print('Loaded data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                    # + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
                    print('–––––––––––––––––––––––––––––––––––––––––––––')
                
                # Plot the data #
                start_local_time = time.time()  # Start the local time.
                
                self.plot(stellar_data_tmp, glx_unit_vector, unit_vector, group_number, subgroup_number)
                print('Plotted data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time))
                print('–––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished RADec for ' + re.split('EAGLE/|/data', sim)[2] + ' in %.4s s' % (time.time() - start_global_time))  # Print total time.
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
        for attribute in ['BirthDensity', 'Coordinates', 'GroupNumber', 'Mass', 'ParticleBindingEnergy', 'StellarFormationTime', 'SubGroupNumber',
                          'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, sim, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        stellar_data['BirthDensity'] /= 6.769911178294543e-31  # Convert back to physical units.
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
    
    
    def mask_galaxies(self, group_number, subgroup_number):
        """
        A method to mask galaxies.
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, unit_vector
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
        unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        return stellar_data_tmp, glx_unit_vector, unit_vector
    
    
    @staticmethod
    def plot(stellar_data_tmp, glx_unit_vector, unit_vector, group_number, subgroup_number):
        """
        A method to plot a hexbin histogram.
        :param stellar_data_tmp: from mask_galaxies
        :param glx_unit_vector: from mask_galaxies
        :param unit_vector: from mask_galaxies
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: None
        """
        
        # Set the style of the plots #
        sns.set()
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.6)
        
        # Generate the figure #
        plt.close()
        figure = plt.figure(0, figsize=(20, 15))
        
        gs = gridspec.GridSpec(2, 2)
        axupperleft = plt.subplot(gs[0, 0], projection="mollweide")
        axlowerleft = plt.subplot(gs[1, 0], projection="mollweide")
        axupperright = plt.subplot(gs[0, 1], projection="mollweide")
        axlowerright = plt.subplot(gs[1, 1], projection="mollweide")
        
        axupperleft.grid(True)
        axlowerleft.grid(True)
        axupperright.grid(True)
        axlowerright.grid(True)
        axupperleft.set_xlabel('RA ($\degree$)')
        axlowerleft.set_xlabel('RA ($\degree$)')
        axupperright.set_xlabel('RA ($\degree$)')
        axlowerright.set_xlabel('RA ($\degree$)')
        axupperleft.set_ylabel('Dec ($\degree$)')
        axlowerleft.set_ylabel('Dec ($\degree$)')
        axupperright.set_ylabel('Dec ($\degree$)')
        axlowerright.set_ylabel('Dec ($\degree$)')
        
        y_tick_labels = np.array(['', '-60', '', '-30', '', 0, '', '30', '', 60])
        x_tick_labels = np.array(['', '-120', '', '-60', '', 0, '', '60', '', 120])
        axupperleft.set_xticklabels(x_tick_labels)
        axupperleft.set_yticklabels(y_tick_labels)
        axlowerleft.set_xticklabels(x_tick_labels)
        axlowerleft.set_yticklabels(y_tick_labels)
        axlowerright.set_xticklabels(x_tick_labels)
        axlowerright.set_yticklabels(y_tick_labels)
        axupperright.set_xticklabels(x_tick_labels)
        axupperright.set_yticklabels(y_tick_labels)
        
        # Generate the RA and Dec projection #
        # from scipy.ndimage import zoom
        # Nparticles = len(stellar_data_tmp['Mass'])
        # Nbins = [int((np.pi - (-np.pi)) / 0.1), int((np.pi / 2 - (-np.pi / 2)) / 0.05)]
        # H, xedges, yedges = np.histogram2d(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]), bins=Nbins)
        # H = zoom(H, 20)
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        # mylevels = np.linspace(1.0, Nbins[1], Nbins[1]) * Nparticles / (Nbins[1] ** 2 / 0.7)
        # hexbin = axupperleft.contourf(H.transpose()[::], origin='lower', cmap='inferno', levels=mylevels, extent=extent)
        hexbin = axupperleft.hexbin(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]), bins='log', cmap='PuRd',
                                    gridsize=100, edgecolor='none', mincnt=1, zorder=-1)  # Element-wise arctan of x1/x2.
        axupperleft.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                            zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(hexbin, ax=axupperleft, orientation='horizontal')
        cbar.set_label('$\mathrm{Particles\; per\; hexbin}$')
        
        # Generate the RA and Dec projection colour-coded by StellarFormationTime #
        scatter = axupperright.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]),
                                       c=stellar_data_tmp['StellarFormationTime'], cmap='jet_r', s=1, zorder=-1)
        
        axupperright.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                             zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axupperright, orientation='horizontal')
        cbar.set_label('$\mathrm{StellarFormationTime}$')
        
        # Generate the RA and Dec projection colour-coded by e^beta-1 #
        velocity_r_sqred = np.divide(np.sum(np.multiply(stellar_data_tmp['Velocity'], stellar_data_tmp['Coordinates']), axis=1) ** 2,
                                     np.sum(np.multiply(stellar_data_tmp['Coordinates'], stellar_data_tmp['Coordinates']), axis=1))
        beta = np.subtract(1, np.divide(
            np.subtract(np.sum(np.multiply(stellar_data_tmp['Velocity'], stellar_data_tmp['Velocity']), axis=1), velocity_r_sqred),
            2 * velocity_r_sqred))
        
        scatter = axlowerleft.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]), c=np.exp(beta - 1),
                                      cmap='tab20', s=1, zorder=-1)
        
        axlowerleft.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                            zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axlowerleft, orientation='horizontal')
        cbar.set_label(r'$\mathrm{log_{10}({\beta/\bar{\beta}})}$')
        
        # Generate the RA and Dec projection colour-coded by BirthDensity #
        scatter = axlowerright.scatter(np.arctan2(unit_vector[:, 1], unit_vector[:, 0]), np.arcsin(unit_vector[:, 2]),
                                       c=stellar_data_tmp['BirthDensity'], cmap='jet', s=1, zorder=-1)
        axlowerright.scatter(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]), np.arcsin(glx_unit_vector[2]), s=300, color='black', marker='X',
                             zorder=-1)
        
        # Generate the color bar #
        cbar = plt.colorbar(scatter, ax=axlowerright, orientation='horizontal')
        cbar.set_label('$\mathrm{BirthDensity}$')
        
        # Save the plot #
        # plt.title('z ~ ' + re.split('_z0|p000', tag)[1])
        plt.savefig(outdir + str(group_number) + str(subgroup_number) + '-' + 'RD' + '-' + date + '.png', bbox_inches='tight')
        
        return None


if __name__ == '__main__':
    # tag = '010_z005p000'
    # sim = '/cosma7/data/dp004/dc-payy1/G-EAGLE/GEAGLE_06/data/'
    # outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RD/G-EAGLE/'  # Path to save plots.
    # SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/RD/G-EAGLE/'  # Path to save/load data.
    tag = '027_z000p101'
    sim = '/cosma5/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'
    outdir = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/RD/EAGLE/'  # Path to save plots.
    SavePath = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/RD/EAGLE/'  # Path to save/load data.
    x = RADec(sim, tag)