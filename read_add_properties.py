import re
import sys
import time

import numpy as np
import astropy.units as u
import eagle_IO.eagle_IO.eagle_IO as E

from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematics

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.


class ReadAddProperties:
    """
    For each galaxy: load its stellar_data_tmp dictionary and add the new property(ies).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift directory
        """
        
        p = 1  # Counter.
        # Initialise a dictionary and arrays to store the data #
        stellar_data_tmp, glx_angular_momenta, glx_masses, disc_fractions, rotationals_over_dispersions, kappas = {}, [], [], [], [], []
        
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.stellar_data, self.subhalo_data = self.read_galaxies(simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        
        job_number = int(sys.argv[1])
        print(job_number)
        # group_numbers = np.array_split(list(self.subhalo_data_tmp['GroupNumber']), 30)
        # subgroup_numbers = np.array_split(list(self.subhalo_data_tmp['SubGroupNumber']), 30)
        group_numbers = np.array(1, 26)
        subgroup_numbers = np.array(0, 1)
        for group_number, subgroup_number in zip(group_numbers[job_number],
                                                 subgroup_numbers[job_number]):  # Loop over all masked haloes and sub-haloes.
            start_local_time = time.time()  # Start the local time.
            
            # Mask galaxies and normalise data #
            stellar_data_tmp, glx_mass, glx_angular_momentum, disc_fraction, rotational_over_dispersion, kappa, c = self.mask_galaxies(group_number,
                                                                                                                                       subgroup_number)
            stellar_data_tmp['c'] = c
            stellar_data_tmp['kappa'] = kappa
            stellar_data_tmp['glx_mass'] = glx_mass
            stellar_data_tmp['disc_fraction'] = disc_fraction
            stellar_data_tmp['glx_angular_momentum'] = glx_angular_momentum
            stellar_data_tmp['rotational_over_dispersion'] = rotational_over_dispersion
            
            # Save data in numpy array #
            np.save(data_path + 'stellar_data_tmps/' + 'stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
            
            print('Masked and saved data for halo ' + str(group_number) + ' in %.4s s' % (time.time() - start_local_time) + ' (' + str(
                round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            p += 1
        
        print('Finished ReadAddProperties for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
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
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'InitialMassWeightedStellarAge', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load particle data in h-free physical CGS units #
        stellar_data = {}
        particle_type = '4'
        file_type = 'PARTDATA'
        for attribute in ['BirthDensity', 'Coordinates', 'GroupNumber', 'Mass', 'Metallicity', 'ParticleBindingEnergy', 'StellarFormationTime',
                          'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        stellar_data['BirthDensity'] *= np.divide(u.g.to(u.Msun), u.cm.to(u.kpc) ** 3)
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
        :param group_number: from list(set(self.subhalo_data_tmp['GroupNumber']))
        :param subgroup_number: from list(set(self.subhalo_data_tmp['SubGroupNumber']))
        :return: stellar_data_tmp, glx_unit_vector
        """
        
        # Select the corresponding halo in order to get its centre of potential #
        halo_mask, = np.where((self.subhalo_data_tmp['GroupNumber'] == group_number) & (self.subhalo_data_tmp['SubGroupNumber'] == subgroup_number))
        
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
        
        # Calculate the disc fraction and the rotational over dispersion velocity ratio #
        kappa_old, disc_fraction, orbital, rotational_over_dispersion, vrots, zaxis, momentum = MorphoKinematics.kinematics_diagnostics(
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'], stellar_data_tmp['ParticleBindingEnergy'])
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        glx_mass = np.sum(stellar_data_tmp['Mass'])
        prc_spc_angular_momentum = np.cross(stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'])  # kpc km s-1
        glx_angular_momentum = np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * prc_spc_angular_momentum, axis=0)  # Msun kpc km s-1
        
        # Calculate kappa #
        coordinates, velocity, prc_angular_momentum, glx_angular_momentum_old = RotateCoordinates.rotate_Jz(stellar_data_tmp)
        
        prc_cylindrical_distance = np.linalg.norm(np.dstack((coordinates[:, 0], coordinates[:, 1]))[0], axis=1)
        kappa = np.sum(0.5 * stellar_data_tmp['Mass'] * ((prc_spc_angular_momentum[:, 2] / prc_cylindrical_distance) ** 2)) / np.sum(
            0.5 * stellar_data_tmp['Mass'] * (np.linalg.norm(velocity, axis=1) ** 2))
        
        # Calculate the concentration index #
        c = np.divide(MorphoKinematics.r_ninety(stellar_data_tmp), MorphoKinematics.r_fifty(stellar_data_tmp))
        
        return stellar_data_tmp, glx_mass, glx_angular_momentum, disc_fraction, rotational_over_dispersion, kappa, c


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    x = ReadAddProperties(simulation_path, tag)
