import re
import sys
import time
import h5py

import numpy as np
import astropy.units as u
import eagle_IO.eagle_IO.eagle_IO as E

from astropy_healpix import HEALPix
from rotate_galaxies import RotateCoordinates
from morpho_kinematics import MorphoKinematic

date = time.strftime('%d_%m_%y_%H%M')  # Date
start_global_time = time.time()  # Start the global time.


class ReadAttributes:
    """
    For each galaxy: read in and save its attribute(s).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        :param simulation_path: simulation directory
        :param tag: redshift directory
        """
        
        p = 1  # Counter.
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.stellar_data, self.gaseous_data, self.dark_matter_data, self.subhalo_data, self.FOF_data = self.read_attributes(simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        
        job_number = int(sys.argv[2]) - 1
        group_numbers = np.array_split(list(self.subhalo_data_tmp['GroupNumber']), 30)
        subgroup_numbers = np.array_split(list(self.subhalo_data_tmp['SubGroupNumber']), 30)
        
        for group_number, subgroup_number in zip(group_numbers[job_number],
                                                 subgroup_numbers[job_number]):  # Loop over all masked haloes and sub-haloes.
            start_local_time = time.time()  # Start the local time.
            
            # Mask galaxies and normalise data #
            stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp = self.mask_galaxies(group_number, subgroup_number)
            
            # Save data in numpy array #
            np.save(data_path + 'FOF_data_tmps/FOF_data_tmp_' + str(group_number), self.FOF_data)
            np.save(data_path + 'subhalo_data_tmps/subhalo_data_tmp_' + str(group_number) + '_' + str(subgroup_number), self.subhalo_data_tmp)
            np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
            np.save(data_path + 'gaseous_data_tmps/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number), gaseous_data_tmp)
            np.save(data_path + 'dark_matter_data_tmps/dark_matter_data_tmp_' + str(group_number) + '_' + str(subgroup_number), dark_matter_data_tmp)
            
            print('Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (
                time.time() - start_local_time) + ' (' + str(round(100 * p / len(set(self.subhalo_data_tmp['GroupNumber'])), 1)) + '%)')
            print('–––––––––––––––––––––––––––––––––––––––––––––')
            p += 1
        
        print('Finished ReadAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    def read_attributes(self, simulation_path, tag):
        """
        Extract particle and subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        :return: stellar_data, gaseous_data, dark_matter_data, subhalo_data, FOF_data
        """
        
        # Load particle data in h-free physical CGS units #
        stellar_data, gaseous_data, dark_matter_data = {}, {}, {}
        file_type = 'PARTDATA'
        particle_type = '4'
        for attribute in ['BirthDensity', 'Coordinates', 'GroupNumber', 'Mass', 'Metallicity', 'ParticleBindingEnergy', 'StellarFormationTime',
                          'SubGroupNumber', 'Velocity']:
            stellar_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        particle_type = '0'
        for attribute in ['Coordinates', 'GroupNumber', 'Mass', 'StarFormationRate', 'SubGroupNumber', 'Velocity']:
            gaseous_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        particle_type = '1'
        for attribute in ['Coordinates', 'GroupNumber', 'SubGroupNumber', 'Velocity']:
            dark_matter_data[attribute] = E.read_array(file_type, simulation_path, tag, '/PartType' + particle_type + '/' + attribute, numThreads=8)
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'CentreOfPotential', 'GroupNumber', 'InitialMassWeightedStellarAge', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Load FOF data in h-free physical CGS units #
        FOF_data = {}
        file_type = 'SUBFIND'
        for attribute in ['Group_M_Crit200', 'FirstSubhaloID']:
            FOF_data[attribute] = E.read_array(file_type, simulation_path, tag, '/FOF/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        stellar_data['Mass'] *= u.g.to(u.Msun)
        stellar_data['Velocity'] *= u.cm.to(u.km)  # per second.
        stellar_data['Coordinates'] *= u.cm.to(u.kpc)
        stellar_data['BirthDensity'] *= np.divide(u.g.to(u.Msun), u.cm.to(u.kpc) ** 3)
        
        gaseous_data['Mass'] *= u.g.to(u.Msun)
        gaseous_data['Velocity'] *= u.cm.to(u.km)  # per second.
        gaseous_data['Coordinates'] *= u.cm.to(u.kpc)
        gaseous_data['StarFormationRate'] *= np.divide(u.g.to(u.Msun), u.seconds.to(u.year))
        
        dark_matter_data['Velocity'] *= u.cm.to(u.km)  # per second.
        dark_matter_data['Mass'] = self.dark_matter_mass(simulation_path, tag) * u.g.to(u.Msun)
        
        subhalo_data['CentreOfPotential'] *= u.cm.to(u.kpc)
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        FOF_data['Group_M_Crit200'] *= u.g.to(u.Msun)
        
        return stellar_data, gaseous_data, dark_matter_data, subhalo_data, FOF_data
    
    
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
        stellar_mask = np.where((self.stellar_data['GroupNumber'] == group_number) & (self.stellar_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(np.subtract(self.stellar_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][halo_mask]), axis=1) <= 30.0))
        
        gaseous_mask = np.where((self.gaseous_data['GroupNumber'] == group_number) & (self.gaseous_data['SubGroupNumber'] == subgroup_number) & (
            np.linalg.norm(np.subtract(self.gaseous_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][halo_mask]), axis=1) <= 30.0))
        
        dark_matter_mask = np.where(
            (self.dark_matter_data['GroupNumber'] == group_number) & (self.dark_matter_data['SubGroupNumber'] == subgroup_number) & (
                np.linalg.norm(np.subtract(self.dark_matter_data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][halo_mask]),
                               axis=1) <= 30.0))
        
        # Mask the temporary dictionary for each galaxy #
        stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp = {}, {}, {}
        for attribute in self.stellar_data.keys():
            stellar_data_tmp[attribute] = np.copy(self.stellar_data[attribute])[stellar_mask]
        for attribute in self.gaseous_data.keys():
            gaseous_data_tmp[attribute] = np.copy(self.gaseous_data[attribute])[gaseous_mask]
        for attribute in self.dark_matter_data.keys():
            dark_matter_data_tmp[attribute] = np.copy(self.dark_matter_data[attribute])[dark_matter_mask]
        
        # Normalise the coordinates and velocities wrt the centre of potential of the subhalo #
        for data in [stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp]:
            data['Coordinates'] = np.subtract(data['Coordinates'], self.subhalo_data_tmp['CentreOfPotential'][halo_mask])
        
        CoM_velocity = np.divide(np.sum(stellar_data_tmp['Mass'][:, np.newaxis] * stellar_data_tmp['Velocity'], axis=0) + np.sum(
            gaseous_data_tmp['Mass'][:, np.newaxis] * gaseous_data_tmp['Velocity'], axis=0) + np.sum(
            dark_matter_data_tmp['Mass'][:, np.newaxis] * dark_matter_data_tmp['Velocity'], axis=0),
                                 np.sum(stellar_data_tmp['Mass'], axis=0) + np.sum(gaseous_data_tmp['Mass'], axis=0) + np.sum(
                                     dark_matter_data_tmp['Mass'], axis=0))  # In km s-1.
        for data in [stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp]:
            data['Velocity'] = np.subtract(data['Velocity'], CoM_velocity)
        
        return stellar_data_tmp, gaseous_data_tmp, dark_matter_data_tmp
    
    
    @staticmethod
    def dark_matter_mass(simulation_path, tag):
        """
        Create a mass array for dark matter particles. As all dark matter particles share the same mass, there exists no PartType1/Mass dataset in
        the snapshot files.
        :return: particle_mass
        """
        
        # Read the required properties from the header and get conversion factors from gas particles #
        f = h5py.File(simulation_path + 'snapshot_' + tag + '/snap_027_z000p101.0.hdf5', 'r')
        a = f['Header'].attrs.get('Time')
        h = f['Header'].attrs.get('HubbleParam')
        n_particles = f['Header'].attrs.get('NumPart_Total')[1]
        dark_matter_mass = f['Header'].attrs.get('MassTable')[1]
        
        cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
        aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
        hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')
        f.close()
        
        # Create an array of length equal to the number of dark matter particles and convert to h-free physical CGS units #
        particle_masses = np.ones(n_particles, dtype='f8') * dark_matter_mass
        particle_masses *= cgs * a ** aexp * h ** hexp
        
        return particle_masses


class AddAttributes:
    """
    For each galaxy: load its stellar_data_tmp dictionary and add the new attribute(s).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        """
        
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.subhalo_data = self.read_attributes(simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        
        job_number = int(sys.argv[2]) - 1
        group_numbers = np.array_split(list(self.subhalo_data_tmp['GroupNumber']), 30)
        subgroup_numbers = np.array_split(list(self.subhalo_data_tmp['SubGroupNumber']), 30)
        
        for group_number, subgroup_number in zip(group_numbers[job_number],
                                                 subgroup_numbers[job_number]):  # Loop over all masked haloes and sub-haloes.
            start_local_time = time.time()  # Start the local time.
            
            if group_number == 7413:
                print("Encountered 7413")
                continue
            
            # Load data #
            stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            stellar_data_tmp = stellar_data_tmp.item()
            
            stellar_data_tmp['c'] = self.concentration_index(stellar_data_tmp)
            stellar_data_tmp['kappa_corotation'] = self.kappa_corotation(stellar_data_tmp)
            stellar_data_tmp['disc_mask_IT20'], stellar_data_tmp['bulge_mask_IT20'] = self.decomposition_IT20(stellar_data_tmp)
            stellar_data_tmp['disc_fraction'], stellar_data_tmp['rotational_over_dispersion'] = self.kinematic_diagnostics(stellar_data_tmp)
            
            # Save data in numpy array #
            np.save(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number), stellar_data_tmp)
            
            print(
                'Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        print('Finished AddAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        :return: subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return subhalo_data
    
    
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
    
    
    @staticmethod
    def decomposition_IT20(stellar_data_tmp):
        """
        Find the particles that belong to the disc and bulge based on the IT20 method.
        :param stellar_data_tmp:
        :return: disc_mask, bulge_mask
        """
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s-1.
        glx_stellar_angular_momentum = np.sum(prc_angular_momentum, axis=0)
        glx_unit_vector = np.divide(glx_stellar_angular_momentum, np.linalg.norm(glx_stellar_angular_momentum))
        
        # Rotate coordinates and velocities of stellar particles wrt galactic angular momentum #
        coordinates, velocity, prc_unit_vector, glx_unit_vector = RotateCoordinates.rotate_X(stellar_data_tmp, glx_unit_vector)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Plot a HEALPix histogram #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.
        
        # Find location of density maximum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate and plot the disc (bulge) mass surface density as the mass within (outside) 30 degrees from the densest pixel #
        angular_theta_from_densest = np.arccos(
            np.sin(lat_densest) * np.sin(np.arcsin(prc_unit_vector[:, 2])) + np.cos(lat_densest) * np.cos(np.arcsin(prc_unit_vector[:, 2])) * np.cos(
                lon_densest - np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0])))  # In radians.
        disc_mask, = np.where(angular_theta_from_densest < np.divide(np.pi, 6.0))
        bulge_mask, = np.where(angular_theta_from_densest > np.divide(np.pi, 6.0))
        
        return disc_mask, bulge_mask
    
    
    @staticmethod
    def concentration_index(stellar_data_tmp):
        """
        Calculate the concentration index R90/R50.
        :param stellar_data_tmp:
        :return: c
        """
        
        c = np.divide(MorphoKinematic.r_ninety(stellar_data_tmp), MorphoKinematic.r_fifty(stellar_data_tmp))
        return c
    
    
    @staticmethod
    def kappa_corotation(stellar_data_tmp):
        """
        Calculate the fraction of a particle's kinetic energy that's invested in corotation.
        :param stellar_data_tmp:
        :return: kappa_corotation
        """
        
        # Rotate the galaxy and calculate the unit vector pointing along the glx_stellar_angular_momentum direction.
        coordinates, velocity, prc_angular_momentum, glx_stellar_angular_momentum = RotateCoordinates.rotate_Jz(stellar_data_tmp)
        prc_spc_angular_momentum = np.cross(coordinates, velocity)  # In kpc km s-1.
        glx_unit_vector = np.divide(glx_stellar_angular_momentum, np.linalg.norm(glx_stellar_angular_momentum))
        spc_angular_momentum_z = np.sum(glx_unit_vector * prc_spc_angular_momentum, axis=1)  # In kpc km s-1.
        prc_cylindrical_distance = np.sqrt(coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2)  # In kpc.
        
        corotation_mask, = np.where(spc_angular_momentum_z > 0)
        specific_angular_velocity = np.divide(spc_angular_momentum_z[corotation_mask], prc_cylindrical_distance[corotation_mask])  # In km s^-1.
        kinetic_energy = np.sum(stellar_data_tmp['Mass'] * np.linalg.norm(velocity, axis=1) ** 2)  # In Msun km^2 s^-2.
        angular_kinetic_energy = np.sum(stellar_data_tmp['Mass'][corotation_mask] * specific_angular_velocity ** 2)  # In Msun km^2 s^-2.
        kappa_corotation = np.divide(angular_kinetic_energy, kinetic_energy)
        
        return kappa_corotation
    
    
    @staticmethod
    def kinematic_diagnostics(stellar_data_tmp):
        """
        Calculate the disc fraction and the rotational over dispersion velocity ratio.
        :param stellar_data_tmp:
        :return: disc_fraction, rotational_over_dispersion
        """
        
        kappa_old, disc_fraction, orbital, rotational_over_dispersion, vrots, zaxis, momentum = MorphoKinematic.kinematic_diagnostics(
            stellar_data_tmp['Coordinates'], stellar_data_tmp['Mass'], stellar_data_tmp['Velocity'], stellar_data_tmp['ParticleBindingEnergy'])
        
        return disc_fraction, rotational_over_dispersion


class AppendAttributes:
    """
    For each galaxy: load its stellar_data_tmp dictionary and append the attribute(s).
    """
    
    
    def __init__(self, simulation_path, tag):
        """
        A constructor method for the class.
        """
        
        # Initialise arrays to store the data #
        glx_stellar_angular_momenta, glx_gaseous_angular_momenta, glx_stellar_masses, glx_gaseous_masses, glx_concentration_indices, \
        glx_kappas_corotation, glx_disc_fractions_IT20, glx_disc_fractions, disc_metallicities, bulge_metallicities, \
        glx_rotational_over_dispersion, group_numbers, subgroup_numbers, glx_star_formation_rate = [], [], [], [], [], [], [], [], [], [], [], [], \
                                                                                                   [], []
        
        # Extract particle and subhalo attributes and convert them to astronomical units #
        self.subhalo_data = self.read_attributes(simulation_path, tag)
        print('Read data for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        self.subhalo_data_tmp = self.mask_haloes()  # Mask haloes: select haloes with masses within 30 kpc aperture higher than 1e9 Msun.
        for group_number, subgroup_number in zip(list(self.subhalo_data_tmp['GroupNumber']),
                                                 list(self.subhalo_data_tmp['SubGroupNumber'])):  # Loop over all masked haloes and sub-haloes.
            if group_number == 7413:
                print("Encountered 7413")
                continue
            start_local_time = time.time()  # Start the local time.
            
            # Load data #
            # stellar_data_tmp = np.load(data_path + 'stellar_data_tmps/stellar_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
            #                            allow_pickle=True)
            # stellar_data_tmp = stellar_data_tmp.item()
            
            gaseous_data_tmp = np.load(data_path + 'gaseous_data_tmp/gaseous_data_tmp_' + str(group_number) + '_' + str(subgroup_number) + '.npy',
                                       allow_pickle=True)
            gaseous_data_tmp = gaseous_data_tmp.item()
            gaseous_data_tmp['StarFormationRate'] *= np.divide(u.g.to(u.Msun), u.seconds.to(u.year))
            
            # Calculate attributes #
            # glx_stellar_mass = np.sum(stellar_data_tmp['Mass'])
            # prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
            #                                                                           stellar_data_tmp['Velocity'])  # In Msun kpc km s-1.
            # glx_stellar_angular_momentum = np.sum(prc_angular_momentum, axis=0)
            #
            # for i, mask in enumerate([stellar_data_tmp['disc_mask_IT20'], stellar_data_tmp['bulge_mask_IT20']]):
            #     component_mass = np.sum(stellar_data_tmp['Mass'][mask])
            #     metals = np.divide(stellar_data_tmp['Metallicity'][mask] * stellar_data_tmp['Mass'][mask], component_mass)
            #     if i == 0:
            #         stellar_data_tmp['disc_metallicity'] = np.divide(np.sum(metals), 0.0134)  # In solar metallicity.
            #     else:
            #         stellar_data_tmp['bulge_metallicity'] = np.divide(np.sum(metals), 0.0134)  # In solar metallicity.
            
            glx_gaseous_mass = np.sum(gaseous_data_tmp['Mass'])
            prc_angular_momentum = gaseous_data_tmp['Mass'][:, np.newaxis] * np.cross(gaseous_data_tmp['Coordinates'],
                                                                                      gaseous_data_tmp['Velocity'])  # In Msun kpc km s-1.
            glx_gaseous_angular_momentum = np.sum(prc_angular_momentum, axis=0)
            
            # Append attributes into single arrays #
            # group_numbers.append(group_number)
            # subgroup_numbers.append(subgroup_number)
            # glx_stellar_masses.append(glx_stellar_mass)
            # glx_stellar_angular_momenta.append(glx_stellar_angular_momentum)
            # glx_concentration_indices.append(stellar_data_tmp['c'])
            # glx_disc_fractions.append(stellar_data_tmp['disc_fraction'])
            # disc_metallicities.append(stellar_data_tmp['disc_metallicity'])
            # bulge_metallicities.append(stellar_data_tmp['bulge_metallicity'])
            # glx_kappas_corotation.append(stellar_data_tmp['kappa_corotation'])
            # glx_rotational_over_dispersion.append(stellar_data_tmp['rotational_over_dispersion'])
            # disc_fraction_IT20 = np.divide(np.sum(stellar_data_tmp['Mass'][stellar_data_tmp['disc_mask_IT20']]), np.sum(stellar_data_tmp['Mass']))
            # glx_disc_fractions_IT20.append(disc_fraction_IT20)
            
            glx_gaseous_masses.append(glx_gaseous_mass)
            glx_gaseous_angular_momenta.append(glx_gaseous_angular_momentum)
            glx_star_formation_rate.append(gaseous_data_tmp['StarFormationRate'])
            
            print(
                'Masked and saved data for halo ' + str(group_number) + '_' + str(subgroup_number) + ' in %.4s s' % (time.time() - start_local_time))
            print('–––––––––––––––––––––––––––––––––––––––––––––')
        
        # Save data in numpy array #
        # np.save(data_path + 'group_numbers', group_numbers)
        # np.save(data_path + 'subgroup_numbers', subgroup_numbers)
        # np.save(data_path + 'glx_stellar_masses', glx_stellar_masses)
        # np.save(data_path + 'glx_disc_fractions', glx_disc_fractions)
        # np.save(data_path + 'disc_metallicities', disc_metallicities)
        # np.save(data_path + 'glx_stellar_angular_momenta', glx_stellar_angular_momenta)
        # np.save(data_path + 'bulge_metallicities', bulge_metallicities)
        # np.save(data_path + 'glx_kappas_corotation', glx_kappas_corotation)
        # np.save(data_path + 'glx_disc_fractions_IT20', glx_disc_fractions_IT20)
        # np.save(data_path + 'glx_concentration_indices', glx_concentration_indices)
        # np.save(data_path + 'glx_rotational_over_dispersion', glx_rotational_over_dispersion)
        
        np.save(data_path + 'glx_gaseous_masses', glx_gaseous_masses)
        np.save(data_path + 'glx_star_formation_rate', glx_star_formation_rate)
        np.save(data_path + 'glx_gaseous_angular_momenta', glx_gaseous_angular_momenta)
        
        print('Finished AppendAttributes for ' + re.split('Planck1/|/PE', simulation_path)[1] + ' in %.4s s' % (time.time() - start_global_time))
        print('–––––––––––––––––––––––––––––––––––––––––––––')
    
    
    @staticmethod
    def read_attributes(simulation_path, tag):
        """
        Extract subhalo attributes and convert them to astronomical units.
        :param simulation_path: simulation directory
        :param tag: redshift folder
        :return: subhalo_data
        """
        
        # Load subhalo data in h-free physical CGS units #
        subhalo_data = {}
        file_type = 'SUBFIND'
        for attribute in ['ApertureMeasurements/Mass/030kpc', 'GroupNumber', 'SubGroupNumber']:
            subhalo_data[attribute] = E.read_array(file_type, simulation_path, tag, '/Subhalo/' + attribute, numThreads=8)
        
        # Convert attributes to astronomical units #
        subhalo_data['ApertureMeasurements/Mass/030kpc'] *= u.g.to(u.Msun)
        
        return subhalo_data
    
    
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


if __name__ == '__main__':
    tag = '027_z000p101'
    simulation_path = '/cosma7/data/Eagle/ScienceRuns/Planck1/L0100N1504/PE/REFERENCE/data/'  # Path to EAGLE data.
    data_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/data/'  # Path to save/load data.
    if str(sys.argv[1]) == '-r':
        print('Reading attributes')
        x = ReadAttributes(simulation_path, tag)
    elif str(sys.argv[1]) == '-ad':
        print('Adding attributes')
        x = AddAttributes(simulation_path, tag)
    elif str(sys.argv[1]) == '-ap':
        print('Appending attributes')
        x = AppendAttributes(simulation_path, tag)
