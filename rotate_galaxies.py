import numpy as np
import astropy.units as u

from astropy_healpix import HEALPix


class RotateCoordinates:
    """
    Rotate coordinates and velocities wrt different quantities.
    """
    
    
    @staticmethod
    def rotate_X(stellar_data_tmp, glx_unit_vector):
        """
        Rotate first about z-axis to set y=0 and then about the y-axis to set z=0
        :param stellar_data_tmp: from mask_galaxies
        :param glx_unit_vector: from mask_galaxies
        :return: stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector
        """
        # Calculate the rotation matrices and combine them #
        ra = np.arctan2(glx_unit_vector[1], glx_unit_vector[0])
        dec = np.arcsin(glx_unit_vector[2])
        
        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(dec), 0, np.sin(dec)], [0, 1, 0], [-np.sin(dec), 0, np.cos(dec)]])
        Ryz = np.matmul(Ry, Rz)
        
        # Rotate the coordinates and velocities of stellar particles #
        coordinates = np.matmul(Ryz, stellar_data_tmp['Coordinates'][..., None]).squeeze()
        velocities = np.matmul(Ryz, stellar_data_tmp['Velocity'][..., None]).squeeze()
        
        # Calculate the angular momentum for each particle and for the galaxy and the unit vector parallel to the galactic angular momentum vector #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(coordinates, velocities)  # In Msun kpc km s-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s-1.
        glx_unit_vector = np.divide(glx_angular_momentum, np.linalg.norm(glx_angular_momentum))
        prc_unit_vector = np.divide(prc_angular_momentum, np.linalg.norm(prc_angular_momentum, axis=1)[:, np.newaxis])
        
        return coordinates, velocities, prc_unit_vector, glx_unit_vector
    
    
    @staticmethod
    def rotate_densest(prc_unit_vector, glx_unit_vector):
        """
        Rotate first about z-axis to set y=0and then about the y-axis to set z=0
        :param prc_unit_vector:
        :param glx_unit_vector:
        :return: prc_unit_vector, glx_unit_vector
        """
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        ra = np.degrees(np.arctan2(prc_unit_vector[:, 1], prc_unit_vector[:, 0]))
        dec = np.degrees(np.arcsin(prc_unit_vector[:, 2]))
        
        # Create HEALPix map #
        nside = 2 ** 5  # Define the resolution of the grid (number of divisions along the side of a base-resolution pixel).
        hp = HEALPix(nside=nside)  # Initialise the HEALPix pixellisation class.
        indices = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)  # Create list of HEALPix indices from particles' ra and dec.
        density = np.bincount(indices, minlength=hp.npix)  # Count number of data points in each HEALPix pixel.
        
        # Find location of density maximum and plot its positions and the ra and dec of the galactic angular momentum #
        index_densest = np.argmax(density)
        lon_densest = (hp.healpix_to_lonlat([index_densest])[0].value + np.pi) % (2 * np.pi) - np.pi
        lat_densest = (hp.healpix_to_lonlat([index_densest])[1].value + np.pi / 2) % (2 * np.pi) - np.pi / 2
        
        # Calculate the rotation matrices and combine them #
        ra = np.float(lon_densest)
        dec = np.float(lat_densest)
        print(density[index_densest])
        # ra = np.arctan2(glx_unit_vector[1], glx_unit_vector[0])
        # dec = np.arcsin(glx_unit_vector[2])
        
        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(dec), 0, np.sin(dec)], [0, 1, 0], [-np.sin(dec), 0, np.cos(dec)]])
        Ryz = np.matmul(Ry, Rz)
        
        prc_unit_vector = np.matmul(Ryz, prc_unit_vector[..., None]).squeeze()
        glx_unit_vector = np.matmul(Ryz, glx_unit_vector)
        
        return prc_unit_vector, glx_unit_vector
    
    
    @staticmethod
    def rotate_Jz(stellar_data_tmp):
        """
        Rotate a galaxy such that its angular momentum is along the z axis/
        :param stellar_data_tmp:
        :return: stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_angular_momentum, glx_angular_momentum
        """
        
        # Calculate the angular momentum of the galaxy #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(stellar_data_tmp['Coordinates'],
                                                                                  stellar_data_tmp['Velocity'])  # In Msun kpc km s-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s-1.
        
        # Define the rotation matrices #
        a = np.matrix([glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]]) / np.linalg.norm(
            [glx_angular_momentum[0], glx_angular_momentum[1], glx_angular_momentum[2]])
        b = np.matrix([0, 0, 1])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b.T)
        vx = np.matrix([[0, -v[0, 2], v[0, 1]], [v[0, 2], 0, -v[0, 0]], [-v[0, 1], v[0, 0], 0]])
        transform = np.eye(3, 3) + vx + (vx * vx) * ((1 - c[0, 0]) / s ** 2)
        
        # Rotate the coordinates and velocities #
        coordinates = np.array([np.matmul(transform, stellar_data_tmp['Coordinates'][i].T) for i in range(0, len(stellar_data_tmp['Coordinates']))])[
                      :, 0]
        velocities = np.array([np.matmul(transform, stellar_data_tmp['Velocity'][i].T) for i in range(0, len(stellar_data_tmp['Velocity']))])[:, 0]
        
        # Calculate the rotated angular momentum of the galaxy #
        prc_angular_momentum = stellar_data_tmp['Mass'][:, np.newaxis] * np.cross(coordinates, velocities)  # In Msun kpc km s-1.
        glx_angular_momentum = np.sum(prc_angular_momentum, axis=0)  # In Msun kpc km s-1.
        
        return coordinates, velocities, prc_angular_momentum, glx_angular_momentum
