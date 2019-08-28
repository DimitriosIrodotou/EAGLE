# Import required python libraries #
import h5py
import numpy as np

from files import *


class Convert:
    """
    A class that contains all the methods (static) to convert properties to physical units.
    """


    @staticmethod
    def convert_to_physical_units(properties, component):
        # Ask the user if they want to convert the property to physical units #
        convert = input('Do you want to convert the property to physical units? y/n: ')
        rules = [convert is not 'y', convert is not 'n']  # Warn the user that the answer was wrong.
        while all(rules) is True:
            convert = input('Wrong input! Do you want to convert the property to physical units? y/n: ')
            rules = [convert is not 'y', convert is not 'n']
        if component is 'g':
            x = Convert.convert_groups_to_physical_units(properties[0])
            y = Convert.convert_groups_to_physical_units(properties[1])
        elif component is 's':
            x = Convert.convert_stars_to_physical_units(properties[0])
            y = Convert.convert_stars_to_physical_units(properties[1])

        return x, y


    @staticmethod
    def convert_groups_to_physical_units(property):
        """
        A function to convert galactic properties to physical units.

        :param property: galactic property (input from user from groups_io.Ask.read).
        :return: x, y
        """

        # Get the conversion factors #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            a = f['Header'].attrs.get('Time')
            h = f['Header'].attrs.get('HubbleParam')
            cgs = f['Subhalo'][property].attrs.get('CGSConversionFactor')
            aexp = f['Subhalo'][property].attrs.get('aexp-scale-exponent')
            hexp = f['Subhalo'][property].attrs.get('h-scale-exponent')

        # Convert to physical units #
        prop = np.load('./data/groups/subhalo/' + property + '.npy')
        prop = np.multiply(prop, cgs * a ** aexp * h ** hexp, dtype='f8')

        return prop


    @staticmethod
    def convert_stars_to_physical_units(property):
        """
        A function to convert stellar properties to physical units.

        :param property: stellar property (input from user from groups_io.Ask.read).
        :return: x, y
        """

        # Get the conversion factors #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            a = f['Header'].attrs.get('Time')
            h = f['Header'].attrs.get('HubbleParam')
            cgs = f['Subhalo']['Stars'][property].attrs.get('CGSConversionFactor')
            aexp = f['Subhalo']['Stars'][property].attrs.get('aexp-scale-exponent')
            hexp = f['Subhalo']['Stars'][property].attrs.get('h-scale-exponent')

        # Convert to physical units #
        prop = np.load('./data/groups/stars/' + property + '.npy')
        prop = np.multiply(prop, cgs * a ** aexp * h ** hexp, dtype='f8')

        return prop