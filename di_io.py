# Import required python libraries #
import glob
import time
from pprint import pprint

import h5py
import numpy as np

from files import *

start_time_Askread = time.time()  # Start the time.


def read():
    """
    A function to ask the user which files/properties they want to read.

    :return: None
    """
    # Show the available properties to the user
    print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Found ' + str(len(glob.glob('./data/groups/subhalo/*.npy'))) + ' available subhalo properties:')
    for name in glob.glob('./data/groups/subhalo/*.npy'):
        print(name)
    print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
    print('Found ' + str(len(glob.glob('./data/groups/stars/*.npy'))) + ' available stellar properties:')
    for name in glob.glob('./data/groups/stars/*.npy'):
        print(name)
    print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

    # Ask if the user wants to read new data #
    read = input('Do you want to read new data? y/n: ')
    rules = [read is not 'y', read is not 'n']

    # Warn the user that the answer was wrong #
    while all(rules) is True:
        read = input('Wrong input! Do you want to read new data? y/n: ')
        rules = [read is not 'y', read is not 'n']

    if read is 'y':  # Read
        # Ask if the user wants to read subhalo or stellar properties #
        component = input('Do you want to read subhalo groups or stars? g/s: ')
        rules = [component is not 'g', component is not 's']

        # Warn the user that the answer was wrong #
        while all(rules) is True:
            component = input('Wrong input! Do you want to read subhalo groups or stars? g/s: ')
            rules = [component is not 'g', component is not 's']

        if component is 'g':  # Subhalo groups
            Show.show_subhalo_properties()  # Show them the properties.

            # Ask the user how many properties they want to read #
            prop_num = input("How many properties do you want to read?: ")
            properties = [input("Property " + str(i + 1) + " : ") for i in range(int(prop_num))]

            Eat.eat_groups(properties)  # Open-read-save the properties chosen by the user.

        elif component is 's':  # Stars
            Show.show_stellar_properties()  # Show them the properties.

            # Ask the user how many properties they want to read #
            prop_num = input("How many properties do you want to read?: ")
            properties = [input("Property " + str(i + 1) + " : ") for i in range(int(prop_num))]

            Eat.eat_stars(properties)  # Open-read-save the properties chosen by the user.

    elif read is 'n':  # Don't read.
        return None

    print("--- Finished reading in %.5s seconds ---" % (time.time() - start_time_Askread))  # Print read time.

    return None


class Show:
    """
    A class that contains all the methods (static) that show properties to the user.
    """


    @staticmethod
    def show_subhalo_properties():
        """
        A function to show subhalo properties to the user.

        :return: None
        """

        # Show the available properties to the user #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Found ' + str(len(f['Subhalo'].keys())) + ' properties for this subhalo:')
            pprint(str(f['Subhalo'].keys())[15:-2])
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        return None


    @staticmethod
    def show_stellar_properties():
        """
        A function to show stellar properties to the user.

        :return: None
        """

        # Show the available properties to the user #
        with h5py.File(file + str(first_file) + '.hdf5', 'r') as f:
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
            print('Found ' + str(len(f['Subhalo']['Stars'].keys())) + ' properties for stars' + ':')
            pprint(str(f['Subhalo']['Stars'].keys())[15:-2])
            print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')

        return None


class Eat:
    """
    A class that contains all the methods (static) to eat data.
    """


    @staticmethod
    def eat_groups(properties):
        """
        A function to eat (i.e., open-read-save) groups.

        :param properties: galactic properties (input from user from groups_io.Ask.read).
        :return: None

        """

        ga_number = 0
        # Determine the size of the arrays to hold the data #
        for iFile in range(first_file, last_file + 1):
            with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                shape = [len(f['Subhalo'][properties[i]].shape) for i in range(len(properties))]
                ga_number += f['Header'].attrs.get('Nsubgroups')
                print('Found ' + str(f['Header'].attrs.get('Nsubgroups')) + ' galaxies in file ' + str(iFile))
        print('Found ' + str(ga_number) + ' galaxies in total')
        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        # Eat the groups #
        for i in range(len(properties)):
            i_file = 0
            prop = np.empty(ga_number)

            # For 1D arrays #
            if shape[i] == 1:
                # Read in the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = f['Subhalo'][properties[i]]
                        i_file += n_file

            else:  # For 2D arrays.
                # Ask the user what particles they want to read #
                print('Property ' + properties[i] + ' is a 2D array')
                par_type = input('What type of particles do you want to use? 0-5: ')

                # Open and read the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = list(zip(*f['Subhalo'][properties[i]]))[int(par_type)]
                        i_file += n_file

            np.save('./data/groups/subhalo/' + properties[i], prop)

        return None


    @staticmethod
    def eat_stars(properties):
        """
        A function to eat (i.e., open-read-save) stars.

        :param properties: stellar properties (input from user read).
        :return: None
        """

        # Define the name/path of the files you want to read #
        ga_number = 0

        # Determine the size of the arrays to hold the data #
        for iFile in range(first_file, last_file + 1):
            with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                shape = [len(f['Subhalo']['Stars'][properties[i]].shape) for i in range(len(properties))]
                ga_number += f['Header'].attrs.get('Nsubgroups')
                print('Found ' + str(f['Header'].attrs.get('Nsubgroups')) + ' galaxies in file ' + str(iFile))
        print('Found ' + str(ga_number) + ' galaxies in total')
        print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
        # Eat the groups #
        for i in range(len(properties)):
            i_file = 0
            prop = np.empty(ga_number)

            # For 1D arrays #
            if shape[i] == 1:
                # Read in the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = f['Subhalo']['Stars'][properties[i]]
                        i_file += n_file

            else:  # For 2D arrays.
                # Ask the user what particles they want to read #
                print('Property ' + properties[i] + ' is a 2D array')
                par_type = input('What type of particles do you want to use? 0-5: ')

                # Open and read the data #
                for iFile in range(first_file, last_file + 1):
                    with h5py.File(file + '%i' % iFile + '.hdf5', 'r') as f:
                        n_file = f['Header'].attrs.get('Nsubgroups')
                        prop[i_file:i_file + n_file] = list(zip(*f['Subhalo']['Stars'][properties[i]]))[int(par_type)]
                        i_file += n_file

            np.save('./data/groups/stars/' + properties[i], prop)  # Save the arrays so you can load them multiple times in different scripts #

        return None