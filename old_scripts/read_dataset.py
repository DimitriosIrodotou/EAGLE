import h5py
import numpy as np


def read_particles(par_type, att, nfiles):
    """
    A function to read a selected type of particles.

    :param par_type: particle type
    :param att: attribute
    :param nfiles: number of files
    :return: data in physical units
    """

    # Output array.
    data = []  # Empty an array to hold the data.

    # Loop over each file and extract the data.
    for i in range(nfiles):
        with h5py.File(
                '/cosma6/data/dp004/dc-irod1/G-EAGLE/G-EAGLE_06/data/particledata_010_z005p000/eagle_subfind_particles_010_z005p000.%i.hdf5' % i,
                'r') as f:
            tmp = f['PartType' + str(par_type)][str(att)][...]
            data.append(tmp)

            # Get conversion factors.
            hexp = f['PartType' + str(par_type)][str(att)].attrs.get('h-scale-exponent')
            cgs = f['PartType' + str(par_type)][str(att)].attrs.get('CGSConversionFactor')
            aexp = f['PartType' + str(par_type)][str(att)].attrs.get('aexp-scale-exponent')

            # Get expansion factor and Hubble parameter from the header.
            a = f['Header'].attrs.get('Time')
            h = f['Header'].attrs.get('HubbleParam')

    # Combine to a single array.
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)

    # Convert to physical.
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, cgs * a ** aexp * h ** hexp, dtype='f8')

    return data


def read_subhaloes(att, nfiles):
    """
    A function to read subhaloes.

    :param att: attribute
    :param nfiles: number of files
    :return: data in physical units
    """

    data = []  # Empty an array to hold the data.

    # Loop over each file and extract the data #
    for i in range(nfiles):
        with h5py.File('/cosma6/data/dp004/dc-irod1/G-EAGLE/G-EAGLE_06/data/groups_010_z005p000/eagle_subfind_tab_010_z005p000.%i.hdf5' % i,
                       'r') as f:
            tmp = f['Subhalo'][str(att)][...]
            data.append(tmp)

            # Get conversion factors #
            hexp = f['Subhalo'][str(att)].attrs.get('h-scale-exponent')
            cgs = f['Subhalo'][str(att)].attrs.get('CGSConversionFactor')
            aexp = f['Subhalo'][str(att)].attrs.get('aexp-scale-exponent')

            # Get expansion factor and Hubble parameter from the header #
            a = f['Header'].attrs.get('Time')
            h = f['Header'].attrs.get('HubbleParam')

    # Combine to a single array #
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)

    # Convert to physical units #
    # if data.dtype != np.int32 and data.dtype != np.int64:
    #     data = np.multiply(data, cgs * a ** aexp * h ** hexp, dtype='f8')

    return data