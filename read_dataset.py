import h5py
import numpy as np


def read_dataset(par_type, att, nfiles=1):
    """
    A function to read a selected dataset.
    :param par_type: particle type.
    :param att: attribute.
    :param nfiles: number of files.
    :return: datain physical units.
    """

    # Output array.
    data = []

    # Loop over each file and extract the data.
    for i in range(nfiles):
        with h5py.File('/Users/Bam/PycharmProjects/EAGLE/EAGLE_data/particledata_028_z000p000/eagle_subfind_particles_028_z000p000.%i.hdf5' % i,
                       'r') as f:
            tmp = f['PartType%i/%s' % (par_type, att)][...]
            data.append(tmp)

            # Get conversion factors.
            cgs = f['PartType' + par_type + att].attrs.get('CGSConversionFactor')
            aexp = f['PartType' + par_type + att].attrs.get('aexp-scale-exponent')
            hexp = f['PartType' + par_type + att].attrs.get('h-scale-exponent')

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