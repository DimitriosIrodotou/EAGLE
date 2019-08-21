import h5py
import numpy as np


# Read a selected dataset, itype is the PartType and att is the attribute name
def read_dataset(itype, att, nfiles=1):
    # Output array.
    data = []
    # Loop over each file and extract the data.
    for i in range(nfiles):
        f = h5py.File('/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_Data/0021/particledata_007_z008p000/eagle_subfind_particles_007_z008p000.%i.hdf5' % i, 'r')
        tmp = f['PartType%i/%s' % (itype, att)][...]
        data.append(tmp)
        # Get conversion factors.
        cgs = f['PartType%i/%s' % (itype, att)].attrs.get('CGSConversionFactor')
        aexp = f['PartType%i/%s' % (itype, att)].attrs.get('aexp-scale-exponent')
        hexp = f['PartType%i/%s' % (itype, att)].attrs.get('h-scale-exponent')
        # Get expansion factor and Hubble parameter from the header.
        a = f['Header'].attrs.get('Time')
        h = f['Header'].attrs.get('HubbleParam')
    f.close()
    # Combine to a single array.
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)
    # Convert to physical.
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, cgs * a ** aexp * h ** hexp, dtype='f8')
    return data