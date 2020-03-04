import h5py
import numpy as np


def read_dataset_dm_mass():
    """ Special case for the mass of dark matter particles. """
    with h5py.File('/Users/Bam/PycharmProjects/EAGLE/EAGLE_data/particledata_028_z000p000/eagle_subfind_particles_028_z000p000.0.hdf5', 'r') as f:
        h = f['Header'].attrs.get('HubbleParam')
        a = f['Header'].attrs.get('Time')
        dm_mass = f['Header'].attrs.get('MassTable')[1]
        n_particles = f['Header'].attrs.get('NumPart_ThisFile')[1]

        # Create an array of length n_particles each set to dm_mass.
        m = np.ones(n_particles, dtype='f8') * dm_mass

        # Use the conversion factors from the mass entry in the gas particles.
        cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
        aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
        hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')

    # Convert to physical.
    m = np.multiply(m, cgs * a ** aexp * h ** hexp, dtype='f8')

    return m