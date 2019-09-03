import h5py


def read_header():
    """ Read various attributes from the header group. """
    with h5py.File('/cosma6/data/dp004/dc-irod1/G-EAGLE/G-EAGLE_06/data/particledata_010_z005p000/eagle_subfind_particles_010_z005p000.0.hdf5',
                   'r') as f:
        a = f['Header'].attrs.get('Time')  # Scale factor.
        h = f['Header'].attrs.get('HubbleParam')  # h.
        boxsize = f['Header'].attrs.get('BoxSize')  # L [Mph/h].

    return a, h, boxsize