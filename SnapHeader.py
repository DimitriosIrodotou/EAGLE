import h5py


def read_header():
    # Read various attributes from the header group
    f = h5py.File('/Users/Bam/PycharmProjects/G-EAGLE/G-EAGLE_Data/0021/groups_010_z005p000/eagle_subfind_tab_010_z005p000.0.hdf5', 'r')
    a = f['Header'].attrs.get('Time')  # Scale factor.
    h = f['Header'].attrs.get('HubbleParam')  # h.
    boxsize = f['Header'].attrs.get('BoxSize')  # L [Mph/h].
    f.close()
    return a, h, boxsize