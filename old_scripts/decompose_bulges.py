import numpy as np


class DecomposeBulges:
    """
    Decompose bulges into three sub-regions according to their binding energy.
    """
    
    
    @staticmethod
    def central_bulge(bulge_mask, stellar_data_tmp):
        """
        Rotate first about z-axis to set y=0 and then about the y-axis to set z=0
        :param bulge_mask: from plots
        :param stellar_data_tmp: from mask_galaxies
        :return: stellar_data_tmp['Coordinates'], stellar_data_tmp['Velocity'], prc_unit_vector, glx_unit_vector
        """
        # Calculate the cylindrical distance of each particle and sort their masses based on that #
        cylindrical_distance = np.sqrt(stellar_data_tmp['Coordinates'][:, 0] ** 2 + stellar_data_tmp['Coordinates'][:, 1] ** 2)
        sort = np.argsort(cylindrical_distance)
        sorted_cylindrical_distance = cylindrical_distance[sort]
        
        # Calculate the optical radius of the galaxy as the radius that contains 83% of the total stellar mass #
        j = 0
        running_mass = 0.0
        optical_mass = 0.83 * np.sum(stellar_data_tmp['Mass'])
        while running_mass < optical_mass:
            running_mass += stellar_data_tmp['Mass'][sort][j]
            j += 1
        
        optical_radius = sorted_cylindrical_distance[j - 1]
        
        # Find the bulge particles that are more bound than the minimum binding energy of stars with r ≥ 0.5 * optical radius #
        radius_mask = np.where(cylindrical_distance >= 2 * optical_radius)[0]
        energy_mask = np.where(stellar_data_tmp['ParticleBindingEnergy'][bulge_mask] <= min(stellar_data_tmp['ParticleBindingEnergy'][radius_mask]))[
            0]
        
        import matplotlib.pyplot as plt
        
        plt.close()
        plt.figure(0, figsize=(20, 22.5))
        count, xedges, yedges = np.histogram2d(stellar_data_tmp['Coordinates'][energy_mask, 0], stellar_data_tmp['Coordinates'][energy_mask, 1],
                                               bins=500, range=[[-30, 30], [-30, 30]])
        plots_path = '/cosma7/data/dp004/dc-irod1/EAGLE/python/plots/SP/'  # Path to save plots.
        plt.imshow(count.T, extent=[-30, 30, -30, 30], origin='lower', cmap='nipy_spectral_r', interpolation='gaussian', aspect='auto')
        plt.savefig(plots_path + 'SP' + '-' + '.png', bbox_inches='tight')
        
        return None