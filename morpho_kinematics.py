import numpy as np
from scipy import interpolate, linalg


class MorphoKinematic:

    @staticmethod
    def weighted_median(a, weights=None):
        """
        Calculate the weighted median.
        :param a: Input array.
        :param weights: Weights.
        :return: P (median)
        """

        if weights is None:
            weights = np.ones(np.array(a).shape)
        A = np.array(a).astype('float')
        W = np.array(weights).astype('float')
        if not (np.product(np.isnan(A))):
            I = np.argsort(A)
            cumweight = np.hstack([0, np.cumsum(W[I])])
            X = np.hstack([0, (cumweight[:-1] + cumweight[1:]) / (2 * cumweight[-1]), 1])
            Y = np.hstack([np.min(A), A[I], np.max(A)])
            P = interpolate.interp1d(X, Y)(0.5)
            return float(P)
        else:
            return np.nan


    @staticmethod
    def kinematic_diagnostics(coordinates, masses, velocities, binding_energies):
        """
        Calculate various kinematics parameters
        :param coordinates: Coordinates of particles.
        :param masses: Masses of particles.
        :param velocities: Velocities of particles.
        :param binding_energies: Specific binding energies of particles.
        :return: kappa, disc_fraction, circularity, rotational_over_dispersion, vrots, rotational_velocity, sigma_0, delta
        """

        # Group the attributes of the particles #
        prc_attributes = np.vstack([coordinates.T, masses, velocities.T, binding_energies]).T
        prc_distances = np.linalg.norm(prc_attributes[:, :3], axis=1)
        glx_mass = np.sum(prc_attributes[:, 3])

        # Calculate the angular momenta #
        prc_s_angular_momentum = np.cross(prc_attributes[:, :3], prc_attributes[:, 4:7])  # In kpc km s^-1.
        glx_angular_momentum = np.sum(prc_attributes[:, 3][:, np.newaxis] * prc_s_angular_momentum, axis=0)  # In Msun kpc km s^-1.
        glx_angular_momentum_magnitude = np.linalg.norm(glx_angular_momentum)  # In Msun kpc km s^-1.

        # Calculate cylindrical quantities #
        zaxis = glx_angular_momentum / glx_angular_momentum_magnitude  # Unit vector pointing along the glx_angular_momentum direction.
        zheight = np.sum(zaxis * prc_attributes[:, :3], axis=1)  # Projection of the coordinate vectors on the unit vector.
        cylposition = prc_attributes[:, :3] - zheight[:, np.newaxis] * [zaxis]
        cyldistances = np.sqrt(prc_distances ** 2 - zheight ** 2)
        smomentumz = np.sum(zaxis * prc_s_angular_momentum, axis=1) # z-component of the specific angular momentum.
        vrots = smomentumz / cyldistances
        vrads = np.sum(cylposition * prc_attributes[:, 4:7] / cyldistances[:, np.newaxis], axis=1)
        vheis = np.sum(zaxis * prc_attributes[:, 4:7], axis=1)

        # Calculate kinetic energy fraction invested in co-rotation #
        Mvrot2 = np.sum((prc_attributes[:, 3] * vrots ** 2)[vrots > 0])
        kappa = Mvrot2 / np.sum(prc_attributes[:, 3] * (np.linalg.norm(prc_attributes[:, 4:7], axis=1)) ** 2)

        # Calculate disc-to-total masses fraction estimated from the counter-rotating spheroid #
        disc_fraction = 1 - 2 * np.sum(prc_attributes[vrots <= 0, 3]) / glx_mass

        # Calculate the mean orbital circularity #
        sbindingenergy = prc_attributes[:, 7]
        sortE = np.argsort(sbindingenergy)
        unsortE = np.argsort(sortE)
        jzE = np.vstack([sbindingenergy, smomentumz]).T[sortE]
        circularity = (jzE[:, 1] / np.maximum.accumulate(np.abs(jzE[:, 1])))[unsortE]
        orbi = np.median(circularity)

        # Calculate rotation-to-dispersion and dispersion anisotropy parameter.
        rotational_velocity = np.abs(MorphoKinematic.weighted_median(vrots, weights=prc_attributes[:, 3]))
        sigma_xy = np.sqrt(np.average(np.sum(prc_attributes[:, [3]] * np.vstack([vrads, vrots]).T ** 2, axis=0) / glx_mass))
        sigma_0 = np.sqrt(sigma_xy ** 2 - 0.5 * rotational_velocity ** 2)
        sigma_z = np.sqrt(np.average(vheis ** 2, weights=prc_attributes[:, 3]))
        rotational_over_dispersion = rotational_velocity / sigma_0
        delta = 1 - (sigma_z / sigma_0) ** 2

        return kappa, disc_fraction, circularity, rotational_over_dispersion, vrots, rotational_velocity, sigma_0, delta


    @staticmethod
    def morphological_diagnostics(coordinates, masses, velocities, aperture=0.03, reduced_structure=True):
        """
        Calculate the morphological diagnostics through the (reduced or not) inertia tensor.

        Returns the morphological diagnostics for the input particles.
        masses : array_like of dtype float, shape (n, )
            Particles masses (in unit of masses M)
        velocities : array_like of dtype float, shape (n, 3)
            Particles coordinates (in unit of velocity V) such that velocities[:,0] = Vx,
            velocities[:,1] = Vy & velocities[:,2] = Vz
        reduced_structure : bool, optional
            Boolean to allow the computation to adopt the iterative reduced form of the
            inertia tensor. Default to True
        ellip : float
            The ellipticity parameter 1-c/a.
        triax : float
            The triaxiality parameter (a^2-b^2)/(a^2-c^2).
        Transform : array of dtype float, shape (3, 3)
            The orthogonal matrix representing the 3 axes as unit vectors: in real-world
            coordinates, Transform[0] = major, Transform[1] = inter, Transform[2] = minor.
        abc : array of dtype float, shape (3, )
            The corresponding (a,b,c) lengths (in unit of length L).
        :param coordinates: Coordinates of the particles.
        :param masses: Masses of particles.
        :param velocities:
        :param reduced_structure:
        :return:
        """

        # Group the attributes of the particles #
        particlesall = np.vstack([coordinates.T, masses, velocities.T]).T
        distancesall = np.linalg.norm(particlesall[:, :3], axis=1)

        # Restrict particles
        extract = (distancesall < aperture)
        particles = particlesall[extract].copy()
        prc_distances = distancesall[extract].copy()
        glx_mass = np.sum(particles[:, 3])

        # Calculate glx_angular_momentum
        prc_s_angular_momentum = np.cross(particlesall[:, :3], particlesall[:, 4:7])
        glx_angular_momentum = np.sum(particles[:, 3][:, np.newaxis] * prc_s_angular_momentum[extract], axis=0)

        # Calculate morphological diagnostics
        s = 1
        q = 1
        Rsphall = 1 + reduced_structure * (distancesall - 1)
        stop = False
        while not ('structure' in locals()) or (reduced_structure and not (stop)):
            particles = particlesall[extract].copy()
            Rsph = Rsphall[extract]
            Rsph /= np.median(Rsph)

            # Calculate structure tensor
            structure = np.sum(
                (particles[:, 3] / Rsph ** 2)[:, np.newaxis, np.newaxis] * (np.matmul(particles[:, :3, np.newaxis], particles[:, np.newaxis, :3])),
                axis=0) / np.sum(particles[:, 3] / Rsph ** 2)

            # Diagonalise structure tensor
            eigval, eigvec = linalg.eigh(structure)

            # Get structure direct oriented orthonormal base
            eigvec[:, 2] *= np.round(np.sum(np.cross(eigvec[:, 0], eigvec[:, 1]) * eigvec[:, 2]))

            # Return minor axe
            structmainaxe = eigvec[:, np.argmin(eigval)].copy()

            # Permute base and align Y axis with minor axis in glx_angular_momentum direction
            sign = int(np.sign(np.sum(glx_angular_momentum * structmainaxe) + np.finfo(float).tiny))
            structmainaxe *= sign
            temp = np.array([1, sign, 1]) * (eigvec[:, (np.argmin(eigval) + np.array([(3 + sign) / 2, 0, (3 - sign) / 2])) % 3])
            eigval = eigval[(np.argmin(eigval) + np.array([(3 + sign) / 2, 0, (3 - sign) / 2])) % 3]

            # Permute base to align Z axis with major axis
            foo = (np.argmax(eigval) / 2) * 2
            temp = np.array([(-1) ** (1 + foo / 2), 1, 1]) * (temp[:, [2 - foo, 1, foo]])
            eigval = eigval[[2 - foo, 1, foo]]
            # Calculate change of basis matrix
            transform = linalg.inv(temp)
            stop = (np.max((1 - np.sqrt(eigval[:2] / eigval[2]) / np.array([q, s])) ** 2) < 1e-4)
            if (reduced_structure and not (stop)):
                q, s = np.sqrt(eigval[:2] / eigval[2])
                Rsphall = linalg.norm(np.matmul(transform, particlesall[:, :3, np.newaxis])[:, :, 0] / np.array([q, s, 1]), axis=1)
                extract = (Rsphall < aperture / (q * s) ** (1 / 3.))
        Transform = transform.copy()
        ellip = 1 - np.sqrt(eigval[1] / eigval[2])
        triax = (1 - eigval[0] / eigval[2]) / (1 - eigval[1] / eigval[2])
        Transform = Transform[..., [2, 0, 1], :]  # so that transform[0] = major, transform[1] = inter, transform[2] = minor
        abc = np.sqrt(eigval[[2, 0, 1]])

        return ellip, triax, Transform, abc


    @staticmethod
    def r_mass(stellar_data_tmp, fraction):
        """
        Calculate the radius that contains a provided fraction of the total stellar mass.
        stellar_data_tmp: from read_add_attributes.py.
        :return: r_mass
        """

        # Calculate the spherical distance of each particle and sort their masses based on that #
        prc_spherical_radius = np.sqrt(np.sum(stellar_data_tmp['Coordinates'] ** 2, axis=1))
        sort = np.argsort(prc_spherical_radius)
        sorted_prc_spherical_radius = prc_spherical_radius[sort]

        # Calculate r_50 #
        total_mass = np.sum(stellar_data_tmp['Mass'])
        cumulative_mass = np.cumsum(stellar_data_tmp['Mass'][sort])
        index = np.argmin(np.abs(cumulative_mass - (fraction * total_mass)))
        r_mass = sorted_prc_spherical_radius[index]

        return r_mass
