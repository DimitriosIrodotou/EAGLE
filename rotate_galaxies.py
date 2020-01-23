import pylab

import numpy as np


class RotateGalaxies:
    
    @staticmethod
    def get_principal_axis(coordinates, masses, Ldir):
        
        px, py, pz = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        
        # Calculate the components of the moment of inertia tensor #
        tensor = pylab.zeros((3, 3))
        tensor[0, 0] = np.sum(masses * (py * py + pz * pz))
        tensor[1, 1] = np.sum(masses * (px * px + pz * pz))
        tensor[2, 2] = np.sum(masses * (px * px + py * py))
        
        tensor[0, 1] = - np.sum(masses * px * py)
        tensor[1, 0] = tensor[0, 1]
        tensor[0, 2] = - np.sum(masses * px * pz)
        tensor[2, 0] = tensor[0, 2]
        tensor[1, 2] = - np.sum(masses * py * pz)
        tensor[2, 1] = tensor[1, 2]
        
        # Get the eigenvalues and eigenvectors and calculate the principal axes #
        eigvalues, eigvectors = np.linalg.eig(tensor)
        
        A1 = np.sum(Ldir * eigvectors[:, 0])
        A2 = np.sum(Ldir * eigvectors[:, 1])
        A3 = np.sum(Ldir * eigvectors[:, 2])
        A = np.abs(np.array([A1, A2, A3]))
        
        # Align X axis with the major axis #
        i, = np.where(A == A.max())
        xdir = eigvectors[:, i[0]]
        
        if np.sum(xdir * Ldir) < 0:
            xdir *= -1.0
        
        # Align y-axis with the intermediate axis #
        j, = np.where(A != max(A))
        i2 = eigvalues[j].argsort()
        ydir = eigvectors[:, j[i2[1]]]
        
        if ydir[0] < 0:
            ydir *= -1.0
        
        zdir = np.cross(xdir, ydir)
        
        return xdir, ydir, zdir
    
    
    @staticmethod
    def rotate(property, dir1, dir2, dir3, glx_unit_vector):
        matrix = np.array([dir1, dir2, dir3])
        rotation_matrix = np.array(matrix.transpose())

        rotated_property = np.dot(property, rotation_matrix)

        # up = [1.0, 0.0, 0.0]
        # vec_in = np.asarray(glx_unit_vector)
        # vec_p1 = np.cross(up, vec_in)
        # vec_p1 = vec_p1 / np.sum(vec_p1 ** 2).sum() ** 0.5
        # vec_p2 = np.cross(vec_in, vec_p1)
        #
        # matrix = np.concatenate((vec_p1, vec_p2, vec_in)).reshape((3, 3))
        # rotmat = np.array(matrix.transpose())
        #
        # rotated_property = np.dot(rotmat, property.T).T
        
        return rotated_property