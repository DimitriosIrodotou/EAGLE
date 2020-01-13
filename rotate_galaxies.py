import scipy
import pylab

import numpy as np


class RotateGalaxies:
    
    @staticmethod
    def get_principal_axis(coordinates, masses, Ldir):
        
        px, py, pz = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        
        # Calculate the components of the moment of inertia tensor #
        tensor = pylab.zeros((3, 3))
        tensor[0, 0] = (masses * (py * py + pz * pz)).sum()
        tensor[1, 1] = (masses * (px * px + pz * pz)).sum()
        tensor[2, 2] = (masses * (px * px + py * py)).sum()
        
        tensor[0, 1] = - (masses * px * py).sum()
        tensor[1, 0] = tensor[0, 1]
        tensor[0, 2] = - (masses * px * pz).sum()
        tensor[2, 0] = tensor[0, 2]
        tensor[1, 2] = - (masses * py * pz).sum()
        tensor[2, 1] = tensor[1, 2]
        
        # Get the eigenvalues and eigenvectors and calculate the principle axes #
        eigval, eigvec = scipy.linalg.eig(tensor)
        
        A1 = (Ldir * eigvec[:, 0]).sum()
        A2 = (Ldir * eigvec[:, 1]).sum()
        A3 = (Ldir * eigvec[:, 2]).sum()
        
        A = np.abs(np.array([A1, A2, A3]))
        i, = np.where(A == A.max())
        xdir = eigvec[:, i[0]]
        
        if (xdir * Ldir).sum() < 0:
            xdir *= -1.0
        
        j, = np.where(A != A.max())
        i2 = eigval[j].argsort()
        ydir = eigvec[:, j[i2[1]]]
        
        if ydir[0] < 0:
            ydir *= -1.0
        
        zdir = np.cross(xdir, ydir)
        
        return xdir, ydir, zdir
    
    
    @staticmethod
    def rotate(coordinates, dir1, dir2, dir3):
        matrix = np.array([dir1, dir2, dir3])
        rotmat = np.array(matrix.transpose())
        
        rotated_coordinates = np.dot(coordinates, rotmat)
        
        return rotated_coordinates