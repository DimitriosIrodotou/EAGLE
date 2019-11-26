import pylab

import numpy as np
import scipy.linalg

from const import *
from utilities import *
from cosmological_factors import *


class RotateGalaxies:
    
    @staticmethod
    def get_principal_axis(coordinates, masses, velocities):
        
        L = np.cross(coordinates.astype('float64'), (velocities.astype('float64') * masses[:, None].astype('float64')))
        Ltot = L.sum(axis=0)
        Ldir = Ltot / np.sqrt((Ltot ** 2).sum())
        
        px, py, pz = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        
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
    
    
    def rotateto(dir, dir2=None, dir3=None):
        if dir2 is None or dir3 is None:
            # get normals
            dir2 = pylab.zeros(3)
            if dir[0] != 0 and dir[1] != 0:
                dir2[0] = -dir[1]
                dir2[1] = dir[0]
            else:
                dir2[0] = 1
            dir2 /= np.sqrt((dir2 ** 2).sum())
            dir3 = np.cross(dir, dir2)
        
        matrix = np.array([dir, dir2, dir3])
        
        for value in data:
            if value == 'bhts' or value == 'bpos' or value == 'bvel':  # do not rotate the BH time step field
                continue
            if data[value] is not None and data[value].ndim == 2 and pylab.shape(data[value])[1] == 3:
                rotate_value(value, matrix)
        convenience()
        return