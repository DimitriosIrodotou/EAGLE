import numpy as np


class RotateCoordinates:
    
    @staticmethod
    def rotate(ra, dec, glx_unit_vector):
        """
        Rotate first about z-axis to set y=0and then about the y-axis to set z=0
        :param ra:
        :param dec:
        :param glx_unit_vector:
        :return:
        """
        # Calculate the rotation matrices and combine them #
        Rz = np.array([[np.cos(ra), np.sin(ra), 0], [-np.sin(ra), np.cos(ra), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(dec), 0, np.sin(dec)], [0, 1, 0], [-np.sin(dec), 0, np.cos(dec)]])
        Ryz = np.matmul(Ry, Rz)
        
        glx_unit_vector = np.matmul(Ryz, glx_unit_vector)
        
        # Calculate the ra and dec of the (unit vector of) angular momentum for each particle #
        rotated_ra = np.degrees(np.arctan2(glx_unit_vector[1], glx_unit_vector[0]))
        rotated_dec = np.degrees(np.arcsin(glx_unit_vector[2]))
        
        return rotated_ra, rotated_dec