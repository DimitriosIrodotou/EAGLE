from __future__ import print_function

import numpy as np
# all constants in cgs

# gravitational constant
G = 6.674e-8            # [ cm^3 g^-1 s^-1 ]

# avogadro constant
NA = 6.0221418e23       # [ ]

# boltzmann constant
KB = 1.3806504e-16      # [ erg K^-1 ]

# planck constant
H = 6.62606896e-27      # [ erg s ]

# speed of light in vacuum
c = 2.99792458e10       # [ cm s^-1 ]

# solar mass
msol = 1.989e33         # [ g ]

# solar radius
rsol = 6.955e10         # [ cm ]

# solar luminosity
lsol = 3.839e33         # [ erg s^-1 ]

# electron charge
qe = 4.80320427e-10 # [ esu ]

# atomic mass unit
amu = 1.6605390401e-24 # [ g ]

# ev2erg
ev2erg = 1.602177e-12 # [ erg eV^-1 ]

# parsec in cm
parsec = 3.08568025e18 # [ cm ]

# conversion factor for cosmological magnetic field
bfac = np.sqrt(1e10 * msol) / np.sqrt(1e6 * parsec) * 1e5 / (1e6 * parsec)

# golden ratio for image heights
golden_ratio = (np.sqrt(5)-1)/2

# colors
colorset = [ 'k', 'b', 'r', 'g', 'c', 'm', 'y' ]

# atomic symbols
asymb = [ 'neutron', 'h', 'he',
          'li', 'be',  'b',  'c',  'n',  'o',  'f', 'ne',
          'na', 'mg', 'al', 'si',  'p',  's', 'cl', 'ar',
           'k', 'ca', 'sc', 'ti',  'v', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
          'rb', 'sr',  'y', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te',  'i', 'xe',
          'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'hf',
          'ta',  'w', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
          'fr', 'ra', 'ac', 'th', 'pa',  'u', 'np', 'pu', 'am', 'cm', 'bk', 'cf', 'es', 'fm', 'md', 'no', 'lr', 'rf',
          'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg','uub','uut','uuq','uup','uuh','uus','uuo' ]

# atomic numbers to symbols
asymb2num = {}
for i in range( len( asymb ) ):
    if (asymb[i] in asymb2num):
        print("asymb2num key error: key `%s` already exists." % asymb[i])
    else:
        asymb2num[ asymb[i] ] = i

class nuc:
    def __init__( self, name ):
        self.name = name

        self.na, self.nz = name2nuc( self.name )
        self.symb = asymb[self.nz]
        self.nn = self.na - self.nz
        return

    def get_helm_line(self):
        return "{:>5s}{:4d}{:4d}".format(self.name, self.na, self.nz)


def nuc2name( na, nz ):
    if na == 1 and nz == 0:
        return 'n'
    elif na == 1 and nz == 1:
        return 'p'
    elif na == 2 and nz == 1:
        return 'd'
    elif na == 3 and nz == 1:
        return 't'
    else:
        return "%s%d" % (asymb[nz], na)

def name2nuc( name ):   # returns na, nz
    nucname = ""
    nuca = ""

    name = name.strip().lower()
    for j in range( len( name ) ):
        if name[j].isalpha():
            nucname += name[j]
        else:
            nuca += name[j]

    if len( nuca ) > 0:
        return int( nuca ), asymb2num[ nucname ]
    else:
        if ( nucname == "n" ):
            return 1, 0
        elif ( nucname == "p" ):
            return 1, 1
        elif ( nucname == "d" ):
            return 2, 1
        elif ( nucname == "t" ):
            return 3, 1
        else:
            print("unknown nucleus:", name)
            return 0, 0
    return 0, 0
