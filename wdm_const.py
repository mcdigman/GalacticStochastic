"""wavelet transform constants"""

from collections import namedtuple

import numpy as np

#LISA constants
Larm = 2.5e9 # Mean arm length of the LISA detector (meters)
Sps = 2.25e-22 # Photon shot noise power
Sacc = 9.0e-30 # Acceleration noise power
kappa0 = 0.0 # Initial azimuthal position of the guiding center
lambda0 = 0.0 # Initial orientation of the LISA constellation
fstr = 0.01908538064 # Transfer frequency
ec = 0.0048241852175 # LISA orbital eccentricity; should be Larm/(2*AU*np.sqrt(3))?
fm = 3.168753575e-8 # LISA modulation frequency


LISAConstants = namedtuple('LISAConstants', ['Larm', 'Sps', 'Sacc', 'kappa0', 'lambda0', 'fstr', 'ec', 'fm'])

lisa_const = LISAConstants(Larm, Sps, Sacc, kappa0, lambda0, fstr, ec, fm)
