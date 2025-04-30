"""wavelet transform constants"""

from collections import namedtuple

import numpy as np

Nf = 2048 # frequency layers
Nt = 512*8
dt = 1.00250244140625*8*15./(2*Nf*Nt/2048**2) # time cadence

mult = 8  # over sampling

Nsf = 150  # frequency steps
Nfd = 80  # number of f-dots
dfdot = 0.1 # fractional fdot increment
Nfd_negative = 40

Nst = mult*100  # time steps

nx = 4.0    # filter steepness in frequency



dkstep = np.int64(Nst//mult)
if dkstep*mult != Nst:
    raise ValueError('ratio of Nst and mult must be an integer')


#LISA constants
Larm = 2.5e9 # Mean arm length of the LISA detector (meters)
Sps = 2.25e-22 # Photon shot noise power
Sacc = 9.0e-30 # Acceleration noise power
kappa0 = 0.0 # Initial azimuthal position of the guiding center
lambda0 = 0.0 # Initial orientation of the LISA constellation
fstr = 0.01908538064 # Transfer frequency
ec = 0.0048241852175 # LISA orbital eccentricity; should be Larm/(2*AU*np.sqrt(3))?
fm = 3.168753575e-8 # LISA modulation frequency


#derived constants
N = Nt*Nf #total points
Tobs = dt*N #duration
DT = dt*Nf #width of wavelet pixel in time
DF = 1./(2*dt*Nf)   #width of wavelet pixel in frequency
K = mult*2*Nf #filter length
Tw = dt*K #filter duration
L = 512  #reduced filter length - must be a power of 2
p = K/L #downsample factor
dom = 2.*np.pi/Tw #angular frequency spacing
OM = np.pi/dt  #Nyquist angular frequency
DOM = OM/Nf #2 pi times DF
insDOM = 1./np.sqrt(DOM)
B = OM/(2*Nf)
A = (DOM-B)/2
BW = (A+B)/np.pi #total width of wavelet in frequency
#nonzero terms in phi transform (only need 0 and positive)
df = BW/Nsf

dfd = DF/Tw*dfdot

#number of TDI channels to use
NC = 3


WDMWaveletConstants = namedtuple('WDMWaveletConstants', ['Nf', 'Nt', 'dt', 'mult', 'Nsf', 'Nfd', 'dfdot', 'Nfd_negative', 'Nst', 'Tobs', 'NC', 'DF', 'DT', 'nx', 'dfd', 'df', 'BW', 'Tw', 'K', 'A', 'B', 'dom', 'DOM', 'insDOM'])

wdm_const = WDMWaveletConstants(Nf, Nt, dt, mult, Nsf, Nfd, dfdot, Nfd_negative, Nst, Tobs, NC, DF, DT, nx, dfd, df, BW, Tw, K, A, B, dom, DOM, insDOM)

LISAConstants = namedtuple('LISAConstants', ['Larm', 'Sps', 'Sacc', 'kappa0', 'lambda0', 'fstr', 'ec', 'fm'])

lisa_const = LISAConstants(Larm, Sps, Sacc, kappa0, lambda0, fstr, ec, fm)
