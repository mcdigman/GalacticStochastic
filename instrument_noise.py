"""get the instrument noise profile"""

import numpy as np
from numpy.random import normal

#import numba as nb
from numba import njit
#from numba.experimental import jitclass

from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

#try:
#    import numpy.random_intel as random
#except ImportError:
#    import numpy.random as random
#import numpy.random as random


def instrument_noise1(f, lc):
    #Power spectral density of the detector noise and transfer frequency
    SAE = np.zeros(f.size)
    Sps = 9.e-24 #should match sangria v2? Should it be backlinknoise or readoutnoise?
    Sacc = 5.76e-30 #from sangria v2
    fonfs = f/lc.fstr
    #To match the LDC power spectra need a factor of 2 here. No idea why... (one sided/two sided?)
    LC = 2.0*fonfs*fonfs
    #roll-offs
    rolla = (1.0+pow((4.0e-4/f), 2.0))*(1.0+pow((f/8.0e-3), 4.0))
    rollw = (1.0+pow((2.0e-3/f), 4.0))
    #Calculate the power spectral density of the detector noise at the given frequency
    #not and exact match to the LDC, but within 10%
    SAE = LC*16.0/3.0*pow(np.sin(fonfs), 2.0)*( (2.0+np.cos(fonfs))*(Sps)*rollw + 2.0*(3.0+2.0*np.cos(fonfs)+np.cos(2.0*fonfs))*(Sacc/pow(2.0*np.pi*f, 4.0)*rolla) ) / pow(2.0*lc.Larm, 2.0)
    return SAE


#@njit()
def instrument_noise_AET(f, lc, wc):
    """get power spectral density in all 3 channels, assuming identical in all arms"""
    #see arXiv:2005.03610
    #see arXiv:1002.1291
    fonfs = f/lc.fstr

    LC = 64/(3*lc.Larm**2)
    mult_all = LC*fonfs**2*np.sin(fonfs)**2
    mult_sa = (4*lc.Sacc/(2*np.pi)**4)*(1+16.e-8/f**2)*(1.0+(f/8.0e-3)**4.)/f**4
    mult_sp = lc.Sps*(1.0+(2.0e-3/f)**4.)

    cosfonfs = np.cos(fonfs)

    SAET = np.zeros((f.size, wc.NC))

    #SAET[:, 0] = mult_all*(mult_sa*(1+cosfonfs+cosfonfs**2)+mult_sp*(2+cosfonfs))
    SAET[:, 0] = instrument_noise1(f, lc) #TODO make this all self consistent
    SAET[:, 1] = SAET[:, 0]
    SAET[:, 2] = mult_all*(mult_sa/2*(1-2*cosfonfs+cosfonfs**2)+mult_sp*(1-cosfonfs))
    return SAET


#@njit()
def instrument_noise_AET_wdm_m(lc, wc):
    """
    get the instrument noise curve as a function of frequency for the wdm
    wavelet decomposition

    Parameters
    ----------
    lc : namedtuple
        constants for LISA constellation specified in wdm_const.py
    wc : namedtuple
        constants for WDM wavelet basis also from wdm_const.py

    Returns
    -------
    SAET_m : numpy.ndarray (Nf x NC)
        array of the instrument noise curve for each TDI channel
        array shape is (freq. layers x number of TDI channels)
    """

    #TODO why no plus 1?
    ls = np.arange(-wc.Nt//2, wc.Nt//2)
    fs = ls/wc.Tobs
    phif = np.sqrt(wc.dt)*phitilde_vec(2*np.pi*fs*wc.dt, wc.Nf, wc.nx)

    #TODO check ad hoc normalization factor
    SAET_m = instrument_noise_AET_wdm_loop(phif, lc, wc)
    return SAET_m


#@njit()
def instrument_noise_AET_wdm_loop(phif, lc, wc):
    """helper to get the instrument noise for wdm"""
    #realistically this really only needs run once and is fast enough without jit
    #TODO check normalization
    #TODO get first and last bins correct
    #nrm =   np.sqrt(2*wc.Nf*wc.dt)*np.linalg.norm(phif)
    #nrm =   2*np.sqrt(2*wc.dt)*np.linalg.norm(phif)
    nrm = np.sqrt(12318/wc.Nf)*np.linalg.norm(phif)
    print('nrm instrument', nrm)
    phif /= nrm
    phif2 = phif**2

    SAET_M = np.zeros((wc.Nf, wc.NC))
    half_Nt = wc.Nt//2
    fs_long = np.arange(-half_Nt, half_Nt+wc.Nf*half_Nt)/wc.Tobs
    #prevent division by 0
    fs_long[half_Nt] = fs_long[half_Nt+1]
    SAET_long = instrument_noise_AET(fs_long, lc, wc)
    #excise the f=0 point
    SAET_long[half_Nt, :] = 0.
    #apply window in loop
    for m in range(0, wc.Nf):
        SAET_M[m] = np.dot(phif2, SAET_long[m*half_Nt:(m+2)*half_Nt])

    return SAET_M


#@jitclass([('prune', nb.b1), ('SAET', nb.float64[:, :, :]), ('inv_SAET', nb.float64[:, :, :]), ('inv_chol_SAET', nb.float64[:, :, :]), ('chol_SAET', nb.float64[:, :, :])])
class DiagonalNonstationaryDenseInstrumentNoiseModel:
    """
    a class to handle the fully diagonal nonstationary
    instrument noise model to feed to snr and fisher matrix calculations
    """
    def __init__(self, SAET, wc, prune):
        """
        initialize the fully diagonal, nonstationary instrument noise model

        Parameters
        ----------
        SAET : numpy.ndarray
            array of instrument noise curve for each TDI curve - usually output
            from instrument_noise_AET_wdm_m
            shape: (Nf x NC)=(freq layers x number of TDI channels)
        wc : namedtuple
            constants for WDM wavelet basis also from wdm_const.py
        prune : bool
            if prune=True, cut the 1st and last values,
            which may not be calculated correctly

        Returns
        -------
        DiagonalNonstationaryDenseInstrumentNoiseModel : class
        """
        self.prune = prune
        self.SAET = SAET
        self.wc = wc
        self.inv_SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.inv_chol_SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.chol_SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        for j in range(0, self.wc.Nt):
            for itrc in range(0, self.wc.NC):
                self.chol_SAET[j, :, itrc] = np.sqrt(self.SAET[j, :, itrc])
                self.inv_chol_SAET[j, :, itrc] = 1./self.chol_SAET[j, :, itrc]
                self.inv_SAET[j, :, itrc] = self.inv_chol_SAET[j, :, itrc]**2
        if self.prune:
            self.chol_SAET[:, 0, :] = 0.
            self.inv_chol_SAET[:, 0, :] = 0.
            self.inv_SAET[:, 0, :] = 0.


    def generate_dense_noise(self):
        """
        generate random noise for full matrix

        Parameters
        ----------
        There are no parameters required

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, NC) Number of time pixels,
            Freq layers, Number of TDI channels. All specified by wdm_const.py
        """
        noise_res = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        for j in range(0, self.wc.Nt):
            noise_res[j, :, :] = normal(0., 1., (self.wc.Nf, self.wc.NC))*self.chol_SAET[j, :, :]
        return noise_res


    def get_sparse_snrs(self, NUs, lists_pixels, wavelet_data, nt_min=0, nt_max=-1):
        """get snr of waveform in each channel"""
        return get_sparse_snr_helper(NUs, lists_pixels, wavelet_data, nt_min, nt_max, self.wc, self.inv_chol_SAET)


#@jitclass([('prune', nb.b1), ('SAET_m', nb.float64[:, :]), ('inv_SAET_m', nb.float64[:, :]), ('inv_chol_SAET_m', nb.float64[:, :]), ('SAET', nb.float64[:, :, :]), ('inv_SAET', nb.float64[:, :, :]), ('inv_chol_SAET', nb.float64[:, :, :]), ('chol_SAET_m', nb.float64[:, :]), ('chol_SAET', nb.float64[:, :, :]), ('mean_SAE', nb.float64[:]), ('inv_chol_mean_SAE', nb.float64[:])])
class DiagonalStationaryDenseInstrumentNoiseModel:
    """a class to handle the fully diagonal stationary
    instrument noise model to feed to snr and fisher matrix calculations"""
    def __init__(self, SAET_m, wc, prune):
        """initialize the stationary instrument noise model

        Parameters
        ----------
        SAET : numpy.ndarray
            array of instrument noise curve for each TDI curve - usually output
            from instrument_noise_AET_wdm_m
            shape: (Nf x NC) freq layers x number of TDI channels
        wc : namedtuple
            constants for WDM wavelet basis also from wdm_const.py
        prune : bool
            if prune=True, cut the 1st and last values,
            which may not be calculated correctly

        Returns
        -------
        DiagonalStationaryDenseInstrumentNoiseModel : class
        """
        #self.NC = NC
        self.prune = prune
        #self.SAET_m = instrument_noise_AET_wdm_m()
        self.SAET_m = SAET_m
        self.wc = wc
        self.inv_SAET_m = np.zeros((self.wc.Nf, self.wc.NC))
        self.inv_chol_SAET_m = np.zeros((self.wc.Nf, self.wc.NC))
        self.chol_SAET_m = np.zeros((self.wc.Nf, self.wc.NC))
        for m in range(0, self.wc.Nf):
            if self.prune and (m==0 or m==wc.Nf):
                #currently m can't be Nf but that is the value that should be pruned
                continue
            for itrc in range(0, self.wc.NC):
                self.inv_SAET_m[m, itrc] = 1./self.SAET_m[m, itrc]
                self.chol_SAET_m[m, itrc] = np.sqrt(self.SAET_m[m, itrc])
                self.inv_chol_SAET_m[m, itrc] = 1./self.chol_SAET_m[m, itrc]

        self.SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.inv_SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.inv_chol_SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.chol_SAET = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        for j in range(0, self.wc.Nt):
            for itrc in range(0, self.wc.NC):
                self.SAET[j, 1:, itrc] = self.SAET_m[1:, itrc]
                self.inv_SAET[j, 1:, itrc] = self.inv_SAET_m[1:, itrc]
                self.inv_chol_SAET[j, 1:, itrc] = self.inv_chol_SAET_m[1:, itrc]
                self.chol_SAET[j, 1:, itrc] = self.chol_SAET_m[1:, itrc]
                if j%2==0:
                    #NOTE right now this just necessarily drops the highest frequency term, also note that lowest frequency term may be a bit different
                    self.SAET[j, 0, itrc] = self.SAET_m[0, itrc]
                    self.inv_SAET[j, 0, itrc] = self.inv_SAET_m[0, itrc]
                    self.inv_chol_SAET[j, 0, itrc] = self.inv_chol_SAET_m[0, itrc]
                    self.chol_SAET[j, 0, itrc] = self.chol_SAET_m[0, itrc]

        self.mean_SAE = SAET_m[:, 0]
        self.inv_chol_mean_SAE = 1./np.sqrt(self.mean_SAE)
        if self.prune:
            self.mean_SAE[0] = 0.
            self.inv_chol_mean_SAE[0] = 0.
            #currently not right size for pruning


    def generate_dense_noise(self):
        """
        generate random noise for full matrix

        Parameters
        ----------
        There are no parameters required

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, NC) Number of time pixels,
            Freq layers, Number of TDI channels. All specified by wdm_const.py
        """
        noise_res = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        for j in range(0, self.wc.Nt):
            noise_res[j, :, :] = normal(0., 1., (self.wc.Nf, self.wc.NC))*self.chol_SAET[j, :, :]
        return noise_res


    def get_sparse_snrs(self, NUs, lists_pixels, wavelet_data, nt_min=0, nt_max=-1):
        """
        get s/n of waveform in each TDI channel. parameters usually come from
        BinaryWaveletAmpFreqDT.get_unsorted_coeffs() from
        wavelet_detector_waveforms.

        Parameters
        ----------
        NUs : numpy.ndarray
            number of wavelet coefficients used in sparse representation
            shape: number of TDI channels
        lists_pixels : numpy.ndarray
            stores the index of x,y coordinates of the pixels that
            shape: (NC, ____) number of TDI channels x total possible wavelet
            basis. Total possible wavelet basis is specified in
            wavelet_detector_waveforms.py
        wavelet_data : numpy.ndarray
            stores the value of the pixels specified by lists_pixels.
            Shape is the same as lists_pixels: shape: (NC, ____)
        nt_min : int, default=0
            time pixels that are start/end of slice for evaluating.
            Used for selecting a subset of time pixels
        nt_max : int, default=-1

        Returns
        -------
        snr : numpy.ndarray
            an array of shape (NC) which is the S/N for each TDI channel.
        """
        if nt_max == -1:
            nt_max = self.wc.Nt
        return get_sparse_snr_helper(NUs, lists_pixels, wavelet_data, nt_min, nt_max, self.wc, self.inv_chol_SAET)


@njit()
def get_sparse_snr_helper(NUs, lists_pixels, wavelet_data, nt_min, nt_max, wc, inv_chol_SAET):
    """
    calculates the S/N ratio for each TDI channel for a given waveform.

    Parameters
    ----------
    NUs : numpy.ndarray
        number of wavelet coefficients used in sparse representation
        shape: number of TDI channels
    lists_pixels
    wavelet_data
    nt_min
    nt_max
    wc : namedtuple
        constants for WDM wavelet basis also from wdm_const.py
    inv_chol_SAET :


    Returns
    -------

    """
    if nt_max == -1:
        nt_max = wc.Nt
    snr2s = np.zeros(wc.NC)
    for itrc in range(0, wc.NC):
        i_itrs = np.mod(lists_pixels[itrc, 0:NUs[itrc]], wc.Nf).astype(np.int64)
        j_itrs = (lists_pixels[itrc, 0:NUs[itrc]]-i_itrs)//wc.Nf
        for mm in range(0, NUs[itrc]):
            if nt_min<=j_itrs[mm]<nt_max:
                mult = inv_chol_SAET[j_itrs[mm], i_itrs[mm], itrc]*wavelet_data[itrc, mm]
                snr2s[itrc]+= mult*mult
        return np.sqrt(snr2s)
