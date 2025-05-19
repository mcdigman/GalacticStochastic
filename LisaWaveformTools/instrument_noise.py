"""get the instrument noise profile"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

# import numba as nb
from numba import njit
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

from LisaWaveformTools.lisa_config import LISAConstants
from WaveletWaveforms.wdm_config import WDMWaveletConstants

# from numba.experimental import jitclass

class DenseNoiseModel(ABC):
    def __init__(self, wc: WDMWaveletConstants, prune, seed=-1) -> None:
        self.wc = wc
        self.prune = prune
        self.seed = seed

    @abstractmethod
    def generate_dense_noise(self) -> npt.NDArray[np.float64]:
        """Generate random noise for full matrix"""

    @abstractmethod
    def get_sparse_snrs(self, wavelet_waveform, nt_min=0, nt_max=-1) -> npt.NDArray[np.float64]:
        """Get S/N of waveform in each TDI channel"""

    @abstractmethod
    def get_S_stat_m(self) -> npt.NDArray[np.float64]:
        """Get the mean noise covariance matrix as a function of time"""

    @abstractmethod
    def get_inv_chol_S(self) -> npt.NDArray[np.float64]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""

    @abstractmethod
    def get_inv_S(self) -> npt.NDArray[np.float64]:
        """Get the inverse of the dense noise covariance matrix"""

    @abstractmethod
    def get_chol_S(self) -> npt.NDArray[np.float64]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""

    @abstractmethod
    def get_S(self) -> npt.NDArray[np.float64]:
        """Get the dense noise covariance matrix"""


def instrument_noise1(f, lc: LISAConstants):
    # Power spectral density of the detector noise and transfer frequency
    SAE = np.zeros(f.size)
    Sps = 9.e-24     # should match sangria v2? Should it be backlinknoise or readoutnoise?
    Sacc = 5.76e-30  # from sangria v2
    fonfs = f / lc.fstr
    # To match the LDC power spectra need a factor of 2 here. No idea why... (one sided/two sided?)
    LC = 2.0 * fonfs * fonfs
    # roll-offs
    rolla = (1.0 + pow((4.0e-4 / f), 2.0)) * (1.0 + pow((f / 8.0e-3), 4.0))
    rollw = 1.0 + pow((2.0e-3 / f), 4.0)
    # Calculate the power spectral density of the detector noise at the given frequency
    # not and exact match to the LDC, but within 10%
    SAE = LC * 16.0 / 3.0 * pow(np.sin(fonfs), 2.0) * (
            (2.0 + np.cos(fonfs)) * Sps * rollw + 2.0 * (3.0 + 2.0 * np.cos(fonfs) + np.cos(2.0 * fonfs))
            *
            (Sacc / pow(2.0 * np.pi * f, 4.0) * rolla)) / pow(2.0 * lc.Larm, 2.0)
    return SAE


# @njit()
def instrument_noise_AET(f, lc: LISAConstants, wc: WDMWaveletConstants):
    """Get power spectral density in all 3 channels, assuming identical in all arms"""
    # see arXiv:2005.03610
    # see arXiv:1002.1291
    fonfs = f / lc.fstr

    LC = 64 / (3 * lc.Larm**2)
    mult_all = LC * fonfs**2 * np.sin(fonfs)**2
    mult_sa = (4 * lc.Sacc / (2 * np.pi)**4) * (1 + 16.e-8 / f**2) * (1.0 + (f / 8.0e-3)**4.) / f**4
    mult_sp = lc.Sps * (1.0 + (2.0e-3 / f)**4.)

    cosfonfs = np.cos(fonfs)

    S_inst = np.zeros((f.size, wc.NC))

    S_inst[:, 0] = instrument_noise1(f, lc)  # TODO make this all self consistent
    S_inst[:, 1] = S_inst[:, 0]
    S_inst[:, 2] = mult_all * (mult_sa / 2 * (1 - 2 * cosfonfs + cosfonfs ** 2) + mult_sp * (1 - cosfonfs))
    return S_inst


# @njit()
def instrument_noise_AET_wdm_loop(phif: npt.NDArray[np.float64], lc: LISAConstants, wc: WDMWaveletConstants):
    """Helper to get the instrument noise for wdm"""
    # realistically this really only needs run once and is fast enough without jit
    # TODO check normalization
    # TODO get first and last bins correct
    nrm: float = np.sqrt(12318. / wc.Nf) * np.linalg.norm(phif)
    print('nrm instrument', nrm)
    phif /= nrm
    phif2 = phif**2

    S_inst_m = np.zeros((wc.Nf, wc.NC))
    half_Nt = wc.Nt // 2
    fs_long = np.arange(-half_Nt, half_Nt + wc.Nf * half_Nt) / wc.Tobs
    # prevent division by 0
    fs_long[half_Nt] = fs_long[half_Nt + 1]
    S_inst_long = instrument_noise_AET(fs_long, lc, wc)
    # excise the f=0 point
    S_inst_long[half_Nt, :] = 0.
    # apply window in loop
    for m in range(wc.Nf):
        S_inst_m[m] = np.dot(phif2, S_inst_long[m * half_Nt:(m + 2) * half_Nt])

    return S_inst_m


# @njit()
def instrument_noise_AET_wdm_m(lc: LISAConstants, wc: WDMWaveletConstants):
    """Get the instrument noise curve as a function of frequency for the wdm
    wavelet decomposition

    Parameters
    ----------
    lc : namedtuple
        constants for LISA constellation specified in lisa_config.py
    wc : namedtuple
        constants for WDM wavelet basis also from wdm_config.py

    Returns
    -------
    S_stat_m : numpy.ndarray (Nf x NC)
        array of the instrument noise curve for each TDI channel
        array shape is (freq. layers x number of TDI channels)
    """
    # TODO why no plus 1?
    ls = np.arange(-wc.Nt // 2, wc.Nt // 2)
    fs = ls / wc.Tobs
    phif = np.sqrt(wc.dt) * phitilde_vec(2 * np.pi * fs * wc.dt, wc.Nf, wc.nx)

    # TODO check ad hoc normalization factor
    S_inst_m = instrument_noise_AET_wdm_loop(phif, lc, wc)
    return S_inst_m


@njit()
def get_sparse_snr_helper(wavelet_waveform, nt_min, nt_max, wc: WDMWaveletConstants, inv_chol_S):
    """Calculates the S/N ratio for each TDI channel for a given waveform.

    Parameters
    ----------
    wavelet_waveform: namedtuple SparseTaylorTimeWaveform
        a sparse wavelet domain waveform
    nt_min
    nt_max
    wc : namedtuple
        constants for WDM wavelet basis also from wdm_config.py
    inv_chol_S :


    Returns
    -------

    """
    NUs = wavelet_waveform.N_set
    lists_pixels = wavelet_waveform.pixel_index
    wavelet_data = wavelet_waveform.wave_value
    if nt_max == -1:
        nt_max = wc.Nt
    snr2s = np.zeros(wc.NC)
    for itrc in range(wc.NC):
        i_itrs = np.mod(lists_pixels[itrc, 0:NUs[itrc]], wc.Nf).astype(np.int64)
        j_itrs = (lists_pixels[itrc, 0:NUs[itrc]] - i_itrs) // wc.Nf
        for mm in range(NUs[itrc]):
            if nt_min <= j_itrs[mm] < nt_max:
                mult = inv_chol_S[j_itrs[mm], i_itrs[mm], itrc] * wavelet_data[itrc, mm]
                snr2s[itrc] += mult * mult
    return np.sqrt(snr2s)


# @jitclass([('prune', nb.b1), ('S', nb.float64[:, :, :]), ('inv_S', nb.float64[:, :, :]), ('inv_chol_S', nb.float64[:, :, :]), ('chol_S', nb.float64[:, :, :])])
class DiagonalNonstationaryDenseNoiseModel(DenseNoiseModel):
    """a class to handle the fully diagonal nonstationary
    instrument noise model to feed to snr and fisher matrix calculations
    """

    def __init__(self, S, wc: WDMWaveletConstants, prune, seed=-1) -> None:
        """Initialize the fully diagonal, nonstationary noise model

        Parameters
        ----------
        S : numpy.ndarray
            array of dense noise curves for each TDI channel
            shape: (Nt x Nf x NC)=(freq layers x number of TDI channels)
        wc : namedtuple
            constants for WDM wavelet basis also from wdm_config.py
        prune : bool
            if prune=True, cut the 1st and last values,
            which may not be calculated correctly
        seed : int
            non-negative integer random seed for the noise generator;
            if set, generate_dense_noise will always produce the same result
            if -1, generate_dense_noise will get a new seed every time

        Returns
        -------
        DiagonalNonstationaryDenseNoiseModel : class
        """
        self.prune = prune
        self.S = S
        self.wc = wc
        self.seed = seed
        if self.seed < -1:
            msg = 'random seed cannot be negative; use -1 to use a different seed each time'
            raise ValueError(msg)

        self.inv_S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.inv_chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        if self.prune:
            i_offset = 1
        else:
            i_offset = 0
        for j in range(self.wc.Nt):
            for itrc in range(self.wc.NC):
                self.chol_S[j, i_offset:, itrc] = np.sqrt(self.S[j, i_offset:, itrc])
                self.inv_chol_S[j, i_offset:, itrc] = 1. / self.chol_S[j, i_offset:, itrc]
                self.inv_S[j, i_offset:, itrc] = self.inv_chol_S[j, i_offset:, itrc] ** 2
        if self.prune:
            self.chol_S[:, 0, :] = 0.
            self.inv_chol_S[:, 0, :] = 0.
            self.inv_S[:, 0, :] = 0.

    def generate_dense_noise(self):
        """Generate random noise for full matrix

        Parameters
        ----------
        There are no parameters required

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, NC) Number of time pixels,
            Freq layers, Number of TDI channels. All specified by wdm_config.py
        """
        noise_res = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        if self.seed == -1:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed)

        for j in range(self.wc.Nt):
            noise_res[j, :, :] = rng.normal(0., 1., (self.wc.Nf, self.wc.NC)) * self.chol_S[j, :, :]
        return noise_res

    def get_sparse_snrs(self, wavelet_waveform, nt_min=0, nt_max=-1):
        """Get snr of waveform in each channel"""
        return get_sparse_snr_helper(wavelet_waveform, nt_min, nt_max, self.wc, self.inv_chol_S)

    def get_S_stat_m(self):
        """Get the mean noise covariance matrix as a function of time"""
        return np.mean(self.S, axis=0)

    def get_inv_chol_S(self):
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self.inv_chol_S

    def get_inv_S(self):
        """Get the inverse of the dense noise covariance matrix"""
        return self.inv_S

    def get_chol_S(self):
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        return self.chol_S

    def get_S(self):
        """Get the dense noise covariance matrix"""
        return self.chol_S



# @jitclass([('prune', nb.b1), ('S_stat_m', nb.float64[:, :]), ('inv_S_stat_m', nb.float64[:, :]), ('inv_chol_S_stat_m', nb.float64[:, :]), ('S', nb.float64[:, :, :]), ('inv_S', nb.float64[:, :, :]), ('inv_chol_S', nb.float64[:, :, :]), ('chol_S_stat_m', nb.float64[:, :]), ('chol_S', nb.float64[:, :, :]), ('seed', nb.int64)])
class DiagonalStationaryDenseNoiseModel(DenseNoiseModel):
    """a class to handle the a diagonal stationary
     noise model to feed to snr and fisher matrix calculations
    """

    def __init__(self, S_stat_m, wc: WDMWaveletConstants, prune, seed=-1) -> None:
        """Initialize the stationary instrument noise model

        Parameters
        ----------
        S_stat_m : numpy.ndarray
            array of stationary noise curves for each TDI channel,
            such as instrument noise output from instrument_noise_AET_wdm_m
            shape: (Nf x NC) freq layers x number of TDI channels
        wc : namedtuple
            constants for WDM wavelet basis also from wdm_config.py
        prune : bool
            if prune=True, cut the 1st and last values,
            which may not be calculated correctly
        seed : int
            non-negative integer random seed for the noise generator;
            if set, generate_dense_noise will always produce the same result
            if -1, generate_dense_noise will get a new seed every time

        Returns
        -------
        DiagonalStationaryDenseNoiseModel : class
        """
        self.prune = prune
        self.S_stat_m = S_stat_m
        self.wc = wc
        self.seed = seed
        if self.seed < -1:
            msg = 'random seed cannot be negative; use -1 to use a different seed each time'
            raise ValueError(msg)

        self.inv_S_stat_m = np.zeros((self.wc.Nf, self.wc.NC))
        self.inv_chol_S_stat_m = np.zeros((self.wc.Nf, self.wc.NC))
        self.chol_S_stat_m = np.zeros((self.wc.Nf, self.wc.NC))

        for m in range(self.wc.Nf):
            if self.prune and  m in (0, wc.Nf):
                # currently m iterator doesn't even go to Nf,
                # but if it did it would also need to be pruned
                continue
            for itrc in range(self.wc.NC):
                self.inv_S_stat_m[m, itrc] = 1. / self.S_stat_m[m, itrc]
                self.chol_S_stat_m[m, itrc] = np.sqrt(self.S_stat_m[m, itrc])
                self.inv_chol_S_stat_m[m, itrc] = 1. / self.chol_S_stat_m[m, itrc]

        self.S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.inv_S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.inv_chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        self.chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        for j in range(self.wc.Nt):
            for itrc in range(self.wc.NC):
                self.S[j, 1:, itrc] = self.S_stat_m[1:, itrc]
                self.inv_S[j, 1:, itrc] = self.inv_S_stat_m[1:, itrc]
                self.inv_chol_S[j, 1:, itrc] = self.inv_chol_S_stat_m[1:, itrc]
                self.chol_S[j, 1:, itrc] = self.chol_S_stat_m[1:, itrc]
                if j % 2 == 0:
                    # NOTE right now this just necessarily drops the highest frequency term, also note that lowest frequency term may be a bit different
                    self.S[j, 0, itrc] = self.S_stat_m[0, itrc]
                    self.inv_S[j, 0, itrc] = self.inv_S_stat_m[0, itrc]
                    self.inv_chol_S[j, 0, itrc] = self.inv_chol_S_stat_m[0, itrc]
                    self.chol_S[j, 0, itrc] = self.chol_S_stat_m[0, itrc]

    def generate_dense_noise(self):
        """Generate random noise for full matrix

        Parameters
        ----------
        There are no parameters required

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, NC) Number of time pixels,
            Freq layers, Number of TDI channels. All specified by wdm_config.py
        """
        noise_res = np.zeros((self.wc.Nt, self.wc.Nf, self.wc.NC))
        if self.seed == -1:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed)

        for j in range(self.wc.Nt):
            noise_res[j, :, :] = rng.normal(0., 1., (self.wc.Nf, self.wc.NC)) * self.chol_S[j, :, :]
        return noise_res

    def get_sparse_snrs(self, wavelet_waveform, nt_min=0, nt_max=-1):
        """Get s/n of waveform in each TDI channel. Parameters usually come from
        BinaryWaveletAmpFreqDT.get_unsorted_coeffs() from
        wavelet_detector_waveforms.

        Parameters
        ----------
        wavelet_waveform: namedtuple SparseTaylorTimeWaveform
            a sparse wavelet domain waveform
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
        return get_sparse_snr_helper(wavelet_waveform, nt_min, nt_max, self.wc, self.inv_chol_S)

    def get_S_stat_m(self):
        """Get the mean noise covariance matrix as a function of time"""
        return self.S_stat_m

    def get_inv_chol_S(self):
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self.inv_chol_S

    def get_inv_S(self):
        """Get the inverse of the dense noise covariance matrix"""
        return self.inv_S

    def get_chol_S(self):
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        return self.chol_S

    def get_S(self):
        """Get the dense noise covariance matrix"""
        return self.chol_S
