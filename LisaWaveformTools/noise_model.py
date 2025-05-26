"""get the instrument noise profile"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

# import numba as nb
from numba import njit

from WaveletWaveforms.sparse_waveform_functions import SparseWaveletWaveform
from WaveletWaveforms.wdm_config import WDMWaveletConstants

# from numba.experimental import jitclass


class DenseNoiseModel(ABC):
    def __init__(self, wc: WDMWaveletConstants, prune, n_snr: int, n_noise: int, seed: int = -1) -> None:
        self.wc = wc
        self.prune = prune
        self.seed = seed
        self.nc_snr = n_snr
        self.nc_noise = n_noise

    @abstractmethod
    def generate_dense_noise(self) -> npt.NDArray[np.float64]:
        """Generate random noise for full matrix"""

    @abstractmethod
    def get_sparse_snrs(self, wavelet_waveform: SparseWaveletWaveform, nt_min=0, nt_max=-1) -> npt.NDArray[np.float64]:
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

    def get_nc_snr(self) -> int:
        """Get the number of S/N channels"""
        return self.nc_snr

    def get_nc_noise(self) -> int:
        """Get the number of noise channels"""
        return self.nc_noise


@njit()
def get_sparse_snr_helper(
    wavelet_waveform: SparseWaveletWaveform,
    nt_min,
    nt_max,
    wc: WDMWaveletConstants,
    inv_chol_S: npt.NDArray[float],
    nc_snr,
) -> npt.NDArray[float]:
    """Calculates the S/N ratio for each TDI channel for a given waveform.

    Parameters
    ----------
    wavelet_waveform: SparseWaveletWaveform
        a sparse wavelet domain waveform
    nt_min
    nt_max
    wc : namedtuple
        constants for WDM wavelet basis also from wdm_config.py
    inv_chol_S :


    Returns
    -------

    """
    if nt_max == -1:
        nt_max = wc.Nt
    snr2s = np.zeros(nc_snr)
    for itrc in range(nc_snr):
        i_itrs = np.mod(wavelet_waveform.pixel_index[itrc, : wavelet_waveform.n_set[itrc]], wc.Nf).astype(np.int64)
        j_itrs = (wavelet_waveform.pixel_index[itrc, : wavelet_waveform.n_set[itrc]] - i_itrs) // wc.Nf
        for mm in range(wavelet_waveform.n_set[itrc]):
            if nt_min <= j_itrs[mm] < nt_max:
                mult = inv_chol_S[j_itrs[mm], i_itrs[mm], itrc] * wavelet_waveform.wave_value[itrc, mm]
                snr2s[itrc] += mult * mult
    return np.sqrt(snr2s)


# @jitclass(
#    [
#        ('prune', nb.b1),
#        ('S', nb.float64[:, :, :]),
#        ('inv_S', nb.float64[:, :, :]),
#        ('inv_chol_S', nb.float64[:, :, :]),
#        ('chol_S', nb.float64[:, :, :]),
#    ]
# )
class DiagonalNonstationaryDenseNoiseModel(DenseNoiseModel):
    """a class to handle the fully diagonal nonstationary
    instrument noise model to feed to snr and fisher matrix calculations
    """

    def __init__(self, S: npt.NDArray[float], wc: WDMWaveletConstants, prune, nc_snr: int, seed=-1) -> None:
        """Initialize the fully diagonal, nonstationary noise model

        Parameters
        ----------
        S : numpy.ndarray
            array of dense noise curves for each TDI channel
            shape: (Nt x Nf x nc_noise)=(freq layers x number of TDI channels)
        wc : namedtuple
            constants for WDM wavelet basis also from wdm_config.py
        prune : bool
            if prune=True, cut the 1st and last values,
            which may not be calculated correctly
        seed : int
            non-negative integer random seed for the noise generator;
            if set, generate_dense_noise will always produce the same result
            if -1, generate_dense_noise will get a new seed every time
        nc_snr : int
            number of TDI channels to calculate S/N for
            (should be less than or equal to the number of TDI channels in S)

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

        self.nc_snr = nc_snr
        self.nc_noise = self.S.shape[-1]
        assert self.nc_snr <= self.nc_noise, (
            'number of TDI channels to calculate S/N for must be less than or equal to the number of TDI channels in S'
        )

        self.inv_S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        self.inv_chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        self.chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        if self.prune:
            i_offset = 1
        else:
            i_offset = 0
        for j in range(self.wc.Nt):
            for itrc in range(self.nc_noise):
                self.chol_S[j, i_offset:, itrc] = np.sqrt(self.S[j, i_offset:, itrc])
                self.inv_chol_S[j, i_offset:, itrc] = 1.0 / self.chol_S[j, i_offset:, itrc]
                self.inv_S[j, i_offset:, itrc] = self.inv_chol_S[j, i_offset:, itrc] ** 2
        if self.prune:
            self.chol_S[:, 0, :] = 0.0
            self.inv_chol_S[:, 0, :] = 0.0
            self.inv_S[:, 0, :] = 0.0

    def generate_dense_noise(self) -> npt.NDArray[np.float64]:
        """Generate random noise for full matrix

        Parameters
        ----------
        There are no parameters required

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, nc_noise) Number of time pixels,
            Freq pixels, Number of TDI channels.
            number of pixels specified by wdm_config.py
        """
        noise_res = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        if self.seed == -1:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed)

        for j in range(self.wc.Nt):
            noise_res[j, :, :] = rng.normal(0.0, 1.0, (self.wc.Nf, self.nc_noise)) * self.chol_S[j, :, :]
        return noise_res

    def get_sparse_snrs(self, wavelet_waveform: SparseWaveletWaveform, nt_min=0, nt_max=-1) -> npt.NDArray[np.float64]:
        """Get snr of waveform in each channel"""
        return get_sparse_snr_helper(wavelet_waveform, nt_min, nt_max, self.wc, self.inv_chol_S, self.nc_snr)

    def get_S_stat_m(self) -> npt.NDArray[np.float64]:
        """Get the mean noise covariance matrix as a function of time"""
        return np.mean(self.S, axis=0)

    def get_inv_chol_S(self) -> npt.NDArray[np.float64]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self.inv_chol_S

    def get_inv_S(self) -> npt.NDArray[np.float64]:
        """Get the inverse of the dense noise covariance matrix"""
        return self.inv_S

    def get_chol_S(self) -> npt.NDArray[np.float64]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        return self.chol_S

    def get_S(self) -> npt.NDArray[np.float64]:
        """Get the dense noise covariance matrix"""
        return self.chol_S


# @jitclass(
#    [
#        ('prune', nb.b1),
#        ('S_stat_m', nb.float64[:, :]),
#        ('inv_S_stat_m', nb.float64[:, :]),
#        ('inv_chol_S_stat_m', nb.float64[:, :]),
#        ('S', nb.float64[:, :, :]),
#        ('inv_S', nb.float64[:, :, :]),
#        ('inv_chol_S', nb.float64[:, :, :]),
#        ('chol_S_stat_m', nb.float64[:, :]),
#        ('chol_S', nb.float64[:, :, :]),
#        ('seed', nb.int64),
#    ]
# )
class DiagonalStationaryDenseNoiseModel(DenseNoiseModel):
    """a class to handle the a diagonal stationary
    noise model to feed to snr and fisher matrix calculations
    """

    def __init__(self, S_stat_m, wc: WDMWaveletConstants, prune, nc_snr: int, seed=-1) -> None:
        """Initialize the stationary instrument noise model

        Parameters
        ----------
        S_stat_m : numpy.ndarray
            array of stationary noise curves for each TDI channel,
            such as instrument noise output from instrument_noise_AET_wdm_m
            shape: (Nf x nc_noise) freq layers x number of TDI channels
        wc : namedtuple
            constants for WDM wavelet basis also from wdm_config.py
        prune : bool
            if prune=True, cut the 1st and last values,
            which may not be calculated correctly
        nc_snr : int
            number of TDI channels to calculate S/N for
            (should be less than or equal to the number of TDI channels in S)
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

        self.nc_snr = nc_snr
        self.nc_noise = self.S_stat_m.shape[-1]

        assert self.nc_snr <= self.nc_noise, (
            'number of TDI channels to calculate S/N for must be less than or equal to the number of TDI channels in S'
        )

        self.inv_S_stat_m = np.zeros((self.wc.Nf, self.nc_noise))
        self.inv_chol_S_stat_m = np.zeros((self.wc.Nf, self.nc_noise))
        self.chol_S_stat_m = np.zeros((self.wc.Nf, self.nc_noise))

        for m in range(self.wc.Nf):
            if self.prune and m in (0, wc.Nf):
                # currently m iterator doesn't even go to Nf,
                # but if it did it would also need to be pruned
                continue
            for itrc in range(self.nc_noise):
                self.inv_S_stat_m[m, itrc] = 1.0 / self.S_stat_m[m, itrc]
                self.chol_S_stat_m[m, itrc] = np.sqrt(self.S_stat_m[m, itrc])
                self.inv_chol_S_stat_m[m, itrc] = 1.0 / self.chol_S_stat_m[m, itrc]

        self.S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        self.inv_S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        self.inv_chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        self.chol_S = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        for j in range(self.wc.Nt):
            for itrc in range(self.nc_noise):
                self.S[j, 1:, itrc] = self.S_stat_m[1:, itrc]
                self.inv_S[j, 1:, itrc] = self.inv_S_stat_m[1:, itrc]
                self.inv_chol_S[j, 1:, itrc] = self.inv_chol_S_stat_m[1:, itrc]
                self.chol_S[j, 1:, itrc] = self.chol_S_stat_m[1:, itrc]
                if j % 2 == 0:
                    # NOTE right now this just necessarily drops the highest frequency term
                    # also note that lowest frequency term may be a bit different
                    self.S[j, 0, itrc] = self.S_stat_m[0, itrc]
                    self.inv_S[j, 0, itrc] = self.inv_S_stat_m[0, itrc]
                    self.inv_chol_S[j, 0, itrc] = self.inv_chol_S_stat_m[0, itrc]
                    self.chol_S[j, 0, itrc] = self.chol_S_stat_m[0, itrc]

    def generate_dense_noise(self) -> npt.NDArray[np.float64]:
        """Generate random noise for full matrix

        Parameters
        ----------
        There are no parameters required

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, nc_noise) Number of time pixels,
            Freq pixels, Number of TDI channels.
            Pixel dimensions specified by wdm_config.py
        """
        noise_res = np.zeros((self.wc.Nt, self.wc.Nf, self.nc_noise))
        if self.seed == -1:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed)

        for j in range(self.wc.Nt):
            noise_res[j, :, :] = rng.normal(0.0, 1.0, (self.wc.Nf, self.nc_noise)) * self.chol_S[j, :, :]
        return noise_res

    def get_sparse_snrs(self, wavelet_waveform: SparseWaveletWaveform, nt_min=0, nt_max=-1) -> npt.NDArray[float]:
        """Get s/n of waveform in each TDI channel. Parameters usually come from
        BinaryWaveletAmpFreqDT.get_unsorted_coeffs() from
        wavelet_detector_waveforms.

        Parameters
        ----------
        wavelet_waveform: namedtuple SparseWaveletWaveform
            a sparse wavelet domain waveform
        nt_min : int, default=0
            time pixels that are start/end of slice for evaluating.
            Used for selecting a subset of time pixels
        nt_max : int, default=-1

        Returns
        -------
        snr : numpy.ndarray
            an array of shape (nc_noise) which is the S/N for each TDI channel represented.
        """
        if nt_max == -1:
            nt_max = self.wc.Nt
        return get_sparse_snr_helper(wavelet_waveform, nt_min, nt_max, self.wc, self.inv_chol_S, self.nc_snr)

    def get_S_stat_m(self) -> npt.NDArray[np.float64]:
        """Get the mean noise covariance matrix as a function of time"""
        return self.S_stat_m

    def get_inv_chol_S(self) -> npt.NDArray[np.float64]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self.inv_chol_S

    def get_inv_S(self) -> npt.NDArray[np.float64]:
        """Get the inverse of the dense noise covariance matrix"""
        return self.inv_S

    def get_chol_S(self) -> npt.NDArray[np.float64]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        return self.chol_S

    def get_S(self) -> npt.NDArray[np.float64]:
        """Get the dense noise covariance matrix"""
        return self.chol_S
