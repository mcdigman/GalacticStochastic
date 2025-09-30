"""get the instrument noise profile"""

from abc import ABC, abstractmethod
from typing import override

import h5py
import numpy as np

# import numba as nb
from numba import njit
from numpy.typing import NDArray

from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.wdm_config import WDMWaveletConstants


@njit()
def get_sparse_snr_helper(
    wavelet_waveform: SparseWaveletWaveform,
    nt_lim_snr: PixelGenericRange,
    wc: WDMWaveletConstants,
    inv_chol_S: NDArray[np.floating],
    nc_snr: int,
) -> NDArray[np.floating]:
    """Calculates the S/N ratio for each TDI channel for a given intrinsic_waveform.

    Parameters
    ----------
    wavelet_waveform: SparseWaveletWaveform
        a sparse wavelet domain intrinsic_waveform
    nt_lim_snr: PixelGenericRange
        the range of time pixels to allow
    wc : WDMWaveletConstants
        constants for WDM wavelet basis also from wdm_config.py
    inv_chol_S :


    Returns
    -------

    """
    snr2s = np.zeros(nc_snr)
    for itrc in range(nc_snr):
        i_itrs = np.mod(wavelet_waveform.pixel_index[itrc, : wavelet_waveform.n_set[itrc]], wc.Nf).astype(np.integer)
        j_itrs = (wavelet_waveform.pixel_index[itrc, : wavelet_waveform.n_set[itrc]] - i_itrs) // wc.Nf
        for mm in range(wavelet_waveform.n_set[itrc]):
            if nt_lim_snr.nx_min <= j_itrs[mm] < nt_lim_snr.nx_max:
                mult = inv_chol_S[j_itrs[mm], i_itrs[mm], itrc] * wavelet_waveform.wave_value[itrc, mm]
                snr2s[itrc] += mult * mult
    return np.sqrt(snr2s)

# from numba.experimental import jitclass


class DenseNoiseModel(ABC):
    def __init__(self, wc: WDMWaveletConstants, prune: int, nc_snr: int, nc_noise: int, seed: int = -1, storage_mode: int = 0) -> None:
        self._wc: WDMWaveletConstants = wc
        self.prune: int = prune
        self.seed: int = seed
        self._nc_snr: int = nc_snr
        self._nc_noise: int = nc_noise
        self.storage_mode: int = storage_mode

        if self.storage_mode not in (0, 1):
            msg = 'Unrecognized option for storage mode'
            raise NotImplementedError(msg)

        if self.seed < -1:
            msg = 'random seed cannot be negative; use -1 to use a different seed each time'
            raise ValueError(msg)

        assert self._nc_snr <= self._nc_noise, (
            'number of TDI channels to calculate S/N for must be less than or equal to the number of TDI channels in S'
        )

    @abstractmethod
    def get_inv_chol_S(self) -> NDArray[np.floating]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""

    @abstractmethod
    def get_chol_S(self) -> NDArray[np.floating]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""

    @abstractmethod
    def get_S(self) -> NDArray[np.floating]:
        """Get the dense noise covariance matrix"""

    def get_inv_S(self) -> NDArray[np.floating]:
        """Get the inverse of the dense noise covariance matrix"""
        return self.get_inv_chol_S()**2

    def get_sparse_snrs(self, wavelet_waveform: SparseWaveletWaveform, nt_lim_snr: PixelGenericRange) -> NDArray[np.floating]:
        """Get s/n of intrinsic_waveform in each TDI channel. Parameters usually come from
        LinearFrequencyWaveletWaveformTime.get_unsorted_coeffs() from
        wavelet_detector_waveforms.

        Parameters
        ----------
        wavelet_waveform: SparseWaveletWaveform
            a sparse wavelet domain intrinsic_waveform
        nt_lim_snr: PixelGenericRange
            the range of time pixels to consider for snr calculations

        Returns
        -------
        snr : numpy.ndarray
            an array of shape (nc_noise) which is the S/N for each TDI channel represented.

        """
        return get_sparse_snr_helper(wavelet_waveform, nt_lim_snr, self._wc, self.get_inv_chol_S(), self._nc_snr)

    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'dense_noise_model', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        if group_mode == 0:
            hf_noise = hf_in.create_group(group_name)
        elif group_mode == 1:
            hf_noise = hf_in
        else:
            msg = 'Unrecognized option for group mode'
            raise NotImplementedError(msg)

        hf_noise.attrs['creator_name'] = self.__class__.__name__
        hf_noise.attrs['prune'] = self.prune
        hf_noise.attrs['seed'] = self.seed
        hf_noise.attrs['nc_snr'] = self.get_nc_snr()
        hf_noise.attrs['nc_noise'] = self.get_nc_noise()
        hf_noise.attrs['wc_name'] = self._wc.__class__.__name__
        hf_noise.attrs['storage_mode'] = self.storage_mode

        hf_wc = hf_noise.create_group('wc')
        for key in self._wc._fields:
            hf_wc.attrs[key] = getattr(self._wc, key)

        if self.storage_mode == 0:
            pass
        if self.storage_mode == 1:
            hf_noise.create_dataset('S', data=self.get_S(), compression='gzip')

        return hf_noise

    def get_nc_snr(self) -> int:
        """Get the number of S/N channels"""
        return self._nc_snr

    def get_nc_noise(self) -> int:
        """Get the number of noise channels"""
        return self._nc_noise

    def generate_dense_noise(self, white_mode: int = 0) -> NDArray[np.floating]:
        """Generate random noise for full matrix

        Parameters
        ----------
        white_mode: int
            if white_mode=0, return the instrument noise unwhitened
            if white_mode=1, return the instrument noise completely whitened

        Returns
        -------
        noise_res : numpy.ndarray
            noise matrix of shape (Nt, Nf, nc_noise) Number of time pixels,
            Freq pixels, Number of TDI channels.
            Pixel dimensions specified by wdm_config.py

        """
        if white_mode not in (0, 1):
            msg = 'Unrecognized option for white_mode'
            raise ValueError(msg)
        noise_res = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        if self.seed == -1:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed)

        if white_mode == 0:
            chol_S = self.get_chol_S()
            for j in range(self._wc.Nt):
                noise_res[j, :, :] = rng.normal(0.0, 1.0, (self._wc.Nf, self._nc_noise)) * chol_S[j, :, :]
        if white_mode == 1:
            for j in range(self._wc.Nt):
                noise_res[j, :, :] = rng.normal(0.0, 1.0, (self._wc.Nf, self._nc_noise))
        return noise_res

    def get_S_stat_m(self) -> NDArray[np.floating]:
        """Get the mean noise covariance matrix as a function of time"""
        return np.mean(self.get_S(), axis=0)


class DiagonalNonstationaryDenseNoiseModel(DenseNoiseModel):
    """a class to handle the fully diagonal nonstationary
    instrument noise model to feed to snr and fisher matrix calculations
    """

    def __init__(self, S: NDArray[np.floating], wc: WDMWaveletConstants, prune: int, nc_snr: int, seed: int = -1, storage_mode: int = 0) -> None:
        """Initialize the fully diagonal, nonstationary noise model

        Parameters
        ----------
        S : numpy.ndarray
            array of dense noise curves for each TDI channel
            shape: (Nt x Nf x nc_noise)=(freq layers x number of TDI channels)
        wc : WDMWaveletConstants
            constants for WDM wavelet basis also from wdm_config.py
        prune : int
            if prune=1, cut the 1st and last values,
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
        super().__init__(wc, prune, nc_snr, int(S.shape[-1]), seed, storage_mode)

        self._S: NDArray[np.floating] = S.copy()
        self._inv_chol_S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        self._chol_S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        if self.prune:
            i_offset = 1
        else:
            i_offset = 0
        for j in range(self._wc.Nt):
            for itrc in range(self._nc_noise):
                self._chol_S[j, i_offset:, itrc] = np.sqrt(self._S[j, i_offset:, itrc])
                self._inv_chol_S[j, i_offset:, itrc] = 1.0 / self._chol_S[j, i_offset:, itrc]
        if self.prune:
            self._chol_S[:, 0, :] = 0.0
            self._inv_chol_S[:, 0, :] = 0.0

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'dense_noise_model', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        return super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

    @override
    def get_inv_chol_S(self) -> NDArray[np.floating]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self._inv_chol_S

    @override
    def get_chol_S(self) -> NDArray[np.floating]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        return self._chol_S

    @override
    def get_S(self) -> NDArray[np.floating]:
        """Get the dense noise covariance matrix"""
        return self._chol_S * self._chol_S


class DiagonalStationaryDenseNoiseModel(DenseNoiseModel):
    """a class to handle the a diagonal stationary
    noise model to feed to snr and fisher matrix calculations
    """

    def __init__(self, S_stat_m: NDArray[np.floating], wc: WDMWaveletConstants, prune: int, nc_snr: int, seed: int = -1, storage_mode: int = 0) -> None:
        """Initialize the stationary instrument noise model

        Parameters
        ----------
        S_stat_m : numpy.ndarray
            array of stationary noise curves for each TDI channel,
            such as instrument noise output from instrument_noise_AET_wdm_m
            shape: (Nf x nc_noise) freq layers x number of TDI channels
        wc : WDMWaveletConstants
            constants for WDM wavelet basis also from wdm_config.py
        prune : int
            if prune=1, cut the 1st and last values,
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
        super().__init__(wc, prune, nc_snr, int(S_stat_m.shape[-1]), seed, storage_mode=storage_mode)
        self._S_stat_m_in: NDArray[np.floating] = S_stat_m

        self._inv_chol_S_stat_m: NDArray[np.floating] = np.zeros((self._wc.Nf, self._nc_noise))
        self._chol_S_stat_m: NDArray[np.floating] = np.zeros((self._wc.Nf, self._nc_noise))

        for m in range(self._wc.Nf):
            if self.prune == 1 and m in (0, wc.Nf):
                # currently m iterator doesn't even go to Nf,
                # but if it did it would also need to be pruned
                continue
            for itrc in range(self._nc_noise):
                self._chol_S_stat_m[m, itrc] = np.sqrt(self._S_stat_m_in[m, itrc])
                self._inv_chol_S_stat_m[m, itrc] = 1.0 / self._chol_S_stat_m[m, itrc]

        self._S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        self._inv_chol_S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        self._chol_S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        for j in range(self._wc.Nt):
            for itrc in range(self._nc_noise):
                self._S[j, 1:, itrc] = self._S_stat_m_in[1:, itrc]
                self._inv_chol_S[j, 1:, itrc] = self._inv_chol_S_stat_m[1:, itrc]
                self._chol_S[j, 1:, itrc] = self._chol_S_stat_m[1:, itrc]
                if j % 2 == 0:
                    # NOTE right now this just necessarily drops the highest frequency term
                    # also note that lowest frequency term may be a bit different
                    self._S[j, 0, itrc] = self._S_stat_m_in[0, itrc]
                    self._inv_chol_S[j, 0, itrc] = self._inv_chol_S_stat_m[0, itrc]
                    self._chol_S[j, 0, itrc] = self._chol_S_stat_m[0, itrc]

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'dense_noise_model', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        hf_noise = super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

        # all other attributes can be derived from S_stat_m
        hf_noise.create_dataset('S_stat_m_in', data=self._S_stat_m_in, compression='gzip')

        return hf_noise

    @override
    def get_inv_chol_S(self) -> NDArray[np.floating]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self._inv_chol_S

    @override
    def get_chol_S(self) -> NDArray[np.floating]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        return self._chol_S

    @override
    def get_S(self) -> NDArray[np.floating]:
        """Get the dense noise covariance matrix"""
        return self._S
