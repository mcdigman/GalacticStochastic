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
        n_pixel_loc: int = int(wavelet_waveform.n_set[itrc])
        pixel_index_loc: NDArray[np.integer] = wavelet_waveform.pixel_index[itrc, : n_pixel_loc]
        wave_value_loc: NDArray[np.floating] = wavelet_waveform.wave_value[itrc, :]
        i_itrs: NDArray[np.integer] = np.mod(pixel_index_loc, wc.Nf)
        j_itrs: NDArray[np.integer] = (pixel_index_loc - i_itrs) // wc.Nf
        for mm in range(n_pixel_loc):
            j_loc: int = j_itrs[mm]
            i_loc: int = i_itrs[mm]
            if nt_lim_snr.nx_min <= j_loc < nt_lim_snr.nx_max:
                mult: float = inv_chol_S[j_loc, i_loc, itrc] * wave_value_loc[mm]
                snr2s[itrc] += mult * mult
    return np.sqrt(snr2s)

# from numba.experimental import jitclass


class DenseNoiseModel(ABC):
    def __init__(self, wc: WDMWaveletConstants, *, prune: int, nc_snr: int, nc_noise: int, seed: int = -1, storage_mode: int = 0) -> None:
        self._wc: WDMWaveletConstants = wc
        self._prune: int = prune
        self._seed: int = seed
        self._nc_snr: int = nc_snr
        self._nc_noise: int = nc_noise
        self._storage_mode: int = storage_mode

        if self.prune < 0:
            msg = 'Prune must be postive'
            raise ValueError(msg)

        if self._storage_mode not in (0, 1):
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
    def get_S(self) -> NDArray[np.floating]:
        """Get the dense noise covariance matrix"""

    @property
    def prune(self) -> int:
        """Get the number of lowest frequency bins that are being pruned"""
        return self._prune

    @property
    def seed(self):
        """Get the random seed for generating noise realizations"""
        return self._seed

    def get_inv_S(self) -> NDArray[np.floating]:
        """Get the inverse of the dense noise covariance matrix"""
        res = self.get_inv_chol_S()**2
        res[:, :self.prune, :] = 0.
        return res

    def get_chol_S(self) -> NDArray[np.floating]:
        """Get the cholesky decomposition of the dense noise covariance matrix"""
        # make sure pruned bins are zero
        res: NDArray[np.floating] = np.sqrt(self.get_S())
        res[:, :self.prune, :] = 0.
        return res

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
        hf_noise.attrs['storage_mode'] = self._storage_mode

        hf_wc = hf_noise.create_group('wc')
        for key in self._wc._fields:
            hf_wc.attrs[key] = getattr(self._wc, key)

        if self._storage_mode == 0:
            pass
        if self._storage_mode == 1:
            _ = hf_noise.create_dataset('S', data=self.get_S(), compression='gzip')

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
        assert len(S.shape) == 3
        assert S.shape[0] == wc.Nt
        assert S.shape[1] == wc.Nf
        super().__init__(wc, prune=prune, nc_snr=nc_snr, nc_noise=int(S.shape[-1]), seed=seed, storage_mode=storage_mode)

        self._S: NDArray[np.floating] = S.copy()
        self._inv_chol_S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        for j in range(self._wc.Nt):
            for itrc in range(self._nc_noise):
                chol_S_loc = np.sqrt(self._S[j, self.prune:, itrc])
                self._inv_chol_S[j, self.prune:, itrc] = 1.0 / chol_S_loc

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'dense_noise_model', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        return super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

    @override
    def get_inv_chol_S(self) -> NDArray[np.floating]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self._inv_chol_S

    @override
    def get_S(self) -> NDArray[np.floating]:
        """Get the dense noise covariance matrix"""
        return self._S


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
            if prune=1, cut lowest and highest frequency frequency bins,
                which may not be calculated correctly
            if prune>1, cut up to m lowest frequency bins, plus the highest frequency bin
            if prune=0, try not to cut any bins, which may not work as expected currently
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
        assert len(S_stat_m.shape) == 2
        assert S_stat_m.shape[0] == wc.Nf
        super().__init__(wc, prune=prune, nc_snr=nc_snr, nc_noise=int(S_stat_m.shape[-1]), seed=seed, storage_mode=storage_mode)
        self._S_stat_m_in: NDArray[np.floating] = S_stat_m

        self._inv_chol_S_stat_m: NDArray[np.floating] = np.zeros((self._wc.Nf, self._nc_noise))

        for m in range(self._wc.Nf):
            if self.prune < m and m in (0, wc.Nf):
                # currently m iterator doesn't even go to Nf,
                # but if it did it would also need to be pruned
                continue
            for itrc in range(self._nc_noise):
                chol_S_stat_m_loc = float(np.sqrt(self._S_stat_m_in[m, itrc]))
                self._inv_chol_S_stat_m[m, itrc] = 1.0 / chol_S_stat_m_loc

        self._S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))
        self._inv_chol_S: NDArray[np.floating] = np.zeros((self._wc.Nt, self._wc.Nf, self._nc_noise))

        # in the current representation, the even m=0 and odd m=0 slots represent
        # the highest and lowest frequency modes respectively
        # and would have to be handled differently when pruning
        for j in range(self._wc.Nt):
            for itrc in range(self._nc_noise):
                self._S[j, 1:, itrc] = self._S_stat_m_in[1:, itrc]
                self._inv_chol_S[j, 1:, itrc] = self._inv_chol_S_stat_m[1:, itrc]

                if self.prune == 0 and j % 2 == 0:
                    # NOTE right now this just necessarily drops the highest frequency term
                    # also note that lowest frequency term may be a bit different
                    self._S[j, 0, itrc] = self._S_stat_m_in[0, itrc]
                    self._inv_chol_S[j, 0, itrc] = self._inv_chol_S_stat_m[0, itrc]

    @override
    def store_hdf5(self, hf_in: h5py.Group, *, group_name: str = 'dense_noise_model', group_mode: int = 0) -> h5py.Group:
        """Store attributes, configuration, and results to an hdf5 file."""
        hf_noise = super().store_hdf5(hf_in, group_name=group_name, group_mode=group_mode)

        # all other attributes can be derived from S_stat_m
        _ = hf_noise.create_dataset('S_stat_m_in', data=self._S_stat_m_in, compression='gzip')

        return hf_noise

    @override
    def get_inv_chol_S(self) -> NDArray[np.floating]:
        """Get the inverse cholesky decomposition of the dense noise covariance matrix"""
        return self._inv_chol_S

    @override
    def get_S(self) -> NDArray[np.floating]:
        """Get the dense noise covariance matrix"""
        return self._S
