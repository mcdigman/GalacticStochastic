"""helper functions for Chirp_WDM"""

from typing import NamedTuple

import numpy as np
import WDMWaveletTransforms.fft_funcs as fft
from numba import njit
from numpy.typing import NDArray

from LisaWaveformTools.chirplet_source_time import LinearChirpletSourceWaveformTime
from LisaWaveformTools.lisa_config import LISAConstants
from LisaWaveformTools.stationary_source_waveform import SourceParams, StationaryWaveformTime
from LisaWaveformTools.stationary_time_source import StationarySourceWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_time_coefficients import wavelet
from WaveletWaveforms.wdm_config import WDMWaveletConstants


class SparseCoefficientTable(NamedTuple):
    cM: NDArray[np.complexfloating]
    sM: NDArray[np.complexfloating]


def _get_sparse_t_grid(wc: WDMWaveletConstants) -> PixelGenericRange:
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin = int(wc.K // wc.L)
    assert wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thinning factor'

    M = int(wc.Nf * wc.Nt / sparse_thin)
    print(wc.Nf / sparse_thin)

    dtd = float(wc.dt * sparse_thin)  # downsampled data spacing in time
    # TODO handle t0
    return PixelGenericRange(0, M, dtd, 0.)


@njit()
def _get_sparse_m_pixel_helper(waveform: StationaryWaveformTime, wc: WDMWaveletConstants) -> NDArray[np.integer]:
    assert wc.K % wc.L == 0, 'K currently needs to be an integer multiple of L'
    sparse_thin = int(wc.K // wc.L)
    assert wc.Nf % sparse_thin == 0, 'Nf currently needs to be an integer multiple of sparse thinning factor'

    thin_factor = int(wc.Nf // sparse_thin)

    return (waveform.FT[:, ::thin_factor] / wc.DF).astype(np.int64)


def _update_bounds_helper(m_pixel, wc: WDMWaveletConstants) -> PixelGenericRange:
    # currently assumes frequency is generally increasing
    nt_mins = (np.argmax(m_pixel > -1, axis=-1)).astype(np.int64)
    nt_maxs = (wc.Nt - np.argmax(m_pixel[:, ::-1] < wc.Nf, axis=-1)).astype(np.int64)
    return PixelGenericRange(np.max(nt_mins), np.min(nt_maxs), wc.DT, 0.)


# @njit()
def _sparse_time_DX_assign_loop(m_pixel: NDArray[np.integer], nt_lim: PixelGenericRange, waveform: StationaryWaveformTime, sparse_table: SparseCoefficientTable, wc: WDMWaveletConstants) -> NDArray[np.complexfloating]:
    """Helper to start loop for sparse_wavelet_time"""
    nc_waveform = m_pixel.shape[0]
    assert wc.L % 2 == 0

    half_L = int(wc.L // 2)

    p = float(wc.K / wc.L)
    M = int(wc.Nf * wc.Nt / p)

    DX = np.zeros((nc_waveform, nt_lim.nx_max - nt_lim.nx_min, wc.L), dtype=np.complex128)

    nn = int((wc.Nf * wc.L) / wc.K)
    AT = waveform.AT
    PT = waveform.PT

    for itrc in range(nc_waveform):
        cP = AT[itrc] * np.cos(PT[itrc])
        sP = AT[itrc] * np.sin(PT[itrc])
        for n in range(nt_lim.nx_min, nt_lim.nx_max):
            n_itr = n - nt_lim.nx_min
            ii = nn * n
            mc = m_pixel[itrc, n]
            i_min = max(half_L - ii, 0)
            i_max = min(wc.L, M + half_L - ii)
            for i in range(i_min, i_max):
                k = i + ii - half_L
                DX[itrc, n_itr, i] = cP[k] * sparse_table.cM[mc][i] + sP[k] * sparse_table.sM[mc][i]

            if i_min > 0:
                DX[itrc, n_itr, 0:i_min] = 0.0
            if i_max < wc.L:
                DX[itrc, n_itr, i_max:] = 0.0
    return DX


# @njit()
def _sparse_time_DX_unpack_loop(m_pixel: NDArray[np.integer], nt_lim: PixelGenericRange, DX_trans: NDArray[np.complexfloating], wc: WDMWaveletConstants) -> tuple[NDArray[np.int64], NDArray[np.floating], NDArray[np.integer], int]:
    """Helper to start unpack fft results for sparse_wavelet_time"""
    nc_waveform = m_pixel.shape[0]

    # indicates this pixel not used
    p = wc.K / wc.L
    kx = int(2 * wc.Nf / p)
    half_kx = int(kx // 2)
    N_max = kx * wc.Nt
    waveT = np.zeros((nc_waveform, N_max))
    Tlist = np.full((nc_waveform, N_max), -1, dtype=np.int64)
    N_set = np.zeros(nc_waveform, dtype=np.int64)

    for itrc in range(nc_waveform):
        mm = 0
        for n in range(nt_lim.nx_min, nt_lim.nx_max):
            n_itr = n - nt_lim.nx_min
            mc = m_pixel[itrc, n]
            # negative frequencies
            for j in range(half_kx, kx):
                m = j - kx + mc
                if -1 < m < wc.Nf:
                    if (n + m) % 2 == 0:
                        waveT[itrc, mm] = np.real(DX_trans[itrc, n_itr, j * wc.mult])
                    else:
                        waveT[itrc, mm] = -np.imag(DX_trans[itrc, n_itr, j * wc.mult])
                    Tlist[itrc, mm] = n * wc.Nf + m
                    mm += 1

            # postive frequencies
            for j in range(half_kx):
                m = j + mc
                if -1 < m < wc.Nf:
                    if (n + m) % 2 == 0:
                        waveT[itrc, mm] = np.real(DX_trans[itrc, n_itr, j * wc.mult])
                    else:
                        waveT[itrc, mm] = -np.imag(DX_trans[itrc, n_itr, j * wc.mult])
                    Tlist[itrc, mm] = n * wc.Nf + m
                    mm += 1
            N_set[itrc] = mm
    return Tlist, waveT, N_set, N_max


def sparse_wavelet_time(wave: StationarySourceWaveformTime, sparse_table: SparseCoefficientTable, wc: WDMWaveletConstants) -> SparseWaveletWaveform:
    """Calculate time domain sparse wavelet method"""
    # have sped this up by computing cos(Phase[k]), sin(Phase[k])
    # and passing in pre-computed arrays for cos(2.*np.pi*(double)(j*mq)/(double)(L))
    # and sin(2.*np.pi*(double)(j*mq)/(double)(L)). The array is L*Nf in size

    m_pixel = _get_sparse_m_pixel_helper(wave.tdi_waveform, wc)
    nt_lim_restrict = _update_bounds_helper(m_pixel, wc)
    DX = _sparse_time_DX_assign_loop(m_pixel, nt_lim_restrict, wave.tdi_waveform, sparse_table, wc)
    DX_trans: NDArray[np.complexfloating] = fft.fft(DX, wc.L)
    del DX
    Tlist, waveT, n_set, N_max = _sparse_time_DX_unpack_loop(m_pixel, nt_lim_restrict, DX_trans, wc)
    return SparseWaveletWaveform(waveT, Tlist, n_set, N_max)


def _sparse_table_helper(wc: WDMWaveletConstants) -> SparseCoefficientTable:
    # TODO does this function already exist somewhere else?
    assert wc.L % 2 == 0
    half_L = int(wc.L // 2)
    wave = wavelet(wc, 0, 1.0, n_in=wc.L)

    p = wc.K / wc.L
    nrm = np.sqrt(2.0 / p) * np.linalg.norm(wave)
    wave /= nrm

    # pre-computed phase coefficients
    x = np.outer(np.arange(0, wc.Nf), (2.0 * np.pi * wc.mult / wc.L) * np.arange(-half_L, half_L))
    cH = wave * np.cos(x)
    sH = wave * np.sin(x)

    del wave
    del x

    cM = cH - 1j * sH
    sM = sH + 1j * cH
    return SparseCoefficientTable(cM, sM)


def wavelet_SparseT(params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants) -> SparseWaveletWaveform:
    """Compute the time domain wavelet filter and normalize"""
    sparse_table = _sparse_table_helper(wc)
    nt_lim_grid = _get_sparse_t_grid(wc)
    wave = LinearChirpletSourceWaveformTime(params, nt_lim_grid, lc, response_mode=2)

    return sparse_wavelet_time(wave, sparse_table, wc)


# class BinaryWaveletSparseTime(SparseWaveletSourceWaveform[StationaryWaveformTime]):
#    """Store a sparse binary wavelet for a time domain taylor intrinsic_waveform."""
#
#    def __init__(self, params: SourceParams, wc: WDMWaveletConstants, lc: LISAConstants, nt_lim_waveform: PixelGenericRange, source_waveform: StationarySourceWaveform[StationaryWaveformTime]) -> None:
#        """Construct a sparse binary wavelet for a time domain taylor intrinsic_waveform with interpolation."""
#        self._wc: WDMWaveletConstants = wc
#        self._lc: LISAConstants = lc
#        self._nt_lim_waveform: PixelGenericRange = nt_lim_waveform
#
#        # store the intrinsic_waveform
#        self._source_waveform: StationarySourceWaveform[StationaryWaveformTime] = source_waveform
#
#        # get a blank wavelet intrinsic_waveform with the correct size for the sparse taylor time method
#        # when consistent is set to True, it will be the correct intrinsic_waveform
#        wavelet_waveform_loc: SparseWaveletWaveform = get_empty_sparse_sparse_time_waveform(int(self._lc.nc_waveform), wc)
#
#        # interpolation for wavelet taylor expansion
#        self._sparse_table: SparseCoefficientTable = _sparse_table_helper(self._wc)
#
#        super().__init__(params, wavelet_waveform_loc, source_waveform)
#
#    @override
#    def _update_wavelet_waveform(self) -> None:
#        """Update the wavelet intrinsic_waveform to match the current parameters."""
#        self._wavelet_waveform = sparse_wavelet_time(
#            self._wavelet_waveform,
#            self.source_waveform.tdi_waveform,
#            self._nt_lim_waveform,
#            self._wc,
#            self._taylor_time_table,
#        )
