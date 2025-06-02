"""helper functions for Chirp_WDM"""

import numpy as np
from numba import njit

from LisaWaveformTools.ra_waveform_time import StationaryWaveformTime
from WaveletWaveforms.coefficientsWDM_time_funcs import get_taylor_pixel_direct
from WaveletWaveforms.coefficientsWDM_time_helpers import WaveletTaylorTimeCoeffs
from WaveletWaveforms.sparse_waveform_functions import PixelTimeRange, SparseWaveletWaveform
from WaveletWaveforms.wdm_config import WDMWaveletConstants


@njit(fastmath=True)
def wavemaket_multi_inplace(
    wavelet_waveform: SparseWaveletWaveform,
    waveform: StationaryWaveformTime,
    nt_lim_waveform: PixelTimeRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
    *,
    force_nulls: int=0,
) -> None:
    """Compute the actual wavelets using taylor time method"""
    n_set_old = wavelet_waveform.n_set.copy()

    nc_waveform = wavelet_waveform.wave_value.shape[0]

    for itrc in range(nc_waveform):
        mm = 0
        nf_min = 0
        nf_max = wc.Nf - 1
        for j in range(nt_lim_waveform.nt_min, nt_lim_waveform.nt_max):
            # keep j_ind separate in case we want to apply a time offset in the future
            j_ind = j

            y0 = waveform.FTd[itrc, j] / wc.dfd
            ny = int(np.floor(y0))
            n_ind = ny + wc.Nfd_negative

            assert taylor_table.Nfsam.size == wc.Nfd
            if 0 <= n_ind < wc.Nfd - 1:
                c = waveform.AT[itrc, j] * np.cos(waveform.PT[itrc, j])
                s = waveform.AT[itrc, j] * np.sin(waveform.PT[itrc, j])

                dy = y0 - ny
                assert 0.0 <= dy <= 1.0
                fa = waveform.FT[itrc, j]
                za = fa / wc.df_bw

                Nfsam1_loc = taylor_table.Nfsam[n_ind]
                # assert Nfsam1_loc == int(wc.Nsf + 2 / 3 * np.abs(ny) * wc.dfdot * wc.Nsf)
                Nfsam2_loc = taylor_table.Nfsam[n_ind + 1]
                # assert Nfsam2_loc == int(wc.Nsf + 2 / 3 * np.abs(ny + 1) * wc.dfdot * wc.Nsf)
                HBW = (min(Nfsam1_loc, Nfsam2_loc) - 1) * wc.df_bw / 2

                # lowest frequency layer
                kmin = max(nf_min, int(np.ceil((fa - HBW) / wc.DF)))

                # highest frequency layer
                kmax = min(nf_max, int(np.floor((fa + HBW) / wc.DF)))

                for k in range(kmin, kmax + 1):
                    zmid = (wc.DF / wc.df_bw) * k

                    # we apparently need za - zmid to be positive
                    if za < zmid:
                        zmid = za - np.abs(za - zmid)

                    kk = np.floor(za - zmid - 0.5)
                    zsam = zmid + kk + 0.5
                    kk = int(kk)
                    dx = za - zsam  # used for linear interpolation
                    assert 0.0 <= dx <= 1.0

                    # interpolate over frequency
                    jj1 = kk + Nfsam1_loc // 2
                    jj2 = kk + Nfsam2_loc // 2

                    # prevent case where we would overflow the table
                    if (0 <= jj1 < Nfsam1_loc - 1) and (0 <= jj2 < Nfsam2_loc - 1):
                        wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k

                        assert taylor_table.evcs[n_ind, jj1] != 0.0
                        assert taylor_table.evcs[n_ind, jj1 + 1] != 0.0
                        assert taylor_table.evcs[n_ind + 1, jj2] != 0.0
                        assert taylor_table.evcs[n_ind + 1, jj2 + 1] != 0.0

                        y = (1.0 - dx) * taylor_table.evcs[n_ind, jj1] + dx * taylor_table.evcs[n_ind, jj1 + 1]
                        yy = (1.0 - dx) * taylor_table.evcs[n_ind + 1, jj2] + dx * taylor_table.evcs[n_ind + 1, jj2 + 1]

                        z = (1.0 - dx) * taylor_table.evss[n_ind, jj1] + dx * taylor_table.evss[n_ind, jj1 + 1]
                        zz = (1.0 - dx) * taylor_table.evss[n_ind + 1, jj2] + dx * taylor_table.evss[n_ind + 1, jj2 + 1]

                        # interpolate over fdot
                        y = (1.0 - dy) * y + dy * yy
                        z = (1.0 - dy) * z + dy * zz


                        if (j_ind + k) % 2:
                            wavelet_waveform.wave_value[itrc, mm] = -(c * z + s * y)
                        else:
                            wavelet_waveform.wave_value[itrc, mm] = c * y - s * z

                        mm += 1
                        # end loop over frequency layers
            elif force_nulls == 1:
                # we know what the indices would be for values not precomputed in the table
                # so force values outside the range of the table to 0 instead of dropping,
                # in order to get likelihoods right, which is particularly important around total nulls,
                # which can be quite constraining on the parameters but have large spikes in frequency derivative
                # note that if this happens only very rarely
                # we could just actually calculate the non-precomputed coefficient
                fa = waveform.FT[itrc, j]

                Nfsam1_loc = int((wc.BW + wc.dfd * wc.Tw * ny) / wc.df_bw)
                if Nfsam1_loc % 2 == 1:
                    Nfsam1_loc += 1

                HBW = (Nfsam1_loc - 1) * wc.df_bw / 2
                # lowest frequency layer
                kmin = max(0, int(np.ceil((fa - HBW) / wc.DF)))

                # highest frequency layer
                kmax = min(wc.Nf - 1, int(np.floor((fa + HBW) / wc.DF)))

                for k in range(kmin, kmax + 1):
                    wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k
                    wavelet_waveform.wave_value[itrc, mm] = 0.0
                    mm += 1
        wavelet_waveform.n_set[itrc] = mm

    # clean up any pixels that were set in the old waveform but aren't anymore
    for itrc in range(nc_waveform):
        for itrm in range(wavelet_waveform.n_set[itrc], n_set_old[itrc]):
            wavelet_waveform.pixel_index[itrc, itrm] = -1
            wavelet_waveform.wave_value[itrc, itrm] = 0.0

@njit()
def wavemaket_exact(
    wavelet_waveform: SparseWaveletWaveform,
    waveform: StationaryWaveformTime,
    nt_lim_waveform: PixelTimeRange,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
) -> None:
    """Compute the wavelet values analogousely to wavemaket using the taylor time method.
    Computes the value at every wavelet pixel directly, without the precomputed interpolation table.
    Useful for testing accuracy/if the interpolation table is dense enough, if only a small
    number of points need to be evaluated, if slopes are too large for the interpolation table to be
    practical, as can happen near nulls in the waveform.
    """
    n_set_old = wavelet_waveform.n_set.copy()

    nc_waveform = wavelet_waveform.wave_value.shape[0]

    wavelet_norm = taylor_table.wavelet_norm

    for itrc in range(nc_waveform):
        mm = 0
        nf_min = 0
        nf_max = wc.Nf - 1

        for j in range(nt_lim_waveform.nt_min, nt_lim_waveform.nt_max):
            j_ind = j # keep j_ind separate in case we want to apply a time offset in the future

            c = waveform.AT[itrc, j] * np.cos(waveform.PT[itrc, j])
            s = waveform.AT[itrc, j] * np.sin(waveform.PT[itrc, j])

            fa = waveform.FT[itrc, j]

            y0 = waveform.FTd[itrc, j] / wc.dfd
            ny = int(np.floor(y0))

            Nfsam1_loc = int(wc.Nsf + 2 / 3 * np.abs(ny) * wc.dfdot * wc.Nsf)
            Nfsam2_loc = int(wc.Nsf + 2 / 3 * np.abs(ny + 1) * wc.dfdot * wc.Nsf)
            # TODO why - 1?
            HBW = (min(Nfsam1_loc, Nfsam2_loc) - 1) * wc.df_bw / 2

            # lowest frequency layer
            kmin = int(np.ceil((fa - HBW) / wc.DF))
            kmin = max(nf_min, kmin)
            kmin = min(nf_max, kmin)

            # highest frequency layer
            kmax = min(nf_max, int(np.floor((fa + HBW) / wc.DF)))
            kmax = max(nf_min, kmax)
            kmax = max(kmin, kmax)

            for k in range(kmin, kmax + 1):
                wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k

                y, z = get_taylor_pixel_direct(fa, waveform.FTd[itrc, j], k, wavelet_norm, wc)

                if (j_ind + k) % 2:
                    wavelet_waveform.wave_value[itrc, mm] = -(c * z + s * y)
                else:
                    wavelet_waveform.wave_value[itrc, mm] = c * y - s * z

                mm += 1
                # end loop over frequency layers

        wavelet_waveform.n_set[itrc] = mm

    # clean up any pixels that were set in the old waveform but aren't anymore
    for itrc in range(nc_waveform):
        for itrm in range(wavelet_waveform.n_set[itrc], n_set_old[itrc]):
            wavelet_waveform.pixel_index[itrc, itrm] = -1
            wavelet_waveform.wave_value[itrc, itrm] = 0.0
