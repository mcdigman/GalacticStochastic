"""helper functions for Chirp_WDM"""

import numpy as np
from numba import njit

from LisaWaveformTools.ra_waveform_time import StationaryWaveformTime
from WaveletWaveforms.coefficientsWDM_time_helpers import SparseTaylorWaveform, WaveletTaylorTimeCoeffs
from WaveletWaveforms.wdm_config import WDMWaveletConstants


@njit(fastmath=True)
def wavemaket_multi_inplace(
    wavelet_waveform: SparseTaylorWaveform,
    waveform: StationaryWaveformTime,
    nt_min,
    nt_max,
    wc: WDMWaveletConstants,
    taylor_table: WaveletTaylorTimeCoeffs,
    *,
    force_nulls=False,
) -> None:
    """Compute the actual wavelets using taylor time method"""
    n_set_old = wavelet_waveform.n_set.copy()

    nc_waveform = wavelet_waveform.wave_value.shape[0]

    for itrc in range(nc_waveform):
        mm = 0
        for j in range(nt_min, nt_max):
            j_ind = j

            y0 = waveform.FTd[itrc, j] / wc.dfd
            ny = np.int64(np.floor(y0))
            n_ind = ny + wc.Nfd_negative

            if 0 <= n_ind < wc.Nfd - 2:
                c = waveform.AT[itrc, j] * np.cos(waveform.PT[itrc, j])
                s = waveform.AT[itrc, j] * np.sin(waveform.PT[itrc, j])

                dy = y0 - ny
                fa = waveform.FT[itrc, j]
                za = fa / wc.df

                Nfsam1_loc = taylor_table.Nfsam[n_ind]
                Nfsam2_loc = taylor_table.Nfsam[n_ind + 1]
                HBW = (min(Nfsam1_loc, Nfsam2_loc) - 1) * wc.df / 2

                # lowest frequency layer
                kmin = max(0, np.int64(np.ceil((fa - HBW) / wc.DF)))

                # highest frequency layer
                kmax = min(wc.Nf - 1, np.int64(np.floor((fa + HBW) / wc.DF)))
                for k in range(kmin, kmax + 1):
                    wavelet_waveform.pixel_index[itrc, mm] = j_ind * wc.Nf + k

                    zmid = (wc.DF / wc.df) * k

                    kk = np.floor(za - zmid - 0.5)
                    zsam = zmid + kk + 0.5
                    kk = np.int64(kk)
                    dx = za - zsam  # used for linear interpolation

                    # interpolate over frequency
                    jj1 = kk + Nfsam1_loc // 2
                    jj2 = kk + Nfsam2_loc // 2

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
            elif force_nulls:
                # we know what the indices would be for values not precomputed in the table
                # so force values outside the range of the table to 0 instead of dropping,
                # in order to get likelihoods right, which is particularly important around total nulls,
                # which can be quite constraining on the parameters but have large spikes in frequency derivative
                # note that if this happens only very rarely
                # we could just actually calculate the non-precomputed coefficient
                fa = waveform.FT[itrc, j]

                Nfsam1_loc = np.int64((wc.BW + wc.dfd * wc.Tw * ny) / wc.df)
                if Nfsam1_loc % 2 == 1:
                    Nfsam1_loc += 1

                HBW = (Nfsam1_loc - 1) * wc.df / 2
                # lowest frequency layer
                kmin = max(0, np.int64(np.ceil((fa - HBW) / wc.DF)))

                # highest frequency layer
                kmax = min(wc.Nf - 1, np.int64(np.floor((fa + HBW) / wc.DF)))

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
