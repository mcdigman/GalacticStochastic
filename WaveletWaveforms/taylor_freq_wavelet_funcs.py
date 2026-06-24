"""Functions for constructing wavelet-domain representations of frequency-domain waveforms.

The functions use the WDM wavelet basis.
"""
import numpy as np
from numba import njit

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformFreq
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, SparseWaveletWaveform
from WaveletWaveforms.taylor_freq_coefficients import WaveletTaylorFreqCoeffs, get_taylor_freq_pixel_direct_alt
from WaveletWaveforms.wdm_config import WDMWaveletConstants


@njit()
def wavemakef(
        wavelet_waveform: SparseWaveletWaveform,
        waveform: StationaryWaveformFreq,
        nt_lim_waveform: PixelGenericRange,
        nf_lim_waveform: PixelGenericRange,
        wc: WDMWaveletConstants,
        taylor_table: WaveletTaylorFreqCoeffs,
        *,
        force_nulls: int = 2,
        amplitude_order: int = 0,
) -> None:
    """Calculate expansion using taylor frequency method choosing only selected F indices"""
    # TODO handle t0 or remove it
    # TODO add input validation
    del force_nulls
    assert amplitude_order in (0, 1)

    t0 = nt_lim_waveform.x_min

    n_set_old = wavelet_waveform.n_set.copy()

    nc_waveform = wavelet_waveform.wave_value.shape[0]

    mm = 0
    for itrc in range(nc_waveform):
        mm = 0
        for j in range(nf_lim_waveform.nx_min, nf_lim_waveform.nx_max):
            if mm >= wavelet_waveform.n_pixel_max:
                msg = 'insufficient size allocation, risk of segmentation fault'
                raise ValueError(msg)

            y0 = waveform.TFp[itrc, j] / wc.dtd
            ii = int(np.int64(np.floor(y0))) + wc.Ntd_negative
            if np.isfinite(y0) and np.isfinite(waveform.TF[itrc, j]) and 0 <= ii < wc.Ntd - 2:  # only proceed if t-prime is covered by lookup table
                j_ind = j

                # TODO check for boundary errors with TF
                c = np.cos(waveform.PF[itrc, j] - 2 * np.pi * t0 * j_ind * wc.DF)
                s = np.sin(waveform.PF[itrc, j] - 2 * np.pi * t0 * j_ind * wc.DF)

                # time pixel range at lower t-prime
                # nb1 = (wc.Tw/wc.delt/2+wc.DF*wc.dtd/wc.delt/2*np.abs(ii-wc.Ntd_negative))
                # TODO need to handle odd Nfsam correctly to eliminate
                # Nfsam_loc1 = np.int64(nb1)
                # Nfsam_loc2 = np.int64(nb1+wc.DF*wc.dtd/wc.delt/2)
                Nfsam_loc1 = taylor_table.Nfsam[ii]
                Nfsam_loc2 = taylor_table.Nfsam[ii + 1]
                # assert Nfsam_loc == Nfsam[ii]
                # note jj and Nfsam could be precomputed
                jj = int(np.int64(np.ceil((wc.delt / wc.DT) * Nfsam_loc1)))
                n = int(np.int64((waveform.TF[itrc, j] - t0) / wc.DT))
                # k = 0 is central pixel in time
                # time direction is reversed. Maybe be due to sign issue in phase
                # kxs are strictly decreasing so only have to choose range once

                # lower t-prime layer
                # linearly interpolate over t-prime
                dy = y0 - ii + wc.Ntd_negative

                # central time of pixel nearest to track
                # t0 = n*wc.DT
                zt = (waveform.TF[itrc, j] - n * wc.DT - t0) / wc.delt

                kxmax = min(n + jj + 2, nt_lim_waveform.nx_max)
                kxmin = max(n - jj, nt_lim_waveform.nx_min)  # TODO this changed from 1, correct everywhere
                for kx in range(kxmin, kxmax):
                    k = n - kx
                    tz = k * (wc.DT / wc.delt) + zt
                    dk = int(np.int64(np.floor(tz)))

                    # if kxmin<=kx<kxmax and dk<Nfsam_loc2 and dk<Nfsam_loc1 and k1>=0 and k2>=0:
                    if dk < Nfsam_loc1 and dk < Nfsam_loc2:
                        # time interpolation for lower beta layer
                        k1 = Nfsam_loc1 + dk
                        # time interpolation for upper beta layer
                        k2 = Nfsam_loc2 + dk

                        if k1 >= 0 and k2 >= 0:
                            dx = tz - dk
                            assert taylor_table.evc[ii, k1 + 1] != 0.
                            assert taylor_table.evc[ii + 1, k2 + 1] != 0.
                            assert taylor_table.evc[ii, k1] != 0.
                            assert taylor_table.evc[ii + 1, k2] != 0.
                            assert 0. <= dx <= 1.

                            if not 0. <= dy <= 1.:
                                print(j, waveform.TFp[itrc, j], y0, ii, dy)
                            assert 0. <= dy <= 1.
                            # print('x1', k1+dx, tz, k1, k2, dx)
                            # print('ii1',ii + dy, dy, Nfsam_loc1, Nfsam_loc2)

                            # interpolate over time
                            y = (1. - dx) * taylor_table.evc[ii, k1] + dx * taylor_table.evc[ii, k1 + 1]
                            z = (1. - dx) * taylor_table.evs[ii, k1] + dx * taylor_table.evs[ii, k1 + 1]

                            # print(y,evc_alt,z,evs_alt)

                            yy = (1. - dx) * taylor_table.evc[ii + 1, k2] + dx * taylor_table.evc[ii + 1, k2 + 1]
                            zz = (1. - dx) * taylor_table.evs[ii + 1, k2] + dx * taylor_table.evs[ii + 1, k2 + 1]

                            # interpolate over beta
                            y = (1. - dy) * y + dy * yy
                            z = (1. - dy) * z + dy * zz

                            # y_alt, z_alt, y_alt2_2, z_alt2_2 = get_taylor_freq_pixel_direct_alt(waveform.TF[itrc, j] - t0, waveform.TFp[itrc, j], kx, taylor_table.wavelet_norm, wc)
                            # print(y, y_alt, taylor_table.evc[ii, k1], taylor_table.evc[ii, k1 + 1], taylor_table.evc[ii + 1, k2], taylor_table.evc[ii + 1, k2 + 1])
                            # print(z, z_alt, taylor_table.evs[ii, k1], taylor_table.evs[ii, k1 + 1], taylor_table.evs[ii + 1, k2], taylor_table.evs[ii + 1, k2 + 1])
                            if amplitude_order > 0:
                                if nf_lim_waveform.nx_min < j < nf_lim_waveform.nx_max - 1:
                                    mult2 = (waveform.AF[itrc, j + 1] - waveform.AF[itrc, j - 1]) / (2 * nf_lim_waveform.dx)
                                elif nf_lim_waveform.nx_min == j < nf_lim_waveform.nx_max - 1:
                                    mult2 = (waveform.AF[itrc, j + 1] - waveform.AF[itrc, j]) / (nf_lim_waveform.dx)
                                elif nf_lim_waveform.nx_min < j == nf_lim_waveform.nx_max - 1:
                                    mult2 = (waveform.AF[itrc, j] - waveform.AF[itrc, j - 1]) / (nf_lim_waveform.dx)
                                else:
                                    mult2 = 0.
                            else:
                                mult2 = 0.

                            z_alt2_part1 = -1 / (2 * np.pi) * (1. / wc.delt) * (taylor_table.evc[ii, k1 + 1] - taylor_table.evc[ii, k1])
                            y_alt2_part1 = 1 / (2 * np.pi) * (1. / wc.delt) * (taylor_table.evs[ii, k1 + 1] - taylor_table.evs[ii, k1])
                            z_alt2_part2 = -1 / (2 * np.pi) * (1. / wc.delt) * (taylor_table.evc[ii + 1, k2 + 1] - taylor_table.evc[ii + 1, k2])
                            y_alt2_part2 = 1 / (2 * np.pi) * (1. / wc.delt) * (taylor_table.evs[ii + 1, k2 + 1] - taylor_table.evs[ii + 1, k2])
                            y_alt2 = (1. - dy) * y_alt2_part1 + dy * y_alt2_part2
                            z_alt2 = (1. - dy) * z_alt2_part1 + dy * z_alt2_part2

                            # print(wc.DT / wc.delt)
                            # print('y', y_alt2_2, y_alt2, y_alt2_part1, y_alt2_part2)
                            # print('z', z_alt2_2, z_alt2, z_alt2_part1, z_alt2_part2)

                            # assert_allclose(y_alt2, y_alt2_2, atol=2.e-8, rtol=3.e-3)
                            # assert_allclose(z_alt2, z_alt2_2, atol=2.e-8, rtol=3.e-3)
                            # assert_allclose(y, y_alt, atol=1.e-14, rtol=1.e-10)
                            # assert_allclose(z, z_alt, atol=1.e-14, rtol=1.e-10)

                            mult = waveform.AF[itrc, j]

                            y = mult * y + mult2 * y_alt2
                            z = mult * z + mult2 * z_alt2

                            if j_ind % 2:
                                if (j_ind + kx) % 2:
                                    wavelet_waveform.wave_value[itrc, mm] = (c * z + s * y)
                                else:
                                    wavelet_waveform.wave_value[itrc, mm] = -(c * y - s * z)
                            else:
                                if (j_ind + kx) % 2:
                                    wavelet_waveform.wave_value[itrc, mm] = (c * z + s * y)
                                else:
                                    wavelet_waveform.wave_value[itrc, mm] = (c * y - s * z)
                            wavelet_waveform.pixel_index[itrc, mm] = kx * wc.Nf + j_ind
                            mm += 1

        wavelet_waveform.n_set[itrc] = mm
    for itrc in range(nc_waveform):
        for itrm in range(wavelet_waveform.n_set[itrc], n_set_old[itrc]):
            wavelet_waveform.pixel_index[itrc, itrm] = -1
            wavelet_waveform.wave_value[itrc, itrm] = 0.0


# @njit()
def wavemakef_direct(
        wavelet_waveform: SparseWaveletWaveform,
        waveform: StationaryWaveformFreq,
        nt_lim_waveform: PixelGenericRange,
        nf_lim_waveform: PixelGenericRange,
        wc: WDMWaveletConstants,
        taylor_table: WaveletTaylorFreqCoeffs,
        *,
        amplitude_order: int=1,
) -> None:
    """Calculate expansion using taylor frequency method choosing only selected F indices"""
    # TODO handle t0 or remove it
    # TODO add input validation
    assert amplitude_order in (0, 1)

    t0 = nt_lim_waveform.x_min
    assert t0 == 0.

    n_set_old = wavelet_waveform.n_set.copy()

    nc_waveform = wavelet_waveform.wave_value.shape[0]

    wavelet_norm = taylor_table.wavelet_norm
    # TODO major limitation is only expanding to zeroth order in amplitude; should ideally be at least first

    mm = 0
    for itrc in range(nc_waveform):
        mm = 0
        for j in range(nf_lim_waveform.nx_min, nf_lim_waveform.nx_max):
            if mm >= wavelet_waveform.n_pixel_max:
                msg = 'insufficient size allocation, risk of segmentation fault'
                raise ValueError(msg)

            y0 = waveform.TFp[itrc, j] / wc.dtd
            ii = int(np.int64(np.floor(y0))) + wc.Ntd_negative
            if 0 <= ii < wc.Ntd - 2:  # only proceed if t-prime is covered by lookup table
                j_ind = j

                # TODO check for boundary errors with TF
                c = np.cos(waveform.PF[itrc, j] - 2 * np.pi * t0 * (j_ind * wc.DF + nf_lim_waveform.x_min))
                s = np.sin(waveform.PF[itrc, j] - 2 * np.pi * t0 * (j_ind * wc.DF + nf_lim_waveform.x_min))

                nb1 = (wc.Tw / wc.delt / 2 + wc.DF * wc.dtd / wc.delt / 2 * np.abs(ii - wc.Ntd_negative))
                # TODO need to handle odd Nfsam correctly to eliminate
                jj = int(np.int64(np.ceil((wc.delt / wc.DT) * np.int64(nb1))))
                n = int(np.int64((waveform.TF[itrc, j] - t0) / wc.DT))

                # k = 0 is central pixel in time
                # time direction is reversed. Maybe be due to sign issue in phase
                # kxs are strictly decreasing so only have to choose range once

                kxmax = min(n + jj + 2, nt_lim_waveform.nx_max)
                kxmin = max(n - jj, nt_lim_waveform.nx_min)
                for kx in range(kxmin, kxmax):
                    # y, z = get_taylor_freq_pixel_direct(waveform.TF[itrc, j] - t0, waveform.TFp[itrc, j], kx, wavelet_norm, wc)
                    y1, z1, y2, z2 = get_taylor_freq_pixel_direct_alt(waveform.TF[itrc, j] - t0, waveform.TFp[itrc, j], kx, wavelet_norm, wc)

                    mult1 = waveform.AF[itrc, j]
                    if amplitude_order > 0:
                        if nf_lim_waveform.nx_min < j < nf_lim_waveform.nx_max - 1:
                            mult2 = (waveform.AF[itrc, j + 1] - waveform.AF[itrc, j - 1]) / (2 * nf_lim_waveform.dx)
                        elif nf_lim_waveform.nx_min == j < nf_lim_waveform.nx_max - 1:
                            mult2 = (waveform.AF[itrc, j + 1] - waveform.AF[itrc, j]) / (nf_lim_waveform.dx)
                        elif nf_lim_waveform.nx_min < j == nf_lim_waveform.nx_max - 1:
                            mult2 = (waveform.AF[itrc, j] - waveform.AF[itrc, j - 1]) / (nf_lim_waveform.dx)
                        else:
                            mult2 = 0.
                    else:
                        mult2 = 0.

                    z = z1 * mult1 + z2 * mult2
                    y = y1 * mult1 + y2 * mult2
                    # TODO check if signs of mult2 components are always correct

                    if j_ind % 2:
                        if (j_ind + kx) % 2:
                            wavelet_waveform.wave_value[itrc, mm] = (c * z + s * y)
                        else:
                            wavelet_waveform.wave_value[itrc, mm] = -(c * y - s * z)
                    else:
                        if (j_ind + kx) % 2:
                            wavelet_waveform.wave_value[itrc, mm] = (c * z + s * y)
                        else:
                            wavelet_waveform.wave_value[itrc, mm] = (c * y - s * z)
                    wavelet_waveform.pixel_index[itrc, mm] = kx * wc.Nf + j_ind
                    mm += 1

        wavelet_waveform.n_set[itrc] = mm
    for itrc in range(nc_waveform):
        for itrm in range(wavelet_waveform.n_set[itrc], n_set_old[itrc]):
            wavelet_waveform.pixel_index[itrc, itrm] = -1
            wavelet_waveform.wave_value[itrc, itrm] = 0.0
