"""Test conversion of tdi spectrum has expected properties"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import tomllib
from numba import njit
from numpy.testing import assert_allclose, assert_array_less
from numpy.typing import NDArray
from scipy.signal import csd, welch
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time

from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import LISAConstants, get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalStationaryDenseNoiseModel
from WaveletWaveforms.wdm_config import WDMWaveletConstants, get_wavelet_model


@njit()
def gen_noise_xyz_offdiagonal_helper(chol_SXYZ_m: NDArray[np.floating], noise_res_white_diag: NDArray[np.floating], wc: WDMWaveletConstants, lc: LISAConstants) -> NDArray[np.floating]:
    """Generate noise in XYZ TDI with off diagonal terms in spectrum."""
    signal_wave_xyz = np.zeros((wc.Nt, wc.Nf, lc.nc_snr))
    for i in range(1, wc.Nf):
        chol_S = np.ascontiguousarray(chol_SXYZ_m[i, :, :])
        for j in range(wc.Nt):
            signal_wave_xyz[j, i] = np.dot(chol_S, noise_res_white_diag[j, i])
    return signal_wave_xyz


@pytest.mark.parametrize('Sps_Sacc_rat_mult', [1.0e3, 1.0e-3, 1.0])
def test_aet_convert(Sps_Sacc_rat_mult: float) -> None:
    """Test conversion of XYZ to AET matches expected spectrum.

    Varying ratio of components allows some better confidence in scaling.
    """
    toml_filename_in = 'tests/tdi_diagonal_consistency_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['Sps'] = np.sqrt(Sps_Sacc_rat_mult) * config_in['lisa_constants']['Sps']
    config_in['lisa_constants']['Sacc'] = 1.0 / np.sqrt(Sps_Sacc_rat_mult) * config_in['lisa_constants']['Sacc']

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    # SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    SXYZ_m = instrument_noise_AET_wdm_m(lc, wc, tdi_mode='xyz_equal', diagonal_mode=1)
    seed = 314159265
    rng = np.random.default_rng(seed)

    chol_SXYZ_m = np.zeros((wc.Nf, lc.nc_snr, lc.nc_snr))
    for i in range(SXYZ_m.shape[0]):
        chol_SXYZ_m[i, :, :] = np.linalg.cholesky(SXYZ_m[i, :, :])

    noise_res_white_diag = rng.normal(0.0, 1.0, (wc.Nt, wc.Nf, lc.nc_snr))
    signal_wave_xyz = gen_noise_xyz_offdiagonal_helper(chol_SXYZ_m, noise_res_white_diag, wc, lc)

    # noise_XYZ_dense_pure = DiagonalStationaryDenseNoiseModel(SXYZ_m, wc, prune=0, nc_snr=lc.nc_snr, seed=seed)
    # signal_wave_xyz = noise_XYZ_dense_pure.generate_dense_noise()

    signal_wave_aet = np.zeros_like(signal_wave_xyz)
    signal_wave_aet[:, :, 0] = 1 / 3 * (2 * signal_wave_xyz[:, :, 0] - signal_wave_xyz[:, :, 1] - signal_wave_xyz[:, :, 2])
    signal_wave_aet[:, :, 1] = 1 / np.sqrt(3) * (signal_wave_xyz[:, :, 2] - signal_wave_xyz[:, :, 1])
    signal_wave_aet[:, :, 2] = 1 / 3 * (signal_wave_xyz[:, :, 0] + signal_wave_xyz[:, :, 1] + signal_wave_xyz[:, :, 2])

    signal_time_aet = np.zeros((wc.Nt * wc.Nf, signal_wave_aet.shape[-1]))
    for itrc in range(signal_wave_aet.shape[-1]):
        signal_time_aet[:, itrc] = inverse_wavelet_time(signal_wave_aet[:, :, itrc], wc.Nf, wc.Nt)

    signal_time_xyz = np.zeros((wc.Nt * wc.Nf, signal_wave_aet.shape[-1]))
    for itrc in range(signal_wave_aet.shape[-1]):
        signal_time_xyz[:, itrc] = inverse_wavelet_time(signal_wave_xyz[:, :, itrc], wc.Nf, wc.Nt)

    signal_time_aet_alt = np.zeros_like(signal_time_xyz)
    signal_time_aet_alt[:, 0] = 1 / 3 * (2 * signal_time_xyz[:, 0] - signal_time_xyz[:, 1] - signal_time_xyz[:, 2])
    signal_time_aet_alt[:, 1] = 1 / np.sqrt(3) * (signal_time_xyz[:, 2] - signal_time_xyz[:, 1])
    signal_time_aet_alt[:, 2] = 1 / 3 * (signal_time_xyz[:, 0] + signal_time_xyz[:, 1] + signal_time_xyz[:, 2])

    nperseg = int(2 * wc.Nf)
    fs = 1.0 / wc.dt
    fpsd, psd_AET = welch(signal_time_aet, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fpsd, psd_xyz = welch(signal_time_xyz, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fpsd, psd_AET_alt = welch(signal_time_aet_alt, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    nf_min = max(1, int(np.argmax(fpsd > 4.0 / wc.DT)), int(np.argmax(fpsd > 4.0 / wc.Tw)))
    nf_max = fpsd.size - 2
    print(fpsd.max())
    # import matplotlib.pyplot as plt

    # plt.loglog(fpsd[nf_min:nf_max], psd_AET[nf_min:nf_max, 0])
    # plt.loglog(fpsd[nf_min:nf_max], psd_aet_exp[:, 0])
    # plt.show()

    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max, 0])
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max, 1])
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max, 2])
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz_exp[:, 0])
    # plt.show()

    fcsd, csd_xyz_01 = csd(signal_time_xyz[:, 0], signal_time_xyz[:, 1], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_xyz_02 = csd(signal_time_xyz[:, 0], signal_time_xyz[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_xyz_12 = csd(signal_time_xyz[:, 1], signal_time_xyz[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)

    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_12[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_12[nf_min:nf_max]))
    # plt.semilogx(fpsd[nf_min:nf_max], instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='xyz_equal', diagonal_mode=1)[:, 0, 1])
    # plt.show()

    csd_scale = (lc.fstr / fcsd[nf_min:nf_max]) ** 2
    csd_xyz_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='xyz_equal', diagonal_mode=1)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fpsd[nf_min:nf_max], csd_xyz_exp[:, 0, 1]*csd_scale)
    # plt.show()

    # check unscaled csds
    atol_csd_real = 2.0e0 * np.max(np.abs(np.imag(csd_xyz_01[nf_min:nf_max])))
    assert_allclose(np.real(csd_xyz_01[nf_min:nf_max]), np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.real(csd_xyz_01[nf_min:nf_max]), np.real(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.imag(csd_xyz_01[nf_min:nf_max]), np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.imag(csd_xyz_01[nf_min:nf_max]), np.imag(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    # check descaled csds
    atol_csd_real = 2.0e0 * np.max(np.abs(np.imag(csd_scale * csd_xyz_01[nf_min:nf_max])))
    assert_allclose(csd_scale * np.real(csd_xyz_01[nf_min:nf_max]), csd_scale * np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_xyz_01[nf_min:nf_max]), csd_scale * np.real(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_xyz_12[nf_min:nf_max]), csd_scale * np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_xyz_01[nf_min:nf_max]), csd_scale * np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_xyz_01[nf_min:nf_max]), csd_scale * np.imag(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_xyz_12[nf_min:nf_max]), csd_scale * np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 1], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 2], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 1, 2], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 1, 0], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 2, 0], csd_scale * csd_xyz_exp[:, 0, 2], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 2, 1], csd_scale * csd_xyz_exp[:, 1, 2], atol=1.0e-100, rtol=1.0e-14)

    atol_csd_imag = 2.6e-1 * np.max(np.abs(np.real(csd_scale * csd_xyz_01[nf_min:nf_max])))
    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 1], csd_scale * np.real(csd_xyz_01[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 2], csd_scale * np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * csd_xyz_exp[:, 1, 2], csd_scale * np.real(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(0.0, csd_scale * np.imag(csd_xyz_01[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)

    assert_array_less(0.99, np.corrcoef(csd_scale * csd_xyz_exp[:, 0, 1], csd_scale * np.real(csd_xyz_01[nf_min:nf_max]))[0, 1])
    assert_array_less(0.99, np.corrcoef(csd_scale * csd_xyz_exp[:, 0, 2], csd_scale * np.real(csd_xyz_02[nf_min:nf_max]))[0, 1])
    assert_array_less(0.99, np.corrcoef(csd_scale * csd_xyz_exp[:, 1, 2], csd_scale * np.real(csd_xyz_12[nf_min:nf_max]))[0, 1])

    fcsd, csd_aet_01 = csd(signal_time_aet[:, 0], signal_time_aet[:, 1], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_aet_02 = csd(signal_time_aet[:, 0], signal_time_aet[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_aet_12 = csd(signal_time_aet[:, 1], signal_time_aet[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)

    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_12[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_12[nf_min:nf_max]))
    # plt.semilogx(fpsd[nf_min:nf_max], instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='aet_equal', diagonal_mode=1)[:, 0, 1])
    # plt.show()

    csd_scale = (lc.fstr / fcsd[nf_min:nf_max]) ** 2
    csd_aet_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='aet_equal', diagonal_mode=1)

    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fpsd[nf_min:nf_max], csd_aet_exp[:, 0, 1]*csd_scale)
    # plt.show()

    # check unscaled csds
    atol_csd_real = 2.0e0 * np.max(np.abs(np.imag(csd_aet_01[nf_min:nf_max])))
    assert_allclose(np.real(csd_aet_01[nf_min:nf_max]), np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.real(csd_aet_01[nf_min:nf_max]), np.real(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.imag(csd_aet_01[nf_min:nf_max]), np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.imag(csd_aet_01[nf_min:nf_max]), np.imag(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    # check descaled csds
    atol_csd_real = 2.0e0 * np.max(np.abs(np.imag(csd_scale * csd_aet_01[nf_min:nf_max])))
    assert_allclose(csd_scale * np.real(csd_aet_01[nf_min:nf_max]), csd_scale * np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_aet_01[nf_min:nf_max]), csd_scale * np.real(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_aet_12[nf_min:nf_max]), csd_scale * np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_aet_01[nf_min:nf_max]), csd_scale * np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_aet_01[nf_min:nf_max]), csd_scale * np.imag(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_aet_12[nf_min:nf_max]), csd_scale * np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    assert_allclose(csd_scale * csd_aet_exp[:, 0, 1], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 0, 2], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 1, 2], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 1, 0], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 2, 0], csd_scale * csd_aet_exp[:, 0, 2], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 2, 1], csd_scale * csd_aet_exp[:, 1, 2], atol=1.0e-100, rtol=1.0e-14)

    atol_csd_imag = 2.0e0 * np.max(np.abs(np.real(csd_scale * csd_aet_01[nf_min:nf_max])))
    assert_allclose(csd_scale * csd_aet_exp[:, 0, 1], csd_scale * np.real(csd_aet_01[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * csd_aet_exp[:, 0, 2], csd_scale * np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * csd_aet_exp[:, 1, 2], csd_scale * np.real(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(0.0, csd_scale * np.imag(csd_aet_01[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_aet_12[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)

    if not np.all(csd_aet_exp[:, 0, 1] == 0.0):
        assert_array_less(0.99, np.corrcoef(csd_scale * csd_aet_exp[:, 0, 1], csd_scale * np.real(csd_aet_01[nf_min:nf_max]))[0, 1])
        assert_array_less(0.99, np.corrcoef(csd_scale * csd_aet_exp[:, 0, 2], csd_scale * np.real(csd_aet_02[nf_min:nf_max]))[0, 1])
        assert_array_less(0.99, np.corrcoef(csd_scale * csd_aet_exp[:, 1, 2], csd_scale * np.real(csd_aet_12[nf_min:nf_max]))[0, 1])

    sin_inv = 1.0 / np.sin(fpsd[nf_min:nf_max] / lc.fstr) ** 2
    # mask off points where we are nearly dividing by zero from the comparison
    sin_mask = 1.0 * (sin_inv < 30.0)
    scale_remove = 1.0 / (64 / (3 * lc.Larm**2) * (fpsd[nf_min:nf_max] / lc.fstr) ** 2) * sin_inv * sin_mask
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max, 0] * scale_remove)
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz_exp[:, 0] * scale_remove)
    # plt.show()

    # match unscaled spectrum
    psd_aet_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='aet_equal')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        psd_xyz_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='xyz_equal')
    assert_allclose(psd_AET[nf_min:nf_max], psd_aet_exp, atol=1.3e-37, rtol=1.0e-1)
    assert_allclose(psd_xyz[nf_min:nf_max], psd_xyz_exp, atol=2.0e-37, rtol=1.0e-1)
    assert_allclose(psd_AET_alt[nf_min:nf_max], psd_AET[nf_min:nf_max], atol=1.0e-100, rtol=1.0e-5)

    # match scaled spectrum
    assert_allclose(psd_xyz[nf_min:nf_max].T * scale_remove, psd_xyz_exp.T * scale_remove, atol=1.0e-100, rtol=9.0e-2)
    assert_allclose(psd_AET[nf_min:nf_max].T * scale_remove, psd_aet_exp.T * scale_remove, atol=1.0e-100, rtol=9.0e-2)

    # isolate known periodic components and check they match
    # cos_mult1 = np.sum(psd_xyz[nf_min:nf_max].T*scale_remove*np.cos(fpsd[nf_min:nf_max]/lc.fstr), axis=1)
    # cos_mult2 = np.sum(psd_xyz_exp.T*scale_remove*np.cos(fpsd[nf_min:nf_max]/lc.fstr), axis=1)
    # assert_allclose(cos_mult1, cos_mult2, atol=1.e-100, rtol=3.e-3)

    cos_mult1 = np.sum(psd_xyz[nf_min:nf_max].T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    cos_mult2 = np.sum(psd_xyz_exp.T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    assert_allclose(cos_mult1, cos_mult2, atol=1.0e-100, rtol=3.0e-2)

    mean1 = np.mean(psd_xyz[nf_min:nf_max].T * scale_remove, axis=1)
    mean2 = np.mean(psd_xyz_exp.T * scale_remove, axis=1)
    assert_allclose(mean1, mean2, atol=1.0e-100, rtol=3.0e-3)

    cos_mult1 = np.sum(psd_AET[nf_min:nf_max].T * scale_remove * np.cos(fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    cos_mult2 = np.sum(psd_aet_exp.T * scale_remove * np.cos(fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    assert_allclose(cos_mult1, cos_mult2, atol=1.0e-100, rtol=3.0e-3)

    cos_mult1 = np.sum(psd_AET[nf_min:nf_max].T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    cos_mult2 = np.sum(psd_aet_exp.T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    assert_allclose(cos_mult1, cos_mult2, atol=1.0e-100, rtol=3.0e-2)

    mean1 = np.mean(psd_AET[nf_min:nf_max].T * scale_remove, axis=1)
    mean2 = np.mean(psd_aet_exp.T * scale_remove, axis=1)
    assert_allclose(mean1, mean2, atol=1.0e-100, rtol=3.0e-3)

    cos_mult1 = psd_xyz[nf_min:nf_max, 0] * np.cos(fpsd[nf_min:nf_max] / lc.fstr)
    cos_mult2 = psd_xyz_exp[:, 0] * np.cos(fpsd[nf_min:nf_max] / lc.fstr)
    print(np.sum(cos_mult1), np.sum(cos_mult2), np.sum(cos_mult2) / np.sum(cos_mult1))
    cos_mult1 = psd_xyz[nf_min:nf_max, 0] * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr)
    cos_mult2 = psd_xyz_exp[:, 0] * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr)
    print(np.sum(cos_mult1), np.sum(cos_mult2), np.sum(cos_mult2) / np.sum(cos_mult1))
    print(np.mean(psd_xyz[nf_min:nf_max, 0]), np.mean(psd_xyz_exp[:, 0]), np.mean(psd_xyz[nf_min:nf_max, 0]) / np.mean(psd_xyz_exp[:, 0]))
    # plt.semilogx(fpsd[nf_min:nf_max], (psd_xyz[nf_min:nf_max, 0] - np.mean(psd_xyz[nf_min:nf_max, 0])))
    # plt.semilogx(fpsd[nf_min:nf_max], (psd_xyz_exp[:, 0] - np.mean(
    #    psd_xyz_exp[:, 0])))
    # plt.show()

    # plt.loglog(fpsd[nf_min:nf_max],
    #           psd_xyz[nf_min:nf_max, 0] / psd_xyz_exp[:, 0])
    # plt.show()

    # plt.semilogx(fpsd[nf_min:nf_max],
    #             psd_xyz[nf_min:nf_max, 0] - psd_xyz_exp[:, 0])
    # plt.show()

    # check converting between AET and XYZ works same in both wavelet and time domain
    assert_allclose(signal_time_aet, signal_time_aet_alt, atol=1.0e-50, rtol=1.2e-7)


@pytest.mark.parametrize('Sps_Sacc_rat_mult', [1.0e3, 1.0e-3, 1.0])
def test_xyz_convert(Sps_Sacc_rat_mult: float) -> None:
    """Test conversion of AET to XYZ matches expected spectrum.

    Varying ratio of components allows some better confidence in scaling.
    """
    toml_filename_in = 'tests/tdi_diagonal_consistency_config1.toml'

    with Path(toml_filename_in).open('rb') as f:
        config_in = tomllib.load(f)

    config_in['lisa_constants']['Sps'] = np.sqrt(Sps_Sacc_rat_mult) * config_in['lisa_constants']['Sps']
    config_in['lisa_constants']['Sacc'] = 1.0 / np.sqrt(Sps_Sacc_rat_mult) * config_in['lisa_constants']['Sacc']

    wc = get_wavelet_model(config_in)
    lc = get_lisa_constants(config_in)

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    seed = 314159265
    noise_AET_dense_pure = DiagonalStationaryDenseNoiseModel(SAET_m, wc, prune=0, nc_snr=lc.nc_snr, seed=seed)
    signal_wave_aet = noise_AET_dense_pure.generate_dense_noise()

    signal_wave_xyz = np.zeros_like(signal_wave_aet)
    signal_wave_xyz[:, :, 0] = signal_wave_aet[:, :, 0] + signal_wave_aet[:, :, 2]
    signal_wave_xyz[:, :, 1] = 1 / 2 * (-signal_wave_aet[:, :, 0] - np.sqrt(3) * signal_wave_aet[:, :, 1] + 2 * signal_wave_aet[:, :, 2])
    signal_wave_xyz[:, :, 2] = 1 / 2 * (-signal_wave_aet[:, :, 0] + np.sqrt(3) * signal_wave_aet[:, :, 1] + 2 * signal_wave_aet[:, :, 2])

    signal_time_aet = np.zeros((wc.Nt * wc.Nf, signal_wave_aet.shape[-1]))
    for itrc in range(signal_wave_aet.shape[-1]):
        signal_time_aet[:, itrc] = inverse_wavelet_time(signal_wave_aet[:, :, itrc], wc.Nf, wc.Nt)

    signal_time_xyz = np.zeros((wc.Nt * wc.Nf, signal_wave_aet.shape[-1]))
    for itrc in range(signal_wave_aet.shape[-1]):
        signal_time_xyz[:, itrc] = inverse_wavelet_time(signal_wave_xyz[:, :, itrc], wc.Nf, wc.Nt)

    signal_time_xyz_alt = np.zeros_like(signal_time_aet)
    signal_time_xyz_alt[:, 0] = signal_time_aet[:, 0] + signal_time_aet[:, 2]
    signal_time_xyz_alt[:, 1] = 1 / 2 * (-signal_time_aet[:, 0] - np.sqrt(3) * signal_time_aet[:, 1] + 2 * signal_time_aet[:, 2])
    signal_time_xyz_alt[:, 2] = 1 / 2 * (-signal_time_aet[:, 0] + np.sqrt(3) * signal_time_aet[:, 1] + 2 * signal_time_aet[:, 2])

    nperseg = int(2 * wc.Nf)
    fs = 1.0 / wc.dt
    fpsd, psd_AET = welch(signal_time_aet, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fpsd, psd_xyz_alt = welch(signal_time_xyz, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fpsd, psd_xyz = welch(signal_time_xyz_alt, fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    nf_min = max(1, int(np.argmax(fpsd > 4.0 / wc.DT)), int(np.argmax(fpsd > 4.0 / wc.Tw)))
    nf_max = fpsd.size - 2
    print(fpsd.max())
    # import matplotlib.pyplot as plt
    # plt.loglog(fpsd[nf_min:nf_max], psd_AET[nf_min:nf_max,0])
    # plt.loglog(fpsd[nf_min:nf_max], psd_aet_exp[:,0])
    # plt.show()

    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max,0])
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max,1])
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max,2])
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz_exp[:,0])
    # plt.show()

    fcsd, csd_xyz_01 = csd(signal_time_xyz[:, 0], signal_time_xyz[:, 1], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_xyz_02 = csd(signal_time_xyz[:, 0], signal_time_xyz[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_xyz_12 = csd(signal_time_xyz[:, 1], signal_time_xyz[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)

    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_12[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_12[nf_min:nf_max]))
    # plt.semilogx(fpsd[nf_min:nf_max], instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='xyz_equal', diagonal_mode=1)[:, 0, 1])
    # plt.show()

    csd_scale = (lc.fstr / fcsd[nf_min:nf_max]) ** 2
    csd_xyz_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='xyz_equal', diagonal_mode=1)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_xyz_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_xyz_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fpsd[nf_min:nf_max], csd_xyz_exp[:, 0, 1]*csd_scale)
    # plt.show()

    # check unscaled csds
    atol_csd_real = 1.5e0 * np.max(np.abs(np.imag(csd_xyz_01[nf_min:nf_max])))
    assert_allclose(np.real(csd_xyz_01[nf_min:nf_max]), np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)
    assert_allclose(np.real(csd_xyz_01[nf_min:nf_max]), np.real(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)
    assert_allclose(np.imag(csd_xyz_01[nf_min:nf_max]), np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)
    assert_allclose(np.imag(csd_xyz_01[nf_min:nf_max]), np.imag(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)

    # check descaled csds
    atol_csd_real = 1.6e0 * np.max(np.abs(np.imag(csd_scale * csd_xyz_01[nf_min:nf_max])))
    assert_allclose(csd_scale * np.real(csd_xyz_01[nf_min:nf_max]), csd_scale * np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_xyz_01[nf_min:nf_max]), csd_scale * np.real(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_xyz_12[nf_min:nf_max]), csd_scale * np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_xyz_01[nf_min:nf_max]), csd_scale * np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_xyz_01[nf_min:nf_max]), csd_scale * np.imag(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_xyz_12[nf_min:nf_max]), csd_scale * np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 1], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 2], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 1, 2], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 1, 0], csd_scale * csd_xyz_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 2, 0], csd_scale * csd_xyz_exp[:, 0, 2], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_xyz_exp[:, 2, 1], csd_scale * csd_xyz_exp[:, 1, 2], atol=1.0e-100, rtol=1.0e-14)

    atol_csd_imag = 2.5e-1 * np.max(np.abs(np.real(csd_scale * csd_xyz_01[nf_min:nf_max])))
    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 1], csd_scale * np.real(csd_xyz_01[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)
    assert_allclose(csd_scale * csd_xyz_exp[:, 0, 2], csd_scale * np.real(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)
    assert_allclose(csd_scale * csd_xyz_exp[:, 1, 2], csd_scale * np.real(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_xyz_01[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_xyz_02[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_xyz_12[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)

    assert_array_less(0.99, np.corrcoef(csd_scale * csd_xyz_exp[:, 0, 1], csd_scale * np.real(csd_xyz_01[nf_min:nf_max]))[0, 1])
    assert_array_less(0.99, np.corrcoef(csd_scale * csd_xyz_exp[:, 0, 2], csd_scale * np.real(csd_xyz_02[nf_min:nf_max]))[0, 1])
    assert_array_less(0.99, np.corrcoef(csd_scale * csd_xyz_exp[:, 1, 2], csd_scale * np.real(csd_xyz_12[nf_min:nf_max]))[0, 1])

    fcsd, csd_aet_01 = csd(signal_time_aet[:, 0], signal_time_aet[:, 1], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_aet_02 = csd(signal_time_aet[:, 0], signal_time_aet[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)
    fcsd, csd_aet_12 = csd(signal_time_aet[:, 1], signal_time_aet[:, 2], fs=fs, nperseg=nperseg, scaling='density', window='tukey', axis=0)

    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_12[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_01[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_02[nf_min:nf_max]))
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_12[nf_min:nf_max]))
    # plt.semilogx(fpsd[nf_min:nf_max], instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='aet_equal', diagonal_mode=1)[:, 0, 1])
    # plt.show()

    csd_scale = (lc.fstr / fcsd[nf_min:nf_max]) ** 2
    csd_aet_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='aet_equal', diagonal_mode=1)

    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.real(csd_aet_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_01[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_02[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fcsd[nf_min:nf_max], np.imag(csd_aet_12[nf_min:nf_max]) * csd_scale)
    # plt.semilogx(fpsd[nf_min:nf_max], csd_aet_exp[:, 0, 1]*csd_scale)
    # plt.show()

    # check unscaled csds
    atol_csd_real = 2.0e0 * np.max(np.abs(np.imag(csd_aet_01[nf_min:nf_max])))
    assert_allclose(np.real(csd_aet_01[nf_min:nf_max]), np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.real(csd_aet_01[nf_min:nf_max]), np.real(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.imag(csd_aet_01[nf_min:nf_max]), np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(np.imag(csd_aet_01[nf_min:nf_max]), np.imag(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    # check descaled csds
    atol_csd_real = 2.0e0 * np.max(np.abs(np.imag(csd_scale * csd_aet_01[nf_min:nf_max])))
    assert_allclose(csd_scale * np.real(csd_aet_01[nf_min:nf_max]), csd_scale * np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_aet_01[nf_min:nf_max]), csd_scale * np.real(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.real(csd_aet_12[nf_min:nf_max]), csd_scale * np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_aet_01[nf_min:nf_max]), csd_scale * np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_aet_01[nf_min:nf_max]), csd_scale * np.imag(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * np.imag(csd_aet_12[nf_min:nf_max]), csd_scale * np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)

    assert_allclose(csd_scale * csd_aet_exp[:, 0, 1], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 0, 2], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 1, 2], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 1, 0], csd_scale * csd_aet_exp[:, 0, 1], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 2, 0], csd_scale * csd_aet_exp[:, 0, 2], atol=1.0e-100, rtol=1.0e-14)
    assert_allclose(csd_scale * csd_aet_exp[:, 2, 1], csd_scale * csd_aet_exp[:, 1, 2], atol=1.0e-100, rtol=1.0e-14)

    atol_csd_imag = 2.0e0 * np.max(np.abs(np.real(csd_scale * csd_aet_01[nf_min:nf_max])))
    assert_allclose(csd_scale * csd_aet_exp[:, 0, 1], csd_scale * np.real(csd_aet_01[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * csd_aet_exp[:, 0, 2], csd_scale * np.real(csd_aet_02[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(csd_scale * csd_aet_exp[:, 1, 2], csd_scale * np.real(csd_aet_12[nf_min:nf_max]), atol=atol_csd_real, rtol=1.1e-1)
    assert_allclose(0.0, csd_scale * np.imag(csd_aet_01[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_aet_02[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)
    assert_allclose(0.0, csd_scale * np.imag(csd_aet_12[nf_min:nf_max]), atol=atol_csd_imag, rtol=1.0e-3)

    if not np.all(csd_aet_exp[:, 0, 1] == 0.0):
        assert_array_less(0.99, np.corrcoef(csd_scale * csd_aet_exp[:, 0, 1], csd_scale * np.real(csd_aet_01[nf_min:nf_max]))[0, 1])
        assert_array_less(0.99, np.corrcoef(csd_scale * csd_aet_exp[:, 0, 2], csd_scale * np.real(csd_aet_02[nf_min:nf_max]))[0, 1])
        assert_array_less(0.99, np.corrcoef(csd_scale * csd_aet_exp[:, 1, 2], csd_scale * np.real(csd_aet_12[nf_min:nf_max]))[0, 1])

    sin_inv = 1.0 / np.sin(fpsd[nf_min:nf_max] / lc.fstr) ** 2
    # mask off points where we are nearly dividing by zero from the comparison
    sin_mask = 1.0 * (sin_inv < 30.0)
    scale_remove = 1.0 / (64 / (3 * lc.Larm**2) * (fpsd[nf_min:nf_max] / lc.fstr) ** 2) * sin_inv * sin_mask
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max,0]*scale_remove)
    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz_exp[:,0]*scale_remove)
    # plt.show()

    # match unscaled spectrum
    psd_aet_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='aet_equal')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        psd_xyz_exp = instrument_noise_AET(fpsd[nf_min:nf_max], lc, tdi_mode='xyz_equal')
    assert_allclose(psd_AET[nf_min:nf_max], psd_aet_exp, atol=1.3e-37, rtol=1.0e-1)
    assert_allclose(psd_xyz[nf_min:nf_max], psd_xyz_exp, atol=2.0e-37, rtol=1.0e-1)
    assert_allclose(psd_xyz_alt[nf_min:nf_max], psd_xyz[nf_min:nf_max], atol=1.0e-100, rtol=1.0e-5)

    # match scaled spectrum
    assert_allclose(psd_xyz[nf_min:nf_max].T * scale_remove, psd_xyz_exp.T * scale_remove, atol=1.0e-100, rtol=8.0e-2)
    assert_allclose(psd_AET[nf_min:nf_max].T * scale_remove, psd_aet_exp.T * scale_remove, atol=1.0e-100, rtol=8.0e-2)

    # isolate known periodic components and check they match
    # cos_mult1 = np.sum(psd_xyz[nf_min:nf_max].T*scale_remove*np.cos(fpsd[nf_min:nf_max]/lc.fstr), axis=1)
    # cos_mult2 = np.sum(psd_xyz_exp.T*scale_remove*np.cos(fpsd[nf_min:nf_max]/lc.fstr), axis=1)
    # assert_allclose(cos_mult1, cos_mult2, atol=1.e-100, rtol=3.e-3)

    cos_mult1 = np.sum(psd_xyz[nf_min:nf_max].T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    cos_mult2 = np.sum(psd_xyz_exp.T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    assert_allclose(cos_mult1, cos_mult2, atol=1.0e-100, rtol=3.0e-2)

    mean1 = np.mean(psd_xyz[nf_min:nf_max].T * scale_remove, axis=1)
    mean2 = np.mean(psd_xyz_exp.T * scale_remove, axis=1)
    assert_allclose(mean1, mean2, atol=1.0e-100, rtol=3.0e-3)

    cos_mult1 = np.sum(psd_AET[nf_min:nf_max].T * scale_remove * np.cos(fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    cos_mult2 = np.sum(psd_aet_exp.T * scale_remove * np.cos(fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    assert_allclose(cos_mult1, cos_mult2, atol=1.0e-100, rtol=3.0e-3)

    cos_mult1 = np.sum(psd_AET[nf_min:nf_max].T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    cos_mult2 = np.sum(psd_aet_exp.T * scale_remove * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr), axis=1)
    assert_allclose(cos_mult1, cos_mult2, atol=1.0e-100, rtol=3.0e-2)

    mean1 = np.mean(psd_AET[nf_min:nf_max].T * scale_remove, axis=1)
    mean2 = np.mean(psd_aet_exp.T * scale_remove, axis=1)
    assert_allclose(mean1, mean2, atol=1.0e-100, rtol=3.0e-3)

    cos_mult1 = psd_xyz[nf_min:nf_max, 0] * np.cos(fpsd[nf_min:nf_max] / lc.fstr)
    cos_mult2 = psd_xyz_exp[:, 0] * np.cos(fpsd[nf_min:nf_max] / lc.fstr)
    print(np.sum(cos_mult1), np.sum(cos_mult2), np.sum(cos_mult2) / np.sum(cos_mult1))
    cos_mult1 = psd_xyz[nf_min:nf_max, 0] * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr)
    cos_mult2 = psd_xyz_exp[:, 0] * np.cos(2 * fpsd[nf_min:nf_max] / lc.fstr)
    print(np.sum(cos_mult1), np.sum(cos_mult2), np.sum(cos_mult2) / np.sum(cos_mult1))
    print(np.mean(psd_xyz[nf_min:nf_max, 0]), np.mean(psd_xyz_exp[:, 0]), np.mean(psd_xyz[nf_min:nf_max, 0]) / np.mean(psd_xyz_exp[:, 0]))
    # plt.semilogx(fpsd[nf_min:nf_max], (psd_xyz[nf_min:nf_max,0] - np.mean(psd_xyz[nf_min:nf_max,0])))
    # plt.semilogx(fpsd[nf_min:nf_max], (psd_xyz_exp[:,0] - np.mean(psd_xyz_exp[:,0])))
    # plt.show()

    # plt.loglog(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max,0]/psd_xyz_exp[:,0])
    # plt.show()

    # plt.semilogx(fpsd[nf_min:nf_max], psd_xyz[nf_min:nf_max,0]-psd_xyz_exp[:,0])
    # plt.show()

    # check converting between AET and XYZ works same in both wavelet and time domain
    assert_allclose(signal_time_xyz, signal_time_xyz_alt, atol=1.0e-50, rtol=2.0e-7)
