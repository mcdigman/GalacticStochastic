"""test comparison of signal for sangria v1 verification binaries"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
import WDMWaveletTransforms.fft_funcs as fft
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq, inverse_wavelet_time

from GalacticStochastic.testing_tools import unit_normal_battery
from LisaWaveformTools.instrument_noise import instrument_noise_AET, instrument_noise_AET_wdm_m
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.noise_model import DiagonalNonstationaryDenseNoiseModel, DiagonalStationaryDenseNoiseModel
from WaveletWaveforms.wdm_config import get_wavelet_model


@pytest.mark.parametrize('scale_mult', [1.0, 2.0])
def test_unit_noise_generation_stat(scale_mult: float) -> None:
    """Test unit normal noise for stationary model produced with input spectrum S_stat_m = 1"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)

    ND = wc.Nf * wc.Nt

    nc_noise = 3
    nc_snr = 3

    S_inst_m_one = np.full((wc.Nf, nc_noise), scale_mult)
    noise_model_stat = DiagonalStationaryDenseNoiseModel(S_inst_m_one, wc, False, nc_snr)

    noise_realization = noise_model_stat.generate_dense_noise()

    for itrc in range(nc_noise):
        _ = unit_normal_battery(noise_realization[:, 1:, itrc].flatten(), mult=np.sqrt(scale_mult), do_assert=True)

    freq_mult = np.sqrt(scale_mult) * np.sqrt(ND // 2)

    print('got noise realization')

    for itrc in range(nc_noise):
        noise_realization_freq = inverse_wavelet_freq(noise_realization[:, :, itrc], wc.Nf, wc.Nt)
        _ = unit_normal_battery(np.asarray(np.real(noise_realization_freq), dtype=np.float64), mult=freq_mult, do_assert=True)
        _ = unit_normal_battery(np.asarray(np.imag(noise_realization_freq), dtype=np.float64), mult=freq_mult, do_assert=True)
        noise_realization_time = np.asarray(fft.irfft(noise_realization_freq), dtype=np.float64)
        _ = unit_normal_battery(noise_realization_time, mult=np.sqrt(scale_mult), do_assert=True)
        noise_realization_time = np.asarray(inverse_wavelet_time(noise_realization[:, :, itrc], wc.Nf, wc.Nt), dtype=np.float64)
        _ = unit_normal_battery(noise_realization_time, mult=np.sqrt(scale_mult), do_assert=True)


@pytest.mark.parametrize('var_select', ['const1', 'const2', 'cos1'])
def test_unit_noise_generation_cyclo_time(var_select: str) -> None:
    """Test unit normal noise for nonstationary model produced with input spectrum S_stat_m = 1"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)

    ND = wc.Nf * wc.Nt
    nc_noise = 3
    nc_snr = 3

    ts = np.arange(0, wc.Nt) * wc.DT
    ts_full = np.arange(0, ND) * wc.dt

    S_one = np.full((wc.Nf, nc_noise), 1.0)

    r_cyclo = np.full((wc.Nt, nc_noise), 1.0)
    r_full = np.full((ND, nc_noise), 1.0)

    if var_select == 'const1':
        pass
    elif var_select == 'const2':
        r_cyclo *= 2
        r_full *= 2
    elif var_select == 'cos1':
        for itrc in range(nc_noise):
            r_cyclo[:, itrc] += 0.5 * np.cos(2 * np.pi / ts.max() * 2 * ts)
            r_full[:, itrc] += 0.5 * np.cos(2 * np.pi / ts.max() * 2 * ts_full)
    else:
        msg = 'unrecognized option for var_select'
        raise ValueError(msg)

    S_cyclo = np.zeros((wc.Nt, wc.Nf, nc_noise))

    for itrc in range(nc_noise):
        S_cyclo[:, :, itrc] = np.outer(r_cyclo[:, itrc], S_one[:, itrc])

    noise_model_cyclo = DiagonalNonstationaryDenseNoiseModel(S_cyclo, wc, False, nc_snr)

    noise_realization_var = noise_model_cyclo.generate_dense_noise()

    for itrc in range(nc_noise):
        for itrt in range(wc.Nt):
            _ = unit_normal_battery(
                noise_realization_var[itrt, :, itrc].flatten(), mult=np.sqrt(r_cyclo[itrt, itrc]), do_assert=True,
            )

    for itrc in range(nc_noise):
        # apply the multiplier as a whitening filter in the time domain
        noise_realization_time = (
            1.0 / np.sqrt(r_full[:, itrc]) * inverse_wavelet_time(noise_realization_var[:, :, itrc], wc.Nf, wc.Nt)
        )
        _ = unit_normal_battery(noise_realization_time, do_assert=True)
        # check frequency components were preserved
        noise_realization_freq = np.fft.rfft(noise_realization_time)
        _ = unit_normal_battery(np.real(noise_realization_freq), mult=np.sqrt(ND // 2), do_assert=True)
        _ = unit_normal_battery(np.imag(noise_realization_freq), mult=np.sqrt(ND // 2), do_assert=True)


def test_noise_normalization_match() -> None:
    """Test ability to generate noise matching known spectrum through wavelet methods"""
    toml_filename = 'tests/spectral_noise_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)
    ND = wc.Nt * wc.Nf
    Nf = wc.Nf
    Nt = wc.Nt
    dt = wc.dt
    Tobs = dt * ND
    fs_fft = np.arange(0, ND // 2 + 1) / Tobs
    nc_noise = 3
    nc_snr = 3

    S_inst_m = instrument_noise_AET_wdm_m(lc, wc)
    spectra_need = np.zeros((ND // 2 + 1, nc_noise))
    spectra_need[1:, :] = np.sqrt(ND // 2) * np.sqrt(instrument_noise_AET(fs_fft[1:], lc))

    # check whitened noise matches correct spectrum
    noise_model_stat = DiagonalStationaryDenseNoiseModel(S_inst_m, wc, True, nc_snr)
    noise_wave = noise_model_stat.generate_dense_noise()
    noise_realization_freq = np.zeros((ND // 2 + 1, nc_noise), dtype=np.complex128)
    for itrc in range(nc_noise):
        noise_realization_freq[:, itrc] = inverse_wavelet_freq(noise_wave[:, :, itrc], Nf, Nt)
        # NOTE have to cut off Nt because at very low frequencies
        # we are not currently estimating the spectrum correctly in the wavelet domain
        # also dont't hit the frequencies with big dips
        arglim = np.int64(np.int64(np.pi) * lc.fstr * wc.Tobs)
        _ = unit_normal_battery(
            np.real(noise_realization_freq[Nt // 2:arglim, itrc] / spectra_need[Nt // 2:arglim, itrc]),
            mult=1.0,
            do_assert=True,
        )
        _ = unit_normal_battery(
            np.imag(noise_realization_freq[Nt // 2:arglim, itrc] / spectra_need[Nt // 2:arglim, itrc]),
            mult=1.0,
            do_assert=True,
        )

    # check can generate noise through nonstationary method as well
    noise_model_cyclo = DiagonalNonstationaryDenseNoiseModel(noise_model_stat.S, wc, True, nc_snr)
    noise_wave_var = noise_model_cyclo.generate_dense_noise()
    noise_realization_freq_var = np.zeros((ND // 2 + 1, nc_noise), dtype=np.complex128)
    for itrc in range(nc_noise):
        noise_realization_freq_var[:, itrc] = inverse_wavelet_freq(noise_wave_var[:, :, itrc], Nf, Nt)
        # NOTE have to cut off Nt because at very low frequencies
        # we are not currently estimating the spectrum correctly in the wavelet domain
        # also dont't hit the frequencies with big dips
        arglim = np.int64(np.int64(np.pi) * lc.fstr * wc.Tobs)
        _ = unit_normal_battery(
            np.real(noise_realization_freq_var[Nt // 2:arglim, itrc] / spectra_need[Nt // 2:arglim, itrc]),
            mult=1.0,
            do_assert=True,
        )
        _ = unit_normal_battery(
            np.imag(noise_realization_freq_var[Nt // 2:arglim, itrc] / spectra_need[Nt // 2:arglim, itrc]),
            mult=1.0,
            do_assert=True,
        )
