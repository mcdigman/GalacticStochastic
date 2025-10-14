"""Unit tests for get_time_tdi_amp_phase.
There is also some incidental/partial coverage of wavemaket.
"""

from pathlib import Path

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import butter, filtfilt
from WDMWaveletTransforms.transform_freq_funcs import tukey
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_time

from LisaWaveformTools.algebra_tools import gradient_uniform_inplace
from LisaWaveformTools.lisa_config import get_lisa_constants
from LisaWaveformTools.ra_waveform_time import get_time_tdi_amp_phase
from LisaWaveformTools.spacecraft_objects import AntennaResponseChannels, EdgeRiseModel
from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, sparse_addition_helper
from WaveletWaveforms.taylor_time_coefficients import (
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket
from WaveletWaveforms.wdm_config import get_wavelet_model

# the table takes a while to compute so share it between tests
toml_filename_in = 'tests/galactic_fit_test_config1.toml'

with Path(toml_filename_in).open('rb') as f:
    config_in = tomllib.load(f)

wc_in = get_wavelet_model(config_in)
taylor_time_table = get_taylor_table_time(wc_in, cache_mode='check', output_mode='hf')


def get_waveform_helper(p_input: float, f_input: float, fp_input: float, fpp_input: float, amp_input: float, nt_loc: int, DT: float, nc_waveform: int, max_f: float) -> tuple[StationaryWaveformTime, StationaryWaveformTime, int]:
    """Help get intrinsic_waveform objects."""
    T = np.arange(nt_loc) * DT
    PT = 2 * np.pi * (f_input + 1.0 / 2.0 * fp_input * T + 1.0 / 6.0 * fpp_input * T**2) * T + p_input
    FT = f_input + fp_input * T + 1.0 / 2.0 * fpp_input * T**2
    FTd = fp_input + fpp_input * T
    AT = np.full(nt_loc, amp_input)

    if np.any((max_f < FT) | (FT < 0.0)):
        arg_cut = int(np.argmax((max_f < FT) | (FT < 0.0)))
        PT[arg_cut:] = PT[arg_cut]
        FTd[arg_cut:] = 0.0
        AT[arg_cut:] = 0.0
        if FT[arg_cut] < 0.0:
            FT[arg_cut:] = 0.0
        else:
            FT[arg_cut:] = max_f
    else:
        arg_cut = int(nt_loc)

    AET_PT = np.zeros((nc_waveform, nt_loc))
    AET_FT = np.zeros((nc_waveform, nt_loc))
    AET_FTd = np.zeros((nc_waveform, nt_loc))
    AET_AT = np.zeros((nc_waveform, nt_loc))
    waveform = StationaryWaveformTime(T, PT, FT, FTd, AT)
    AET_waveform = StationaryWaveformTime(T, AET_PT, AET_FT, AET_FTd, AET_AT)
    return waveform, AET_waveform, arg_cut


def get_RR_t_mult(rr_model: str, nt_loc: int, nf_loc: int, dt: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get a multiplier for RR and II for testing"""
    if rr_model == 'const':
        RR_t_mult: NDArray[np.floating] = np.full(nt_loc * nf_loc, 1.0)
        II_t_mult: NDArray[np.floating] = np.full(nt_loc * nf_loc, 1.0)
    elif rr_model == 'lin17':
        RR_t_mult = np.linspace(
            0.17946632189870892,
            0.17946632189870892 - 4.633410079678654e-08 * dt * (nt_loc - nt_loc / 2) * nf_loc,
            nt_loc * nf_loc,
        )
        II_t_mult = np.linspace(
            0.005095443502715089,
            0.005095443502715089 - 1.1460665211908961e-07 * dt * (nt_loc - nt_loc / 2) * nf_loc,
            nt_loc * nf_loc,
        )
    elif rr_model == 'lin18':
        II_t_mult = -np.linspace(
            0.17946632189870892,
            0.17946632189870892 - 4.633410079678654e-08 * dt * (nt_loc - nt_loc / 2) * nf_loc,
            nt_loc * nf_loc,
        )
        RR_t_mult = -np.linspace(
            0.005095443502715089,
            0.005095443502715089 - 1.1460665211908961e-07 * dt * (nt_loc - nt_loc / 2) * nf_loc,
            nt_loc * nf_loc,
        )
    elif rr_model == 'quad2':
        RR_t_mult = np.linspace(-1.0, 1.0, (nt_loc + 1) * nf_loc)[nf_loc :] ** 2
        II_t_mult = np.linspace(-1.0, 1.0, (nt_loc + 1) * nf_loc)[nf_loc :] ** 2
    elif rr_model == 'sin1':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
        II_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
    elif rr_model == 'sin2':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
        II_t_mult = np.cos(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
    elif rr_model == 'sin3':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
        II_t_mult = -np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
    elif rr_model == 'sin4':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
        II_t_mult = np.sin(2 * np.pi * (np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :] + 2.0 / nt_loc))
    elif rr_model == 'sin5':
        RR_t_mult = -np.cos(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
        II_t_mult = -np.sin(2 * np.pi * np.linspace(1.0, -1.0, (nt_loc + 1) * nf_loc)[nf_loc :])
    else:
        msg = 'unrecognized option for rr_model=' + str(rr_model)
        raise ValueError(msg)
    return RR_t_mult, II_t_mult


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.1 / 128,
        0.5 / 128,
        1.0 / 128.0,
        1.1 / 128,
        1.5 / 128,
        2.0 / 128,
        2.5 / 128,
        3.0 / 128,
        4.0 / 128,
        0.01,
        1.0 / 4.0,
        1.0 / 4.0 + 0.5 / 128,
        1.0 / 2.0,
        1.0 / 2.0 + 0.5 / 128,
        0.95,
        0.95 + 1.0 / 128,
        0.999,
        1.0,
        2.0,
        3.0,
        10.0,
        100.0,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0.0, 0.001, 0.01, 1.0 / 2.0, 1.0, 10.0])
@pytest.mark.parametrize(
    'rr_model', ['const', 'lin1', 'lin2', 'lin3', 'lin10', 'lin11', 'lin12', 'lin13', 'lin14', 'lin15', 'lin16'],
)
# @pytest.mark.skip()
def test_ExtractAmpPhase_inplace_basic(f0_mult: float, rr_model: str, f0p_mult: float) -> None:
    """Test the extraction in some easier cases"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)
    er = EdgeRiseModel(-np.inf, np.inf)

    nt_loc = wc.Nt
    nc_waveform = lc.nc_waveform

    # Create fake input data for a pure sine wave
    f_input = wc.DF * wc.Nf * f0_mult
    fp_input = f0p_mult * f_input / wc.Tobs
    amp_input = 1.0
    T = np.arange(nt_loc) * wc.DT
    PT = 2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T
    FT = f_input + fp_input * T  # np.gradient(PT, T, edge_order=2)/(2*np.pi)
    FTd = np.full(nt_loc, fp_input)  # np.gradient(FT, T, edge_order=2)

    AT = np.full(nt_loc, amp_input)

    AET_PT = np.zeros((nc_waveform, nt_loc))
    AET_FT = np.zeros((nc_waveform, nt_loc))
    AET_FTd = np.zeros((nc_waveform, nt_loc))
    AET_AT = np.zeros((nc_waveform, nt_loc))
    waveform = StationaryWaveformTime(T.copy(), PT.copy(), FT.copy(), FTd.copy(), AT.copy())
    AET_waveform = StationaryWaveformTime(T.copy(), AET_PT.copy(), AET_FT.copy(), AET_FTd.copy(), AET_AT.copy())

    # ensure the RRs and IIs are scaled diferently in different channels
    RR_scale_mult = np.array([0.9, 0.5, 0.3])
    II_scale_mult = np.array([0.4, 0.7, 0.8])
    if rr_model == 'const':
        RR_t_mult: NDArray[np.floating] = np.full(nt_loc, 1.0)
        II_t_mult: NDArray[np.floating] = np.full(nt_loc, 1.0)
    elif rr_model == 'lin1':
        RR_t_mult = np.linspace(0.5, 1.5, nt_loc)
        II_t_mult = np.linspace(0.5, 1.5, nt_loc)
    elif rr_model == 'lin2':
        RR_t_mult = np.linspace(1.5, 0.5, nt_loc)
        II_t_mult = np.linspace(1.5, 0.5, nt_loc)
    elif rr_model == 'lin3':
        RR_t_mult = np.linspace(0.5, 1.5, nt_loc)
        II_t_mult = np.linspace(1.5, 0.5, nt_loc)
    elif rr_model == 'lin4':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc)
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc)
    elif rr_model == 'lin5':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc)
        II_t_mult = np.linspace(1.0, -1.0, nt_loc)
    elif rr_model == 'lin6':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin7':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin8':
        RR_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin9':
        RR_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin10':
        RR_t_mult = np.linspace(1.5, 0.5, nt_loc)
        II_t_mult = np.linspace(0.5, 1.5, nt_loc)
    elif rr_model == 'lin11':
        RR_t_mult = np.linspace(-1.5, -0.5, nt_loc)
        II_t_mult = np.linspace(0.5, 1.5, nt_loc)
    elif rr_model == 'lin12':
        RR_t_mult = np.linspace(-1.5, -0.5, nt_loc)
        II_t_mult = np.linspace(-0.5, -1.5, nt_loc)
    elif rr_model == 'lin13':
        RR_t_mult = np.linspace(1.5, 0.5, nt_loc)
        II_t_mult = np.linspace(-0.5, -1.5, nt_loc)
    elif rr_model == 'lin14':
        RR_t_mult = np.linspace(-0.5, -1.5, nt_loc)
        II_t_mult = np.linspace(-1.5, -0.5, nt_loc)
    elif rr_model == 'lin15':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(-1.5, -0.5, nt_loc)
    elif rr_model == 'lin16':
        RR_t_mult = np.linspace(-0.5, -1.5, nt_loc)
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
    else:
        msg = 'unrecognized option for rr_model'
        raise ValueError(msg)

    RR = (np.ones((nc_waveform, nt_loc)).T * RR_scale_mult).T * RR_t_mult
    II = (np.ones((nc_waveform, nt_loc)).T * II_scale_mult).T * II_t_mult
    dRR = np.zeros((nc_waveform, nt_loc))
    dII = np.zeros((nc_waveform, nt_loc))
    ddRR = np.zeros((nc_waveform, nt_loc))
    ddII = np.zeros((nc_waveform, nt_loc))
    spacecraft_channels = AntennaResponseChannels(T, RR.copy(), II.copy(), dRR, dII)
    nt_lim = PixelGenericRange(0, nt_loc, wc.DT, lc.t0)

    # Call the function
    get_time_tdi_amp_phase(spacecraft_channels, AET_waveform, waveform, lc, er, nt_lim)

    # Check that input wavefrom objects have not mutated
    assert np.all(spacecraft_channels.RR == RR)
    assert np.all(spacecraft_channels.II == II)
    assert np.all(AET_waveform.T == T)
    assert np.all(waveform.T == T)
    assert np.all(waveform.AT == AT)
    assert np.all(waveform.PT == PT)
    assert np.all(waveform.FT == FT)
    assert np.all(waveform.FTd == FTd)

    # Check that the inputs respect expected derivatives
    assert_allclose(
        np.gradient(waveform.PT, T, edge_order=2) / (2 * np.pi), waveform.FT, atol=1.0e-12 * f_input, rtol=1.0e-12,
    )
    assert_allclose(np.gradient(waveform.FT, T, edge_order=2), waveform.FTd, atol=1.0e-19 * f_input, rtol=1.0e-10)

    # Check the input looks like a sine wave
    assert_allclose(waveform.FT, f_input + fp_input * T, atol=1.0e-12 * f_input, rtol=1.0e-12)
    assert_allclose(waveform.FTd, fp_input, atol=1.0e-11 * f_input / wc.DT, rtol=1.0e-13)

    for itrc in range(nc_waveform):
        # check the computed dRR and dII match
        assert_allclose(
            spacecraft_channels.dRR[itrc], np.gradient(RR[itrc], T, edge_order=1), atol=1.0e-14, rtol=1.0e-14,
        )
        assert_allclose(
            spacecraft_channels.dII[itrc], np.gradient(II[itrc], T, edge_order=1), atol=1.0e-14, rtol=1.0e-14,
        )

    # Checks that should work for all test variants
    for itrc in range(nc_waveform):
        # test the gradients respect expected rules
        assert_allclose(
            np.gradient(AET_waveform.PT[itrc], T, edge_order=2) / (2 * np.pi),
            AET_waveform.FT[itrc],
            atol=1.0e-7 * f_input,
            rtol=1.0e-8,
        )
        assert_allclose(
            np.gradient(AET_waveform.FT[itrc], T, edge_order=1),
            AET_waveform.FTd[itrc],
            atol=1.0e-7 * f_input / wc.DT,
            rtol=1.0e-8,
        )

        # same rules as the gradients, but test the integrals match too
        assert_allclose(
            AET_waveform.PT[itrc],
            2 * np.pi * cumtrapz(AET_waveform.FT[itrc], T, initial=0) + AET_waveform.PT[itrc, 0],
            atol=1.0e-8 * f0_mult * wc.DT,
            rtol=1.0e-9 * f0_mult * wc.DT,
        )
        assert_allclose(
            AET_waveform.FT[itrc],
            cumtrapz(AET_waveform.FTd[itrc], T, initial=0) + AET_waveform.FT[itrc, 0],
            atol=1.0e-9 * f0_mult,
            rtol=1.0e-12 * f0_mult,
        )

        # check that the amplitudes match the known answer
        assert_allclose(
            AET_waveform.AT[itrc],
            (8 * (FT / lc.fstr) * np.sin(FT / lc.fstr) * waveform.AT) * np.sqrt(RR[itrc] ** 2 + II[itrc] ** 2),
            atol=1.0e-11,
            rtol=1.0e-11,
        )

        # check that phases match
        # p_offset = np.unwrap(np.arctan2(II[itrc], RR[itrc]) % (2 * np.pi))
        # assert_allclose(tdi_waveform.PT[itrc] - 2*np.pi*T*tdi_waveform.FT[itrc],
        #    (intrinsic_waveform.PT - 2*np.pi*T*intrinsic_waveform.FT + p_offset)
        #    % (2*np.pi), atol=1.e-14*f0_mult*_wc.DT, rtol=1.e-14*f0_mult*_wc.DT)

    gradient_uniform_inplace(dII, ddII, wc.DT)
    gradient_uniform_inplace(dRR, ddRR, wc.DT)

    # Checks specific for static sine wave
    for itrc in range(nc_waveform):
        p_offset0: NDArray[np.floating] = np.arctan2(II[itrc], RR[itrc]) % (2 * np.pi)
        dp_offset0: NDArray[np.floating] = np.gradient(p_offset0, T, edge_order=2) / (2 * np.pi)
        dp_offset1: NDArray[np.floating] = (
            -(II[itrc] * spacecraft_channels.dRR[itrc] - RR[itrc] * spacecraft_channels.dII[itrc])
            / (RR[itrc] ** 2 + II[itrc] ** 2)
            / (2 * np.pi)
        )

        # check that derivative of offset phases matches analytic expectation
        assert_allclose(dp_offset0, dp_offset1, atol=1.0e-10 * f_input, rtol=1.0e-5)

        ddp_offset0 = np.gradient(dp_offset0, T, edge_order=1)
        ddp_offset1 = np.gradient(dp_offset1, T, edge_order=1)
        ddp_offset2 = (
            -2
            * (RR[itrc] * dII[itrc] - II[itrc] * dRR[itrc])
            * (II[itrc] * dII[itrc] + RR[itrc] * dRR[itrc])
            / (II[itrc] ** 2 + RR[itrc] ** 2) ** 2
            + (RR[itrc] * ddII[itrc] - II[itrc] * ddRR[itrc]) / (II[itrc] ** 2 + RR[itrc] ** 2)
        ) / (2 * np.pi)

        assert_allclose(ddp_offset0, ddp_offset1, atol=1.0e-7 * f_input / wc.DT, rtol=1.0e-5)
        assert_allclose(ddp_offset1, ddp_offset2, atol=1.0e-7 * f_input / wc.DT, rtol=1.0e-12)
        assert_allclose(
            np.sin(AET_waveform.PT[itrc] - p_offset0),
            np.sin(2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        assert_allclose(
            np.cos(AET_waveform.PT[itrc] - p_offset0),
            np.cos(2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        assert_allclose(
            AET_waveform.PT[itrc] - p_offset0,
            2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T,
            atol=float(1.0e-13 * 2 * np.pi * f_input * T.max()),
            rtol=1.0e-13,
        )
        assert_allclose(
            AET_waveform.FT[itrc] - dp_offset1, f_input + fp_input * T, atol=1.0e-11 * f_input, rtol=1.0e-12,
        )
        assert_allclose(AET_waveform.FTd[itrc] - ddp_offset2, fp_input, atol=1.0e-7 * f_input / wc.DT, rtol=1.0e-8)


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.1 / 128,
        0.5 / 128,
        1.0 / 128.0,
        1.1 / 128,
        1.5 / 128,
        2.0 / 128,
        2.5 / 128,
        3.0 / 128,
        4.0 / 128,
        0.01,
        1.0 / 4.0,
        1.0 / 4.0 + 0.5 / 128,
        1.0 / 2.0,
        1.0 / 2.0 + 0.5 / 128,
        0.95,
        0.95 + 1.0 / 128,
        0.999,
        1.0,
        2.0,
        3.0,
        10.0,
        100.0,
    ],
)
@pytest.mark.parametrize('f0p_mult', [0.0, 0.001, 0.01, 1.0 / 2.0, 1.0, 10.0])
@pytest.mark.parametrize(
    'rr_model',
    [
        'sin1',
        'sin2',
        'sin3',
        'sin4',
        'quad1',
        'quad2',
        'quad3',
        'quad4',
        'lin4',
        'lin5',
        'lin6',
        'lin7',
        'lin8',
        'lin9',
        'lin17',
        'lin18',
    ],
)
# @pytest.mark.skip()
def test_time_tdi_inplace_nearzero(f0_mult: float, rr_model: str, f0p_mult: float) -> None:
    """Test extraction in cases where the RR or II or both go near zero"""
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    lc = get_lisa_constants(config)
    er = EdgeRiseModel(-np.inf, np.inf)

    nt_loc = wc.Nt
    nc_waveform = lc.nc_waveform

    # Create fake input data for a pure sine wave
    f_input = wc.DF * wc.Nf * f0_mult
    fp_input = f0p_mult * f_input / wc.Tobs
    amp_input = 1.0
    T = np.arange(nt_loc) * wc.DT
    PT = 2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T
    FT = f_input + fp_input * T  # np.gradient(PT, T, edge_order=2)/(2*np.pi)
    FTd = np.full(nt_loc, fp_input)  # np.gradient(FT, T, edge_order=2)

    AT = np.full(nt_loc, amp_input)

    AET_PT: NDArray[np.floating] = np.zeros((nc_waveform, nt_loc))
    AET_FT: NDArray[np.floating] = np.zeros((nc_waveform, nt_loc))
    AET_FTd: NDArray[np.floating] = np.zeros((nc_waveform, nt_loc))
    AET_AT: NDArray[np.floating] = np.zeros((nc_waveform, nt_loc))
    waveform = StationaryWaveformTime(T.copy(), PT.copy(), FT.copy(), FTd.copy(), AT.copy())
    AET_waveform = StationaryWaveformTime(T.copy(), AET_PT.copy(), AET_FT.copy(), AET_FTd.copy(), AET_AT.copy())

    # ensure the RRs and IIs are scaled diferently in different channels
    # RR_scale_mult = np.array([0.9, 0.5, 0.3])
    # II_scale_mult = np.array([0.4, 0.7, 0.8])
    RR_scale_mult = np.array([1.0, 1.0, 1.0])
    II_scale_mult = np.array([1.0, 1.0, 1.0])
    if rr_model == 'const':
        RR_t_mult: NDArray[np.floating] = np.full(nt_loc, 1.0)
        II_t_mult: NDArray[np.floating] = np.full(nt_loc, 1.0)
    elif rr_model == 'lin1':
        RR_t_mult = np.linspace(0.5, 1.5, nt_loc)
        II_t_mult = np.linspace(0.5, 1.5, nt_loc)
    elif rr_model == 'lin2':
        RR_t_mult = np.linspace(1.5, 0.5, nt_loc)
        II_t_mult = np.linspace(1.5, 0.5, nt_loc)
    elif rr_model == 'lin3':
        RR_t_mult = np.linspace(0.5, 1.5, nt_loc)
        II_t_mult = np.linspace(1.5, 0.5, nt_loc)
    elif rr_model == 'lin4':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc)
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc)
    elif rr_model == 'lin5':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc)
        II_t_mult = np.linspace(1.0, -1.0, nt_loc)
    elif rr_model == 'lin6':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin7':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin8':
        RR_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin9':
        RR_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin10':
        RR_t_mult = np.linspace(1.5, 0.5, nt_loc)
        II_t_mult = np.linspace(0.5, 1.5, nt_loc)
    elif rr_model == 'lin11':
        RR_t_mult = np.linspace(-1.5, -0.5, nt_loc)
        II_t_mult = np.linspace(0.5, 1.5, nt_loc)
    elif rr_model == 'lin12':
        RR_t_mult = np.linspace(-1.5, -0.5, nt_loc)
        II_t_mult = np.linspace(-0.5, -1.5, nt_loc)
    elif rr_model == 'lin13':
        RR_t_mult = np.linspace(1.5, 0.5, nt_loc)
        II_t_mult = np.linspace(-0.5, -1.5, nt_loc)
    elif rr_model == 'lin14':
        RR_t_mult = np.linspace(-0.5, -1.5, nt_loc)
        II_t_mult = np.linspace(-1.5, -0.5, nt_loc)
    elif rr_model == 'lin15':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
        II_t_mult = np.linspace(-1.5, -0.5, nt_loc)
    elif rr_model == 'lin16':
        RR_t_mult = np.linspace(-0.5, -1.5, nt_loc)
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:]
    elif rr_model == 'lin17':
        RR_t_mult = np.linspace(
            0.17946632189870892, 0.17946632189870892 - 4.633410079678654e-08 * wc.DT * nt_loc, nt_loc,
        )
        II_t_mult = np.linspace(
            0.005095443502715089, 0.005095443502715089 - 1.1460665211908961e-07 * wc.DT * nt_loc, nt_loc,
        )
    elif rr_model == 'lin18':
        II_t_mult = -np.linspace(
            0.17946632189870892, 0.17946632189870892 - 4.633410079678654e-08 * wc.DT * nt_loc, nt_loc,
        )
        RR_t_mult = -np.linspace(
            0.005095443502715089, 0.005095443502715089 - 1.1460665211908961e-07 * wc.DT * nt_loc, nt_loc,
        )
    elif rr_model == 'quad1':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:] ** 2
        II_t_mult = np.linspace(1.0, 1.0, nt_loc + 1)[1:] ** 2
    elif rr_model == 'quad2':
        RR_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:] ** 2
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:] ** 2
    elif rr_model == 'quad3':
        RR_t_mult = np.linspace(1.0, -1.0, nt_loc + 1)[1:] ** 2
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc + 1)[1:] ** 2
    elif rr_model == 'quad4':
        RR_t_mult = np.linspace(1.0, -1.0, nt_loc) ** 2
        II_t_mult = np.linspace(-1.0, 1.0, nt_loc) ** 2
    elif rr_model == 'sin1':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
        II_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
    elif rr_model == 'sin2':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
        II_t_mult = np.cos(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
    elif rr_model == 'sin3':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
        II_t_mult = -np.sin(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
    elif rr_model == 'sin4':
        RR_t_mult = np.sin(2 * np.pi * np.linspace(1.0, -1.0, nt_loc + 1)[1:])
        II_t_mult = np.sin(2 * np.pi * (np.linspace(1.0, -1.0, nt_loc + 1)[1:] + 2.0 / nt_loc))
    else:
        msg = 'unrecognized option for rr_model=' + str(rr_model)
        raise ValueError(msg)

    RR = (np.ones((nc_waveform, nt_loc)).T * RR_scale_mult).T * RR_t_mult
    II = (np.ones((nc_waveform, nt_loc)).T * II_scale_mult).T * II_t_mult
    dRR = np.zeros((nc_waveform, nt_loc))
    dII = np.zeros((nc_waveform, nt_loc))
    ddRR = np.zeros((nc_waveform, nt_loc))
    ddII = np.zeros((nc_waveform, nt_loc))
    spacecraft_channels = AntennaResponseChannels(T, RR.copy(), II.copy(), dRR, dII)
    nt_lim = PixelGenericRange(0, nt_loc, wc.DT, lc.t0)

    # Call the function
    get_time_tdi_amp_phase(spacecraft_channels, AET_waveform, waveform, lc, er, nt_lim)

    # Check that input wavefrom objects have not mutated
    assert np.all(spacecraft_channels.RR == RR)
    assert np.all(spacecraft_channels.II == II)
    assert np.all(AET_waveform.T == T)
    assert np.all(waveform.T == T)
    assert np.all(waveform.AT == AT)
    assert np.all(waveform.PT == PT)
    assert np.all(waveform.FT == FT)
    assert np.all(waveform.FTd == FTd)

    gradient_uniform_inplace(dII, ddII, wc.DT)
    gradient_uniform_inplace(dRR, ddRR, wc.DT)

    # Check that the inputs respect expected derivatives
    assert_allclose(
        np.gradient(waveform.PT, T, edge_order=2) / (2 * np.pi), waveform.FT, atol=1.0e-12 * f_input, rtol=1.0e-12,
    )
    assert_allclose(np.gradient(waveform.FT, T, edge_order=2), waveform.FTd, atol=1.0e-19 * f_input, rtol=1.0e-10)

    # Check the input looks like a sine wave
    assert_allclose(waveform.FT, f_input + fp_input * T, atol=1.0e-12 * f_input, rtol=1.0e-12)
    assert_allclose(waveform.FTd, fp_input, atol=1.0e-11 * f_input / wc.DT, rtol=1.0e-13)

    for itrc in range(nc_waveform):
        # check the computed dRR and dII match
        assert_allclose(
            spacecraft_channels.dRR[itrc], np.gradient(RR[itrc], T, edge_order=1), atol=1.0e-14, rtol=1.0e-14,
        )
        assert_allclose(
            spacecraft_channels.dII[itrc], np.gradient(II[itrc], T, edge_order=1), atol=1.0e-14, rtol=1.0e-14,
        )

    # Checks that should work for all test variants
    for itrc in range(nc_waveform):
        # test the gradients respect expected rules
        assert_allclose(
            np.gradient(AET_waveform.FT[itrc], T, edge_order=1),
            AET_waveform.FTd[itrc],
            atol=1.0e-5 * f_input / wc.DT,
            rtol=1.0e-7,
        )

        # same rules as the gradients, but test the integrals match too
        # assert_allclose(tdi_waveform.FT[itrc], cumtrapz(tdi_waveform.FTd[itrc], T, initial=0.)
        #    + tdi_waveform.FT[itrc, 0], atol=1.e-7*f0_mult, rtol=1.e-12*f0_mult)

        # check that the amplitudes match the known answer
        assert_allclose(
            AET_waveform.AT[itrc],
            (8 * (FT / lc.fstr) * np.sin(FT / lc.fstr) * waveform.AT) * np.sqrt(RR[itrc] ** 2 + II[itrc] ** 2),
            atol=1.0e-11,
            rtol=1.0e-11,
        )

    # Checks specitic for static sine wave
    for itrc in range(nc_waveform):
        p_offset0 = np.arctan2(II[itrc], RR[itrc]) % (2 * np.pi)
        # dp_offset0 = np.gradient(p_offset0, T, edge_order=1) / (2 * np.pi)
        with np.errstate(divide='ignore', invalid='ignore'):
            dp_offset1 = (
                -(II[itrc] * spacecraft_channels.dRR[itrc] - RR[itrc] * spacecraft_channels.dII[itrc])
                / (RR[itrc] ** 2 + II[itrc] ** 2)
                / (2 * np.pi)
            )
        dp_offset1[(II[itrc] == 0.0) & (RR[itrc] == 0.0)] = 0.0

        # ddp_offset0 = np.gradient(dp_offset0, T, edge_order=1)
        ddp_offset1 = np.gradient(dp_offset1, T, edge_order=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ddp_offset2 = (
                -2
                * (RR[itrc] * dII[itrc] - II[itrc] * dRR[itrc])
                * (II[itrc] * dII[itrc] + RR[itrc] * dRR[itrc])
                / (II[itrc] ** 2 + RR[itrc] ** 2) ** 2
                + (RR[itrc] * ddII[itrc] - II[itrc] * ddRR[itrc]) / (II[itrc] ** 2 + RR[itrc] ** 2)
            ) / (2 * np.pi)
        ddp_offset2[(II[itrc] == 0.0) & (RR[itrc] == 0.0)] = 0.0

        assert_allclose(
            np.sin(AET_waveform.PT[itrc] - p_offset0),
            np.sin(2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        assert_allclose(
            np.cos(AET_waveform.PT[itrc] - p_offset0),
            np.cos(2 * np.pi * (f_input + 1.0 / 2 * fp_input * T) * T),
            atol=1.0e-8,
            rtol=1.0e-8,
        )
        # assert_allclose(tdi_waveform.PT[itrc] - p_offset0, 2*np.pi*(f_input+1./2*fp_input*T)*T,
        #   atol=1.e-13*2*np.pi*f_input*T.max(),rtol=1.e-13)
        assert_allclose(
            AET_waveform.FT[itrc] - dp_offset1, f_input + fp_input * T, atol=1.0e-11 * f_input, rtol=1.0e-12,
        )
        assert_allclose(AET_waveform.FTd[itrc] - ddp_offset1, fp_input, atol=1.0e-7 * f_input / wc.DT, rtol=1.0e-5)

        # check that derivative of offset phases matches analytic expectation
        # assert_allclose(dp_offset0, dp_offset1, atol=1.e-8*f_input, rtol=1.e-5)
        # assert_allclose(ddp_offset0, ddp_offset1, atol=1.e-7*f_input/_wc.DT, rtol=1.e-5)
        # assert_allclose(ddp_offset1, ddp_offset2, atol=1.e-11*f_input/_wc.DT, rtol=1.e-12)

    # for itrc in range(_nc_waveform):
    #    # test the gradients respect expected rules
    #    #assert_allclose(np.gradient(tdi_waveform.PT[itrc], T, edge_order=2)/(2*np.pi),
    #       tdi_waveform.FT[itrc], atol=1.e-7*f_input,rtol=1.e-8)

    #    # same rules as the gradients, but test the integrals match too
    #    #assert_allclose(tdi_waveform.PT[itrc], 2*np.pi*cumtrapz(tdi_waveform.FT[itrc], T, initial=0.)
    #       +tdi_waveform.PT[itrc, 0], atol=1.e-8*f0_mult*_wc.DT, rtol=1.e-9*f0_mult*_wc.DT)
    #    #assert_allclose(np.cos(tdi_waveform.PT[itrc]),
    #        np.cos(2*np.pi*cumtrapz(tdi_waveform.FT[itrc], T, initial=0.)
    #        + tdi_waveform.PT[itrc, 0]), atol=1.e-8, rtol=1.e-9)


@pytest.mark.parametrize(
    'f0_mult',
    [
        0.5 / 128,
        1.0 / 128.0,
        1.1 / 128,
        1.5 / 128,
        2.0 / 128,
        2.5 / 128,
        3.0 / 128,
        4.0 / 128,
        0.01,
        0.1,
        1.0 / 4.0,
        1.0 / 4.0 + 0.5 / 128,
        0.3,
        1.0 / 2.0,
        1.0 / 2.0 + 0.5 / 128,
        0.95,
    ],
)
@pytest.mark.parametrize('f0p_mult', [-0.2, 0.0, 1.0e-5, 0.2, 0.9])
@pytest.mark.parametrize('rr_model', ['quad2', 'sin1', 'sin2', 'sin3', 'sin4', 'sin5', 'lin18', 'lin17', 'const'])
# @pytest.mark.skip()
def test_time_tdi_inplace_transform(f0_mult: float, rr_model: str, f0p_mult: float) -> None:
    """Test whether the signal computed in the time domain matches computing
    it in the wavelet domain and transforming to time.
    """
    toml_filename = 'tests/galactic_fit_test_config1.toml'

    with Path(toml_filename).open('rb') as f:
        config = tomllib.load(f)

    wc = get_wavelet_model(config)
    assert wc == wc_in
    lc = get_lisa_constants(config)
    er = EdgeRiseModel(-np.inf, np.inf)

    nt_loc = wc.Nt
    nc_waveform = lc.nc_waveform

    # Create fake input data for a pure sine wave
    p_input = np.pi / 2.
    f_input = wc.DF * wc.Nf * f0_mult
    fp_input = f0p_mult * f_input / wc.Tobs
    fpp_input = 0.
    amp_input = 1.0
    waveform, AET_waveform, _ = get_waveform_helper(
        p_input, f_input, fp_input, fpp_input, amp_input, nt_loc, wc.DT, nc_waveform, max_f=1 / (2 * wc.dt) - 1 / wc.Tobs,
    )
    waveform_fine, AET_waveform_fine, arg_cut_fine = get_waveform_helper(
        p_input, f_input, fp_input, fpp_input, amp_input, nt_loc * wc.Nf, wc.dt, nc_waveform, max_f=1 / (2 * wc.dt) - 1 / wc.Tobs,
    )
    T = waveform.T
    T_fine = waveform_fine.T

    # ensure the RRs and IIs are scaled diferently in different channels
    RR_scale_mult = np.array([0.3, 0.9, 0.7])
    II_scale_mult = np.array([0.3, 0.9, 0.7])

    RR_t_mult, II_t_mult = get_RR_t_mult(rr_model, nt_loc, 1, wc.DT)
    RR_t_mult_fine, II_t_mult_fine = get_RR_t_mult(rr_model, nt_loc, wc.Nf, wc.dt)

    RR = (np.ones((nc_waveform, nt_loc)).T * RR_scale_mult).T * RR_t_mult
    II = (np.ones((nc_waveform, nt_loc)).T * II_scale_mult).T * II_t_mult
    RR_fine = (np.ones((nc_waveform, nt_loc * wc.Nf)).T * RR_scale_mult).T * RR_t_mult_fine
    II_fine = (np.ones((nc_waveform, nt_loc * wc.Nf)).T * II_scale_mult).T * II_t_mult_fine
    dRR = np.zeros((nc_waveform, nt_loc))
    dII = np.zeros((nc_waveform, nt_loc))
    dRR_fine = np.zeros((nc_waveform, nt_loc * wc.Nf))
    dII_fine = np.zeros((nc_waveform, nt_loc * wc.Nf))
    spacecraft_channels = AntennaResponseChannels(T, RR, II, dRR, dII)
    spacecraft_channels_fine = AntennaResponseChannels(T_fine, RR_fine, II_fine, dRR_fine, dII_fine)
    nt_lim = PixelGenericRange(0, nt_loc, wc.DT, lc.t0)
    nt_lim_fine = PixelGenericRange(0, nt_loc * wc.Nf, wc.dt, lc.t0)

    # Call the function
    get_time_tdi_amp_phase(spacecraft_channels, AET_waveform, waveform, lc, er, nt_lim)
    get_time_tdi_amp_phase(spacecraft_channels_fine, AET_waveform_fine, waveform_fine, lc, er, nt_lim_fine)

    # get the sparse wavelet intrinsic_waveform
    wavelet_waveform = get_empty_sparse_taylor_time_waveform(lc.nc_waveform, wc)
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, lc.t0)
    wavemaket(wavelet_waveform, AET_waveform, nt_lim, wc, taylor_time_table, force_nulls=False)

    # get the dense wavelet intrinsic_waveform
    wavelet_dense: NDArray[np.floating] = np.zeros((nt_loc * wc.Nf, nc_waveform))
    sparse_addition_helper(wavelet_waveform, wavelet_dense)
    wavelet_dense = wavelet_dense.reshape((nt_loc, wc.Nf, nc_waveform))

    # get the time domain signal from the wavelets
    signal_time = np.zeros((nt_loc * wc.Nf, nc_waveform))
    for itrc in range(nc_waveform):
        signal_time[:, itrc] = inverse_wavelet_time(wavelet_dense[:, :, itrc], wc.Nf, nt_loc)

    # get the time domain signal from the fine sampling
    # signal should be pure cos
    signal_time_pred_cos = (AET_waveform_fine.AT * np.cos(AET_waveform_fine.PT)).T
    signal_time_pred_sin = (AET_waveform_fine.AT * np.sin(AET_waveform_fine.PT)).T

    # tukey window the signals to cut out edge artifacts
    signal_time_filter = signal_time.copy()

    for itrc in range(nc_waveform):
        # tukey window at start/end and when it goes out of band
        tukey(signal_time_filter[:, itrc], 0.05, arg_cut_fine)
        tukey(signal_time_pred_cos[:, itrc], 0.05, arg_cut_fine)
        tukey(signal_time_pred_sin[:, itrc], 0.05, arg_cut_fine)
        # cut off analysis when it is completely out of band
        if arg_cut_fine < nt_loc * wc.Nf:
            signal_time_filter[arg_cut_fine:, itrc] = 0.0
            signal_time_pred_cos[arg_cut_fine:, itrc] = 0.0
            signal_time_pred_sin[arg_cut_fine:, itrc] = 0.0

    # low pass filter to strip out highest frequency components we cannot represent properly currently
    b, a = butter(1, wc.DF * (wc.Nf - 1), fs=1.0 / wc.dt, btype='low', analog=False)
    for itrc in range(nc_waveform):
        signal_time_pred_cos[:, itrc] = filtfilt(b, a, signal_time_pred_cos[:, itrc])
        signal_time_pred_sin[:, itrc] = filtfilt(b, a, signal_time_pred_sin[:, itrc])
        signal_time_filter[:, itrc] = filtfilt(b, a, signal_time_filter[:, itrc])

    match_cos = np.sum(signal_time_filter * signal_time_pred_cos) / (
        np.linalg.norm(signal_time_filter) * np.linalg.norm(signal_time_pred_cos)
    )
    match_sin = np.sum(signal_time_filter * signal_time_pred_sin) / (
        np.linalg.norm(signal_time_filter) * np.linalg.norm(signal_time_pred_sin)
    )
    resid = np.sum((signal_time_filter - signal_time_pred_cos) ** 2) / (
        np.linalg.norm(signal_time_filter) * np.linalg.norm(signal_time_pred_cos)
    )
    print(f_input)
    print(match_cos)
    print(match_sin)
    print(resid)
    assert resid < 5.0e-3
    assert match_cos > 0.997
    assert match_sin < 1.0e-2
