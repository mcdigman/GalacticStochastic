"""unit tests for code in galactic_fit_helpers.py"""

from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage
import tomllib
from scipy.interpolate import InterpolatedUnivariateSpline

import GalacticStochastic.global_const as gc
from GalacticStochastic.galactic_fit_helpers import filter_periods_fft, get_S_cyclo
from WaveletWaveforms.wdm_config import get_wavelet_model

toml_filename = 'tests/galactic_fit_test_config1.toml'

# we  can use the same baise noise for most things and modulate it as necessary
with Path(toml_filename).open('rb') as f:
    config = tomllib.load(f)

wc = get_wavelet_model(config)

nc_galaxy = 3

seed = 3141592
rng_in = np.random.default_rng(seed)
bg_base = rng_in.normal(0.0, 1.0, (wc.Nt, wc.Nf, nc_galaxy))


@pytest.mark.parametrize('sign_high', [1, -1])
@pytest.mark.parametrize('sign_low', [1, -1])
def test_filter_periods_fft_full(sign_low, sign_high) -> None:
    """Test filtering max period with ffts for fixed fourier amplitude"""
    rng = np.random.default_rng(442)

    period_list = tuple(
        np.arange(
            0, np.int64(gc.SECSYEAR // wc.DT) // 2 + 1 / int(wc.Tobs / gc.SECSYEAR), 1 / int(wc.Tobs / gc.SECSYEAR),
        ),
    )

    amp_exp = np.zeros((len(period_list), nc_galaxy))

    amp_exp[:, 0] = rng.uniform(0.0, 1.0, len(period_list))
    amp_exp[:, 1] = rng.uniform(0.0, 1.0, len(period_list))
    amp_exp[:, 2] = rng.uniform(0.0, 1.0, len(period_list))

    angle_exp = np.zeros((len(period_list), nc_galaxy))

    angle_exp[:, 0] = rng.uniform(0.0, 2 * np.pi, len(period_list))
    angle_exp[:, 1] = rng.uniform(0.0, 2 * np.pi, len(period_list))
    angle_exp[:, 2] = rng.uniform(0.0, 2 * np.pi, len(period_list))

    ts = np.arange(0, wc.Nt) * wc.DT
    xs = np.zeros((wc.Nt, nc_galaxy))
    for itrc in range(nc_galaxy):
        for itrp, period_loc in enumerate(period_list):
            if period_loc == 0.0:
                angle_exp[itrp, itrc] = 0.0
                amp_exp[itrp, itrc] *= sign_low
                xs[:, itrc] += amp_exp[itrp, itrc]
            elif period_loc == np.int64(gc.SECSYEAR // wc.DT) // 2:
                angle_exp[itrp, itrc] = sign_high * np.pi / 4.0
                xs[:, itrc] += amp_exp[itrp, itrc] * np.cos(
                    2 * np.pi / gc.SECSYEAR * period_loc * ts - angle_exp[itrp, itrc],
                )
            else:
                xs[:, itrc] += amp_exp[itrp, itrc] * np.cos(
                    2 * np.pi / gc.SECSYEAR * period_loc * ts - angle_exp[itrp, itrc],
                )

    xs += 1.0

    r_fft1, amp_got, angle_got = filter_periods_fft(xs, wc.Nt, period_list, wc)

    assert np.allclose(amp_got, amp_exp, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(
        (angle_got[:-1] - angle_exp[:-1] + np.pi) % (2 * np.pi) + angle_exp[:-1] - np.pi,
        angle_exp[:-1],
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert np.allclose(angle_got[-1], np.pi / 4.0)

    assert np.allclose(r_fft1, xs, atol=1.0e-10, rtol=1.0e-10)


def test_filter_periods_fft_full2() -> None:
    """Test filtering max period with ffts for random data"""
    rng = np.random.default_rng(442)

    period_list = tuple(
        np.arange(
            0, np.int64(gc.SECSYEAR // wc.DT) // 2 + 1 / int(wc.Tobs / gc.SECSYEAR), 1 / int(wc.Tobs / gc.SECSYEAR),
        ),
    )

    xs = rng.normal(0.0, 1.0, (wc.Nt, nc_galaxy))

    xs += 1.0

    r_fft1, _, _ = filter_periods_fft(xs, wc.Nt, period_list, wc)

    assert np.allclose(r_fft1, xs, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize('itrk', [0.25, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0, 15.0, 16.0, 17.0])
def test_filter_periods_fft1(itrk) -> None:
    """Test filtering 1 period with ffts"""
    period_list = (itrk,)
    amp_exp = np.array([[0.1, 0.2, 1.0]])
    angle_exp = np.array([[0.6, 0.1, 0.3]])
    ts = np.arange(0, wc.Nt) * wc.DT
    xs = np.zeros((wc.Nt, nc_galaxy))
    xs[:, 0] = amp_exp[0, 0] * np.cos(2 * np.pi / gc.SECSYEAR * period_list[0] * ts - angle_exp[0, 0])
    xs[:, 1] = amp_exp[0, 1] * np.cos(2 * np.pi / gc.SECSYEAR * period_list[0] * ts - angle_exp[0, 1])
    xs[:, 2] = amp_exp[0, 2] * np.cos(2 * np.pi / gc.SECSYEAR * period_list[0] * ts - angle_exp[0, 2])

    xs += 1.0

    if np.abs(np.int64(wc.Tobs / gc.SECSYEAR * itrk) - wc.Tobs / gc.SECSYEAR * itrk) > 0.01:
        with pytest.warns(UserWarning, match='fft filtering expects periods to be integer fraction of total time:'):
            filter_periods_fft(xs, wc.Nt, period_list, wc)
        return

    r_fft1, amp_got, _ = filter_periods_fft(xs, wc.Nt, period_list, wc)

    assert np.allclose(amp_got, amp_exp, atol=1.0e-12, rtol=1.0e-12)
    assert np.allclose(angle_exp, angle_exp, atol=1.0e-12, rtol=1.0e-12)

    assert np.allclose(r_fft1, xs, atol=1.0e-12, rtol=1.0e-12)


def test_stationary_mean_scramble_invariance() -> None:
    """S for stationary mean should be independent of time order of the samples; check this is true"""
    # get the background
    bg_here1 = 1000 * bg_base.copy()

    # get scrambled time indices
    idx_sel2 = np.arange(0, wc.Nt)

    rng = np.random.default_rng(442)
    rng.shuffle(idx_sel2)

    # get the same background with time indices scrambled
    bg_here2 = bg_here1[idx_sel2].copy()

    S_inst_m = np.full((wc.Nf, nc_galaxy), 1.0)

    # get both S matrices
    S_got1, _, _, _, _ = get_S_cyclo(bg_here1, S_inst_m, wc, 1.0, 0, period_list=())
    S_got2, _, _, _, _ = get_S_cyclo(bg_here2, S_inst_m, wc, 1.0, 0, period_list=())

    # check for expected invariance
    assert np.allclose(S_got1, S_got2, atol=1.0e-14, rtol=1.0e-13)


def get_noise_model_helper(model_name):
    """Helper to get some useful noise model multipliers given a name"""
    if model_name == 'powerlaw1':
        f_mult = ((np.arange(0, wc.Nf) + 1) / wc.Nf) ** 2
    elif model_name == 'powerlaw2':
        f_mult = ((np.arange(0, wc.Nf) + 1) / wc.Nf) ** -2
    elif model_name == 'white_faint':
        f_mult = np.full(wc.Nf, 1.0e-3)
    elif model_name == 'white_equal':
        f_mult = np.full(wc.Nf, 1.0)
    elif model_name == 'white_bright':
        f_mult = np.full(wc.Nf, 1000.0)
    elif model_name == 'sin1':
        f_mult = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(0, wc.Nf) / 10.0)
    elif model_name == 'sin2':
        f_mult = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(0, wc.Nf) / 100.0)
    elif model_name == 'sin3':
        f_mult = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(0, wc.Nf) / 2.0)
    elif model_name == 'dirac1':
        f_mult = np.full(wc.Nf, 1.0e-3)
        f_mult[0] = 1.0e4
    elif model_name == 'dirac2':
        f_mult = np.full(wc.Nf, 1.0e-3)
        f_mult[wc.Nf // 2] = 1.0e4
    else:
        msg = 'unrecognized option for bg model'
        raise ValueError(msg)
    return f_mult


def stationary_mean_smooth_helper(bg_models, noise_models, smooth_lengthf, filter_periods) -> None:
    """Helper to test stationary mean with several lengths of spectral smoothing
    can reproduce injected input spectrum
    """
    f_mult = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        f_mult[:, itrc] = get_noise_model_helper(bg_models[itrc])

    bg_here = bg_base.copy()

    for itrf in range(wc.Nf):
        bg_here[:, itrf, :] *= f_mult[itrf]

    S_inst_m = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        S_inst_m[:, itrc] = get_noise_model_helper(noise_models[itrc])

    S_got, _, _, _, _ = get_S_cyclo(bg_here, S_inst_m, wc, smooth_lengthf, filter_periods, period_list=())

    # replicate expected smoothed multiplier
    f_mult_smooth = np.zeros_like(f_mult)
    interp_mult = 10
    n_f_interp = interp_mult * wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF * (wc.Nf - 1)), n_f_interp)
    log_fs = np.log10(np.arange(1, wc.Nf) * wc.DF)
    for itrc in range(nc_galaxy):
        log_f_mult_interp = InterpolatedUnivariateSpline(log_fs, np.log10(f_mult[1:, itrc] ** 2 + 1.0e-50), k=3, ext=2)(
            log_fs_interp,
        )

        log_f_mult_smooth_interp = scipy.ndimage.gaussian_filter(log_f_mult_interp, smooth_lengthf * interp_mult)
        f_mult_smooth[:, itrc] = np.hstack(
            [
                f_mult[0, itrc],
                np.sqrt(
                    10 ** InterpolatedUnivariateSpline(log_fs_interp, log_f_mult_smooth_interp, k=3, ext=2)(log_fs)
                    - 1.0e-50,
                ),
            ],
        )

    for itrc in range(nc_galaxy):
        # check that in constant model the resulting spectrum is indeed constant
        assert np.all(S_got[:, :, itrc] == S_got[0, :, itrc])

    # check that adding known noise produces known spectrum
    for itrc in range(nc_galaxy):
        # check no rows outside ~5 sigma of being consistent with expected result
        for itrf in range(wc.Nf):
            got_loc = S_got[0, itrf, itrc]
            pred_loc = S_inst_m[itrf, itrc] + f_mult_smooth[itrf, itrc] ** 2
            assert np.allclose(
                got_loc,
                pred_loc,
                atol=5 * (f_mult_smooth[itrf, itrc] ** 2) / np.sqrt(wc.Nt),
                rtol=5 * (f_mult_smooth[itrf, itrc] ** 2) / np.sqrt(wc.Nt),
            )


@pytest.mark.parametrize('amp2_mult', [0.0, 0.09, 0.11, 0.5, 0.9, 1.0, 2.0])
def test_nonstationary_mean_faint_alternate(amp2_mult) -> None:
    """Test a case where there is a faint alternate frequency with a different periodicity;
    the fainter periodicity should be ignored completely
    """
    smooth_lengthf = 1.0
    filter_periods = 1

    # input periods, amplitudes, phases
    itrk1 = 2
    amp1 = 0.3
    phase1 = 0.1

    itrk2 = 3
    amp2 = amp1 * np.sqrt(amp2_mult)
    phase2 = 0.7

    period_list = tuple(
        np.arange(
            0, np.int64(gc.SECSYEAR // wc.DT) // 2 + 1 / int(wc.Tobs / gc.SECSYEAR), 1 / int(wc.Tobs / gc.SECSYEAR),
        ),
    )

    f_mult1 = np.full((wc.Nf, nc_galaxy), 0.0)
    f_mult2 = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        fs = np.arange(0, wc.Nf)
        f_mult1[:, itrc] = np.exp(-((fs - wc.Nf / 4) ** 2) / (2 * (wc.Nf / 32)))
        f_mult2[:, itrc] = np.exp(-((fs - 3 * wc.Nf / 4) ** 2) / (2 * (wc.Nf / 32)))

    ts = np.arange(0, wc.Nt) * wc.DT
    t_mult1 = np.full((wc.Nt, nc_galaxy), 0.0)
    t_mult2 = np.full((wc.Nt, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        t_mult1[:, itrc] += amp1 * np.cos(2 * np.pi / gc.SECSYEAR * ts * itrk1 - phase1)
        t_mult2[:, itrc] += amp2 * np.cos(2 * np.pi / gc.SECSYEAR * ts * itrk2 - phase2)

    bg_here = np.full_like(bg_base, 1.0)

    for itrc in range(nc_galaxy):
        bg_here[:, :, itrc] *= np.outer(t_mult1[:, itrc], f_mult1[:, itrc]) + np.outer(
            t_mult2[:, itrc], f_mult2[:, itrc],
        )

    S_inst_m = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        S_inst_m[:, itrc] = get_noise_model_helper('white_equal')

    _, _, _, amp_got, _ = get_S_cyclo(bg_here, S_inst_m, wc, smooth_lengthf, filter_periods, period_list=None)

    _, amp_got1, _ = filter_periods_fft(t_mult1**2 + 1.0, wc.Nt, period_list, wc)

    if amp2**2 < 0.1 * amp1**2:
        assert np.allclose(amp_got[2 * int(wc.Tobs / gc.SECSYEAR) * itrk1, :], 1.0, atol=1.0e-10, rtol=1.0e-10)
        assert np.allclose(amp_got[int(wc.Tobs / gc.SECSYEAR) * itrk1, :], 0.0, atol=1.0e-10, rtol=1.0e-10)
        assert np.allclose(amp_got[2 * int(wc.Tobs / gc.SECSYEAR) * itrk2, :], 0.0, atol=1.0e-10, rtol=1.0e-10)
        assert np.allclose(amp_got[int(wc.Tobs / gc.SECSYEAR) * itrk2, :], 0.0, atol=1.0e-10, rtol=1.0e-10)
    else:
        assert not np.allclose(amp_got[2 * int(wc.Tobs / gc.SECSYEAR) * itrk1, :], 1.0, atol=1.0e-10, rtol=1.0e-10)
        assert np.allclose(amp_got[int(wc.Tobs / gc.SECSYEAR) * itrk1, :], 0.0, atol=1.0e-10, rtol=1.0e-10)
        assert not np.allclose(amp_got[2 * int(wc.Tobs / gc.SECSYEAR) * itrk2, :], 0.0, atol=1.0e-10, rtol=1.0e-10)
        assert np.allclose(amp_got[int(wc.Tobs / gc.SECSYEAR) * itrk2, :], 0.0, atol=1.0e-10, rtol=1.0e-10)

    assert np.allclose(
        amp_got[2 * int(wc.Tobs / gc.SECSYEAR) * itrk1, :] + amp_got[2 * int(wc.Tobs / gc.SECSYEAR) * itrk2, :],
        1.0,
        atol=1.0e-10,
        rtol=1.0e-10,
    )

    if amp1 > amp2:
        for itrc in range(nc_galaxy):
            assert np.argmax(amp_got[:, itrc]) == 2 * int(wc.Tobs / gc.SECSYEAR) * itrk1
            assert np.argmax(amp_got1[1:, itrc]) == np.argmax(amp_got[1:, itrc])
    elif amp1 < amp2:
        for itrc in range(nc_galaxy):
            assert np.argmax(amp_got[:, itrc]) == 2 * int(wc.Tobs / gc.SECSYEAR) * itrk2
    else:
        for itrc in range(nc_galaxy):
            assert np.argmax(amp_got[:, itrc]) == 2 * int(wc.Tobs / gc.SECSYEAR) * min(itrk1, itrk2)


def test_nonstationary_mean_zero_case() -> None:
    """Test a case where rec_use is very small/negative for numerical stability/ensure S cannot be nan"""
    smooth_lengthf = 1.0
    filter_periods = 1

    f_mult = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        f_mult[:, itrc] = get_noise_model_helper('white_bright')

    ts = np.arange(0, wc.Nt) * wc.DT
    t_mult = np.full((wc.Nt, nc_galaxy), 1.0)
    for itrc in range(nc_galaxy):
        t_mult[:, itrc] = np.exp(-((ts - gc.SECSYEAR / 2) ** 2) / (2 * (0.05 * gc.SECSYEAR) ** 2))

    bg_here = bg_base.copy()

    for itrf in range(wc.Nf):
        bg_here[:, itrf, :] *= f_mult[itrf]

    for itrt in range(wc.Nt):
        bg_here[itrt, :, :] *= t_mult[itrt, :]

    S_inst_m = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        S_inst_m[:, itrc] = get_noise_model_helper('white_faint')

    S_got, rec_got, _, _, _ = get_S_cyclo(bg_here, S_inst_m, wc, smooth_lengthf, filter_periods, period_list=None)

    assert np.all(rec_got > 0.0)

    # replicate expected smoothed multiplier
    f_mult_smooth = np.zeros_like(f_mult)
    interp_mult = 10
    n_f_interp = interp_mult * wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF * (wc.Nf - 1)), n_f_interp)
    log_fs = np.log10(np.arange(1, wc.Nf) * wc.DF)
    for itrc in range(nc_galaxy):
        log_f_mult_interp = InterpolatedUnivariateSpline(log_fs, np.log10(f_mult[1:, itrc] ** 2 + 1.0e-50), k=3, ext=2)(
            log_fs_interp,
        )

        log_f_mult_smooth_interp = scipy.ndimage.gaussian_filter(log_f_mult_interp, smooth_lengthf * interp_mult)
        f_mult_smooth[:, itrc] = np.hstack(
            [
                f_mult[0, itrc],
                np.sqrt(
                    10 ** InterpolatedUnivariateSpline(log_fs_interp, log_f_mult_smooth_interp, k=3, ext=2)(log_fs)
                    - 1.0e-50,
                ),
            ],
        )

    # check that adding known noise produces known spectrum
    for itrc in range(2):
        # check no rows outside ~5 sigma of being consistent with expected result
        for itrf in range(wc.Nf):
            got_loc = S_got[:, itrf, itrc]
            bg_loc = f_mult_smooth[itrf, itrc] ** 2 * t_mult[:, itrc] ** 2
            pred_loc = S_inst_m[itrf, itrc] + bg_loc
            assert np.allclose(
                got_loc,
                pred_loc,
                atol=5 * (f_mult_smooth[itrf, itrc] ** 2) / np.sqrt(wc.Nt),
                rtol=5 * (bg_loc) / np.sqrt(wc.Nt),
            )


def nonstationary_mean_smooth_helper(
    bg_models, noise_models, smooth_lengthf, filter_periods, itrk1, amp1, phase1,
) -> None:
    """Helper to test stationary mean with several lengths of spectral smoothing
    can reproduce injected input spectrum
    """
    # input periods, amplitudes, phases
    period_list1 = (itrk1,)
    amp_list1 = np.array([amp1])
    phase_list1 = np.array([phase1])

    # periods, amplitudes, phases to record
    period_list2 = (0, itrk1, 2 * itrk1)

    # expected results after processing
    # note that the expected results contains a harmonic because t_mult is squared
    # the values can be obtained from the double angle formula
    amp_exp = np.array([0.0, 2 * amp1 / (1 + amp1**2 / 2), 1 / 2 * amp1**2 / (1 + amp1**2 / 2)])
    phase_exp = np.array([0.0, phase1, 2 * phase1])

    f_mult = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        f_mult[:, itrc] = get_noise_model_helper(bg_models[itrc])

    ts = np.arange(0, wc.Nt) * wc.DT
    t_mult = np.full((wc.Nt, nc_galaxy), 1.0)
    for itrc in range(nc_galaxy):
        for itrp, period_loc in enumerate(period_list1):
            t_mult[:, itrc] += amp_list1[itrp] * np.cos(2 * np.pi / gc.SECSYEAR * ts * period_loc - phase_list1[itrp])

    # checks for closeness can be much stricter if bg_here is 1, but may not cover all variations
    bg_here = bg_base.copy()

    for itrf in range(wc.Nf):
        bg_here[:, itrf, :] *= f_mult[itrf]

    for itrt in range(wc.Nt):
        bg_here[itrt, :, :] *= t_mult[itrt, :]

    S_inst_m = np.full((wc.Nf, nc_galaxy), 0.0)
    for itrc in range(nc_galaxy):
        S_inst_m[:, itrc] = get_noise_model_helper(noise_models[itrc])

    S_got, rec_got, _, amp_got, angle_got = get_S_cyclo(
        bg_here, S_inst_m, wc, smooth_lengthf, filter_periods, period_list=period_list2,
    )

    assert np.all(rec_got > 0.0)

    for itrc in range(nc_galaxy):
        for itrp in range(len(period_list1)):
            assert np.isclose(amp_got[itrp, itrc], amp_exp[itrp], atol=1.0e-2, rtol=1.0e-1)
            assert np.isclose(
                (angle_got[itrp, itrc] - phase_exp[itrp] + np.pi) % (2 * np.pi) + phase_exp[itrp] - np.pi,
                phase_exp[itrp],
                atol=1.0e-2 / (amp_got[itrp, itrc] + 0.001),
                rtol=1.0e-1,
            )

    # replicate expected smoothed multiplier
    f_mult_smooth = np.zeros_like(f_mult)
    interp_mult = 10
    n_f_interp = interp_mult * wc.Nf
    log_fs_interp = np.linspace(np.log10(wc.DF), np.log10(wc.DF * (wc.Nf - 1)), n_f_interp)
    log_fs = np.log10(np.arange(1, wc.Nf) * wc.DF)
    for itrc in range(nc_galaxy):
        log_f_mult_interp = InterpolatedUnivariateSpline(log_fs, np.log10(f_mult[1:, itrc] ** 2 + 1.0e-50), k=3, ext=2)(
            log_fs_interp,
        )

        log_f_mult_smooth_interp = scipy.ndimage.gaussian_filter(log_f_mult_interp, smooth_lengthf * interp_mult)
        f_mult_smooth[:, itrc] = np.hstack(
            [
                f_mult[0, itrc],
                np.sqrt(
                    10 ** InterpolatedUnivariateSpline(log_fs_interp, log_f_mult_smooth_interp, k=3, ext=2)(log_fs)
                    - 1.0e-50,
                ),
            ],
        )

    # check that adding known noise produces known spectrum
    for itrc in range(nc_galaxy):
        # check no rows outside ~5 sigma of being consistent with expected result
        for itrf in range(wc.Nf):
            got_loc = S_got[:, itrf, itrc]
            bg_loc = f_mult_smooth[itrf, itrc] ** 2 * t_mult[:, itrc] ** 2
            pred_loc = S_inst_m[itrf, itrc] + bg_loc
            assert np.allclose(
                got_loc,
                pred_loc,
                atol=5 * (f_mult_smooth[itrf, itrc] ** 2) / np.sqrt(wc.Nt),
                rtol=5 * (bg_loc) / np.sqrt(wc.Nt),
            )


@pytest.mark.parametrize('bg_model', ['powerlaw1', 'sin1', 'powerlaw2'])
@pytest.mark.parametrize('noise_model', ['sin1', 'sin2', 'sin3'])
@pytest.mark.parametrize('itrk', [16])
@pytest.mark.parametrize('phase', [0.2])
def test_nonstationary_bg_power_bg_slope(bg_model, noise_model, itrk, phase) -> None:
    """Test that smoothed time varying spectrum with different brightnesses constant noise model
    produces expected results
    """
    nonstationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, True, itrk, 0.2, phase,
    )


@pytest.mark.parametrize('bg_model', ['white_faint', 'white_equal', 'white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [16])
@pytest.mark.parametrize('phase', [0.2])
def test_nonstationary_bg_power_bg_brightness(bg_model, noise_model, itrk, phase) -> None:
    """Test that smoothed time varying spectrum with different brightnesses constant noise model
    produces expected results
    """
    nonstationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, True, itrk, 0.2, phase,
    )


@pytest.mark.parametrize('bg_model', ['white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [2])
@pytest.mark.parametrize('phase', [0.0, 0.3])
@pytest.mark.parametrize('amp', [0.0, 0.1, 0.2, 0.4, 0.5, 0.8, 0.999, 1.0, 1.2])
def test_nonstationary_bg_power_bg_amp(bg_model, noise_model, itrk, phase, amp) -> None:
    """Test that smoothed time varying spectrum with different modulation amplitudes constant noise model
    produces expected results
    """
    nonstationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, True, itrk, amp, phase,
    )


@pytest.mark.parametrize('bg_model', ['white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [1, 2, 3, 4, 5, 16, 32, 64, 127, 128, 129])
@pytest.mark.parametrize('phase', [0.7])
def test_nonstationary_bg_power_harmonic(bg_model, noise_model, itrk, phase) -> None:
    """Test that smoothed time varying spectrum with constant noise model
    produces expected results with different known injected time variation
    """
    nonstationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, True, itrk, 0.2, phase,
    )


@pytest.mark.parametrize('bg_model', ['white_equal'])
@pytest.mark.parametrize('noise_model', ['white_faint', 'white_equal', 'white_bright'])
@pytest.mark.parametrize('itrk', [16])
@pytest.mark.parametrize('phase', [0.2])
def test_nonstationary_bg_power_noise_brightness(bg_model, noise_model, itrk, phase) -> None:
    """Test that smoothed time varying spectrum with different noises brightnesses constant background brightness
    produces expected results
    """
    nonstationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, True, itrk, 0.2, phase,
    )


@pytest.mark.parametrize('bg_model', ['white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('itrk', [1, 3, 16])
@pytest.mark.parametrize(
    'phase',
    [
        0.0,
        0.2,
        0.3,
        np.pi / 2 - 0.01,
        np.pi / 2.0,
        np.pi / 2 + 0.01,
        np.pi - 0.01,
        np.pi,
        np.pi + 0.01,
        3 * np.pi / 2 - 0.01,
        3 * np.pi / 2.0,
        3 * np.pi / 2 + 0.01,
        2 * np.pi - 0.01,
        2 * np.pi,
        2 * np.pi + 0.01,
    ],
)
def test_nonstationary_bg_power_phase(bg_model, noise_model, itrk, phase) -> None:
    """Test that smoothed time varying spectrum with constant noise model produces expected results
    with different phases
    """
    nonstationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, True, itrk, 0.2, phase,
    )


def test_different_bg_spectra() -> None:
    """Test that smoothed time invariant spectrum produces expected results
    if background spectrum differs between channels
    """
    stationary_mean_smooth_helper(
        ['powerlaw1', 'white_bright', 'powerlaw2'], ['white_equal', 'white_equal', 'white_equal'], 1.0, False,
    )


def test_different_noise_spectra() -> None:
    """Test that smoothed time invariant spectrum produces expected results
    if noise spectrum differs between channels
    """
    stationary_mean_smooth_helper(
        ['white_equal', 'white_equal', 'white_equal'], ['powerlaw1', 'white_equal', 'powerlaw2'], 1.0, False,
    )


def test_different_noise_bg_spectra() -> None:
    """Test that smoothed time invariant spectrum produce expected results
    if noise and background spectra differs between channels
    """
    stationary_mean_smooth_helper(['powerlaw1', 'sin1', 'powerlaw2'], ['sin1', 'sin2', 'sin3'], 1.0, False)


@pytest.mark.parametrize('bg_model', ['white_faint', 'white_equal', 'white_bright'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
def test_stationary_bg_power(bg_model, noise_model) -> None:
    """Test that smoothed time invariant spectrum with constant noise model produces expected results
    with different background noise powers
    """
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, False)


@pytest.mark.parametrize('bg_model', ['white_equal'])
@pytest.mark.parametrize('noise_model', ['white_faint', 'white_equal', 'white_bright'])
def test_stationary_noise_power(bg_model, noise_model) -> None:
    """Test that smoothed time invariant spectrum with constant noise model produces expected results
    with different instrument noise powers
    """
    stationary_mean_smooth_helper([bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], 1.0, False)


@pytest.mark.parametrize(
    'bg_model', ['sin1', 'sin2', 'sin3', 'powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'])
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('smooth_lengthf', [1.0])
def test_stationary_filter_bg_(bg_model, noise_model, smooth_lengthf) -> None:
    """Test nothing unexpected happens if filter_periods is true but the noise model has no harmonics"""
    stationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], smooth_lengthf, True,
    )


@pytest.mark.parametrize(
    'bg_model', ['sin1', 'sin2', 'sin3', 'powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'],
)
@pytest.mark.parametrize('noise_model', ['white_equal'])
@pytest.mark.parametrize('smooth_lengthf', [0.1, 1.0, 10.0, 100.0])
def test_stationary_mean_bg_smooth(bg_model, noise_model, smooth_lengthf) -> None:
    """Test that smoothed time invariant spectrum with constant instrument noise model produces expected results
    with different smoothing lengths
    """
    stationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], smooth_lengthf, False,
    )


@pytest.mark.parametrize('bg_model', ['sin2', 'sin3', 'powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'])
@pytest.mark.parametrize(
    'noise_model', ['sin1', 'sin2', 'sin3', 'powerlaw1', 'powerlaw2', 'white_equal', 'dirac1', 'dirac2'],
)
@pytest.mark.parametrize('smooth_lengthf', [1.0, 100.0])
def test_stationary_mean__instrument_smooth(bg_model, noise_model, smooth_lengthf) -> None:
    """Test that smoothed time invariant spectrum with different pairs of instrument and bg noise model
    produces expected results
    """
    stationary_mean_smooth_helper(
        [bg_model, bg_model, bg_model], [noise_model, noise_model, noise_model], smooth_lengthf, False,
    )
