"""Tests for the dense-stripe Taylor-time wavelet coaddition functions.

The numerical oracle for every correctness test is the existing
:func:`WaveletWaveforms.taylor_time_wavelet_funcs.wavemaket` called with
``force_nulls=0`` and ``amplitude_order=0``, decoded to a dense
``(wc.Nt, wc.Nf, Nc)`` array with
:func:`WaveletWaveforms.sparse_waveform_functions.wavelet_sparse_to_dense` and
sliced to the stripe window. The oracle is independent of the implementation
under test.

A small amount of the oracle's index arithmetic is re-derived in
``_classify_stripe_drops`` and ``_aligned_padding_only_stripe``, but only to
*locate* which cells the oracle drops (bandwidth vs table-overflow) and to
demonstrate the rejected padding-only behavior. The expected *values* always
come from ``wavemaket`` + ``wavelet_sparse_to_dense``.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import tomllib
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, wavelet_sparse_to_dense
from WaveletWaveforms.taylor_time_coefficients import (
    WaveletTaylorTimeCoeffs,
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket
from WaveletWaveforms.taylor_time_wavelet_optimized import (
    WaveletTaylorTimeCoeffsAligned,
    build_aligned_taylor_time_table,
    wavemaket_stripe_dense,
    wavemaket_stripe_dense_aligned,
    wavemaket_stripe_sparse,
)
from WaveletWaveforms.wdm_config import WDMWaveletConstants, get_wavelet_model

# Required-coverage initial-frequency offsets from the stripe center, in full-band
# frequency-pixel units. Includes the on-each-side-of-halfway positions -0.3, -0.25,
# 0.25, 0.3 and the +/-0.5 / 0.0 endpoints.
OFFSETS: tuple[float, ...] = (-0.5, -0.3, -0.25, 0.0, 0.25, 0.3, 0.5)

# Required-coverage linear slopes in full-band frequency pixels per observation period.
SLOPES: tuple[float, ...] = (-2.0, -1.99, -1.0, -1e-3, -1e-10, 0.0, 1e-10, 1e-3, 1.0, 1.99, 2.0)

# (nf_start, stripe_height) cases covering stripe heights 2,3,4,5, even and odd
# nf_start, and a stripe whose top edge sits exactly at wc.Nf (256).
STRIPE_CASES: tuple[tuple[int, int], ...] = ((100, 2), (101, 3), (100, 4), (151, 5), (252, 4), (253, 3))


class _Setup(NamedTuple):
    """Bundle of wavelet constants and Taylor tables shared across tests."""

    wc: WDMWaveletConstants
    table: WaveletTaylorTimeCoeffs
    aligned: WaveletTaylorTimeCoeffsAligned


@pytest.fixture(scope='module')
def setup1() -> _Setup:
    """Load the config1 wavelet constants and the original and aligned Taylor tables."""
    with Path('tests/wavemaket_test_config1.toml').open('rb') as f:
        config = tomllib.load(f)
    wc = get_wavelet_model(config)
    table = get_taylor_table_time(wc, cache_mode='check', output_mode='skip', grid_check_mode=0)
    aligned = build_aligned_taylor_time_table(table, wc)
    return _Setup(wc, table, aligned)


def _build_linear_waveform(
    wc: WDMWaveletConstants, f0: float, ftd_const: float, nc: int
) -> StationaryWaveformTime:
    """Build a waveform with linear frequency f0 + ftd_const * t and a positive amplitude."""
    nt = wc.Nt
    t_grid = np.arange(nt) * wc.DT
    pt = np.zeros((nc, nt))
    ft = np.zeros((nc, nt))
    ftd = np.zeros((nc, nt))
    at = np.zeros((nc, nt))
    for itrc in range(nc):
        ft[itrc] = f0 + ftd_const * t_grid
        ftd[itrc] = ftd_const
        pt[itrc] = 2 * np.pi * (f0 * t_grid + 0.5 * ftd_const * t_grid**2) + 0.3 * itrc
        at[itrc] = (1.0 + 0.2 * itrc) * (1.0 + 0.1 * np.sin(2 * np.pi * t_grid / wc.Tobs))
    return StationaryWaveformTime(t_grid, pt, ft, ftd, at)


def _waveform_from_offset_slope(
    wc: WDMWaveletConstants, nf_start: int, stripe_height: int, offset_pix: float, slope_pix: float, nc: int
) -> StationaryWaveformTime:
    """Build a supported-domain waveform from a stripe-center offset and a per-observation slope."""
    stripe_center_layer = nf_start + (stripe_height - 1) / 2.0
    f0 = (stripe_center_layer + offset_pix) * wc.DF
    ftd_const = slope_pix * wc.DF / wc.Tobs
    return _build_linear_waveform(wc, f0, ftd_const, nc)


def _oracle_stripe(
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    wc: WDMWaveletConstants,
    table: WaveletTaylorTimeCoeffs,
) -> NDArray[np.floating]:
    """Compute the reference stripe slice from wavemaket + wavelet_sparse_to_dense."""
    nc = waveform.AT.shape[0]
    sparse = get_empty_sparse_taylor_time_waveform(nc, wc)
    wavemaket(sparse, waveform, nt_lim, wc, table, force_nulls=0, amplitude_order=0)
    dense = wavelet_sparse_to_dense(sparse, wc)
    return np.ascontiguousarray(dense[:, nf_start:nf_start + stripe_height, :])


def _run_function(
    name: str,
    stripe: NDArray[np.floating],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    setup: _Setup,
) -> None:
    """Dispatch to the named stripe function, mutating ``stripe`` in place."""
    if name == 'sparse':
        wavemaket_stripe_sparse(stripe, waveform, nf_start, stripe_height, nt_lim, setup.wc, setup.table)
    elif name == 'dense':
        wavemaket_stripe_dense(stripe, waveform, nf_start, stripe_height, nt_lim, setup.wc, setup.table)
    elif name == 'aligned':
        wavemaket_stripe_dense_aligned(stripe, waveform, nf_start, stripe_height, nt_lim, setup.wc, setup.aligned)
    else:
        msg = f'unknown function {name}'
        raise ValueError(msg)


def _amplitude_source(waveform: StationaryWaveformTime, nt_lim: PixelGenericRange) -> float:
    """Maximum absolute amplitude over all channels and the selected time samples."""
    return float(np.max(np.abs(waveform.AT[:, nt_lim.nx_min:nt_lim.nx_max])))


@pytest.mark.parametrize(('nf_start', 'stripe_height'), STRIPE_CASES)
@pytest.mark.parametrize('offset_pix', OFFSETS)
def test_stripe_matches_oracle(setup1: _Setup, nf_start: int, stripe_height: int, offset_pix: float) -> None:
    """All three functions reproduce the full oracle stripe and agree pairwise.

    Verifies TR7-TR10 (oracle match over the entire stripe including zero cells)
    and TR3/R3 (pairwise agreement after independent calls) across stripe heights
    2-5, even/odd nf_start, a top-edge stripe, all required offsets, and all
    required slopes.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    for slope_pix in SLOPES:
        waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, offset_pix, slope_pix, nc)
        oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
        amp = _amplitude_source(waveform, nt_lim)
        assert amp > 0.0
        atol = 1e-10 * amp

        outputs: dict[str, NDArray[np.floating]] = {}
        for name in ('sparse', 'dense', 'aligned'):
            stripe = np.zeros((wc.Nt, stripe_height, nc))
            _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
            outputs[name] = stripe
            assert_allclose(
                stripe, oracle, atol=atol, rtol=1e-9,
                err_msg=f'{name} offset={offset_pix} slope={slope_pix} nf_start={nf_start} sh={stripe_height}',
            )

        # pairwise agreement among the three independent results
        assert_allclose(outputs['sparse'], outputs['dense'], atol=atol, rtol=1e-9)
        assert_allclose(outputs['sparse'], outputs['aligned'], atol=atol, rtol=1e-9)
        assert_allclose(outputs['dense'], outputs['aligned'], atol=atol, rtol=1e-9)


@pytest.mark.parametrize(('nf_start', 'stripe_height'), STRIPE_CASES)
def test_pixel_boundary_and_halfway(setup1: _Setup, nf_start: int, stripe_height: int) -> None:
    """Frequencies exactly on a pixel boundary and exactly halfway between boundaries.

    Verifies the on-boundary and halfway required-coverage cases (TR15) for all
    three functions against the oracle.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    k0 = nf_start + stripe_height // 2
    for f0 in (k0 * wc.DF, (k0 + 0.5) * wc.DF):
        waveform = _build_linear_waveform(wc, f0, 0.0, nc)
        oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
        atol = 1e-10 * _amplitude_source(waveform, nt_lim)
        # the source sits inside the stripe, so the oracle must place real power there
        assert np.count_nonzero(oracle) > 0
        for name in ('sparse', 'dense', 'aligned'):
            stripe = np.zeros((wc.Nt, stripe_height, nc))
            _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
            assert_allclose(stripe, oracle, atol=atol, rtol=1e-9, err_msg=f'{name} f0={f0}')


@pytest.mark.parametrize('layout', ['C', 'F'])
def test_nt_2048_matches_oracle(setup1: _Setup, layout: str) -> None:
    """A representative wc.Nt == 2048 case for C- and F-contiguous stripes.

    Verifies TR16 (representative Nt == 2048) and the contiguity-layout coverage.
    The Taylor table is independent of Nt, so the cached config1 table is reused
    with an Nt == 2048 wavelet-constants object.
    """
    wc2048 = setup1.wc._replace(Nt=2048)
    table = setup1.table
    aligned = setup1.aligned
    setup = _Setup(wc2048, table, aligned)
    nc = 3
    nt_lim = PixelGenericRange(0, wc2048.Nt, wc2048.DT, 0.0)
    nf_start, stripe_height = 100, 4
    for offset_pix, slope_pix in ((0.25, 1.0), (-0.3, -1.99), (0.0, 0.0)):
        waveform = _waveform_from_offset_slope(wc2048, nf_start, stripe_height, offset_pix, slope_pix, nc)
        oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc2048, table)
        atol = 1e-10 * _amplitude_source(waveform, nt_lim)
        for name in ('sparse', 'dense', 'aligned'):
            stripe = np.zeros((wc2048.Nt, stripe_height, nc))
            if layout == 'F':
                stripe = np.asfortranarray(stripe)
            _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup)
            assert_allclose(stripe, oracle, atol=atol, rtol=1e-9, err_msg=f'{name} {layout}')


@pytest.mark.parametrize('name', ['sparse', 'dense', 'aligned'])
def test_in_place_mutation(setup1: _Setup, name: str) -> None:
    """Each function mutates the supplied array in place and returns None.

    Verifies TR12/R4 (same object and data pointer retained) and that the call
    is not a no-op for an active case.
    """
    wc = setup1.wc
    nc = 3
    nf_start, stripe_height = 100, 4
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.25, 1.0, nc)
    stripe = np.zeros((wc.Nt, stripe_height, nc))
    stripe_ref = stripe
    ptr_before = stripe.ctypes.data

    result = None
    if name == 'sparse':
        result = wavemaket_stripe_sparse(stripe, waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    elif name == 'dense':
        result = wavemaket_stripe_dense(stripe, waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    elif name == 'aligned':
        result = wavemaket_stripe_dense_aligned(stripe, waveform, nf_start, stripe_height, nt_lim, wc, setup1.aligned)

    assert result is None
    assert stripe is stripe_ref
    assert stripe.ctypes.data == ptr_before
    assert np.count_nonzero(stripe) > 0


@pytest.mark.parametrize(('nf_start', 'stripe_height'), STRIPE_CASES)
@pytest.mark.parametrize('name', ['sparse', 'dense', 'aligned'])
def test_additive_coaddition(setup1: _Setup, name: str, nf_start: int, stripe_height: int) -> None:
    """Pre-filled stripes accumulate additively: result == B + oracle, not oracle.

    Verifies TR12/R5 (``+=`` coaddition) and that cells the oracle drops keep the
    pre-existing value B exactly.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.3, 1.0, nc)
    oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    atol = 1e-10 * _amplitude_source(waveform, nt_lim)

    fill_value = -3.5
    stripe = np.full((wc.Nt, stripe_height, nc), fill_value)
    _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)

    # additive, not overwrite: B + oracle differs from oracle because B != 0
    assert_allclose(stripe, fill_value + oracle, atol=atol, rtol=1e-9)
    assert not np.allclose(stripe, oracle)
    # cells the oracle drops keep exactly the pre-existing B value
    dropped = oracle == 0.0
    assert np.all(np.abs(stripe[dropped] - fill_value) <= atol)


def test_invalid_inputs_assert(setup1: _Setup) -> None:
    """Each required runtime validation assertion fires for invalid input.

    Verifies TR17 (shape, stripe_height, nf_start, channel count, and time range
    assertions) for the dense function, which shares the validation block with
    the other two.
    """
    wc = setup1.wc
    nc = 3
    table = setup1.table
    nf_start, stripe_height = 100, 4
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.0, 0.0, nc)

    # wrong ndim
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height)), waveform, nf_start, stripe_height, nt_lim, wc, table)
    # shape[1] != stripe_height
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height + 1, nc)), waveform, nf_start, stripe_height, nt_lim, wc, table)
    # invalid stripe_height
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, 6, nc)), waveform, nf_start, 6, nt_lim, wc, table)
    # negative nf_start
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height, nc)), waveform, -1, stripe_height, nt_lim, wc, table)
    # stripe extends beyond the full band
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height, nc)), waveform, wc.Nf - 1, stripe_height, nt_lim, wc, table)
    # mismatched channel count
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height, nc - 1)), waveform, nf_start, stripe_height, nt_lim, wc, table)
    # invalid time range: nx_min < 0
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height, nc)), waveform, nf_start, stripe_height, PixelGenericRange(-1, wc.Nt, wc.DT, 0.0), wc, table)
    # invalid time range: nx_max > Nt
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height, nc)), waveform, nf_start, stripe_height, PixelGenericRange(0, wc.Nt + 1, wc.DT, 0.0), wc, table)
    # invalid time range: nx_min > nx_max
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense(np.zeros((wc.Nt, stripe_height, nc)), waveform, nf_start, stripe_height, PixelGenericRange(10, 5, wc.DT, 0.0), wc, table)


def test_aligned_precondition_assertions(setup1: _Setup) -> None:
    """The aligned function rejects non-integer-aligned configurations.

    Verifies TR14 (additional aligned-path assertions): a configuration with
    ``2 * wc.Nsf % 3 != 0`` and one whose ``wc.DF / wc.df_bw`` ratio is perturbed
    away from ``R`` both fail.
    """
    wc = setup1.wc
    nc = 3
    aligned = setup1.aligned
    nf_start, stripe_height = 100, 4
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.0, 0.0, nc)
    stripe = np.zeros((wc.Nt, stripe_height, nc))

    # 2 * Nsf not divisible by 3
    wc_bad_nsf = wc._replace(Nsf=151)
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense_aligned(stripe, waveform, nf_start, stripe_height, nt_lim, wc_bad_nsf, aligned)
    with pytest.raises(AssertionError):
        build_aligned_taylor_time_table(setup1.table, wc_bad_nsf)

    # DF / df_bw ratio perturbed away from R while keeping 2 * Nsf % 3 == 0
    wc_bad_ratio = wc._replace(df_bw=wc.df_bw * 1.001)
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense_aligned(stripe, waveform, nf_start, stripe_height, nt_lim, wc_bad_ratio, aligned)
    with pytest.raises(AssertionError):
        build_aligned_taylor_time_table(setup1.table, wc_bad_ratio)


def test_aligned_precondition_holds(setup1: _Setup) -> None:
    """The aligned test configuration satisfies the integer-alignment precondition.

    Verifies TR14 structural precondition: ``2 * wc.Nsf % 3 == 0``,
    ``R == 2 * wc.Nsf // 3``, ``abs(wc.DF / wc.df_bw - R) <= 1e-15 * max(1, R)``,
    and that the repository Nsf = 150 configuration gives R == 100.
    """
    wc = setup1.wc
    assert 2 * wc.Nsf % 3 == 0
    r_stride = 2 * wc.Nsf // 3
    assert r_stride == 100
    assert setup1.aligned.R == r_stride
    assert abs(wc.DF / wc.df_bw - r_stride) <= 1e-15 * max(1.0, abs(r_stride))


def _assert_aligned_structural(aligned: WaveletTaylorTimeCoeffsAligned, table: WaveletTaylorTimeCoeffs, wc: WDMWaveletConstants) -> None:
    """Assert the aligned table exposes two-sided padding, valid-region metadata, and reuses originals.

    Raises AssertionError if any structural requirement is missing -- used both to
    confirm the real aligned table and to reject an identity wrapper.
    """
    pad = aligned.pad
    assert pad >= 1
    assert aligned.R == 2 * wc.Nsf // 3
    assert aligned.Nfd_negative == wc.Nfd_negative
    assert aligned.Nfsam.size == wc.Nfd
    assert aligned.evc.shape == aligned.evs.shape
    assert aligned.evc.shape[0] == wc.Nfd
    width = aligned.evc.shape[1]
    for n in range(wc.Nfd):
        n_valid = int(aligned.Nfsam[n])
        # left padding and right padding are exactly zero (two-sided padding)
        assert np.all(aligned.evc[n, :pad] == 0.0)
        assert np.all(aligned.evs[n, :pad] == 0.0)
        assert np.all(aligned.evc[n, pad + n_valid:] == 0.0)
        assert np.all(aligned.evs[n, pad + n_valid:] == 0.0)
        # the valid region reuses the original coefficient values bitwise
        assert np.array_equal(aligned.evc[n, pad:pad + n_valid], table.evc[n, :n_valid])
        assert np.array_equal(aligned.evs[n, pad:pad + n_valid], table.evs[n, :n_valid])
    # padded width covers the original valid coefficients plus the jj+1 read on both sides
    assert width >= pad + int(np.max(aligned.Nfsam)) + 1


def test_aligned_two_sided_padding_and_valid_region(setup1: _Setup) -> None:
    """The real aligned table satisfies the structural padding/valid-region requirements.

    Verifies TR14/TR25 (two-sided padding, valid-region metadata, original
    coefficient reuse).
    """
    _assert_aligned_structural(setup1.aligned, setup1.table, setup1.wc)


def test_identity_wrapper_rejected(setup1: _Setup) -> None:
    """An identity wrapper lacking padding fails the structural test and the function assertion.

    Verifies TR14/MC-v2-A: a plain wrapper around WaveletTaylorTimeCoeffs (pad = 0,
    no two-sided padding) is non-compliant.
    """
    wc = setup1.wc
    table = setup1.table
    identity = WaveletTaylorTimeCoeffsAligned(
        table.Nfsam.copy(),
        table.evc.copy(),
        table.evs.copy(),
        table.wavelet_norm.copy(),
        0,
        2 * wc.Nsf // 3,
        wc.Nfd_negative,
    )
    with pytest.raises(AssertionError):
        _assert_aligned_structural(identity, table, wc)

    nc = 3
    nf_start, stripe_height = 100, 4
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.0, 0.0, nc)
    stripe = np.zeros((wc.Nt, stripe_height, nc))
    with pytest.raises(AssertionError):
        wavemaket_stripe_dense_aligned(stripe, waveform, nf_start, stripe_height, nt_lim, wc, identity)


def test_aligned_padding_reads_zero(setup1: _Setup) -> None:
    """Representative reads in the left and right padding return exactly 0.0.

    Verifies TR14 padding test.
    """
    aligned = setup1.aligned
    pad = aligned.pad
    nfd = aligned.evc.shape[0]
    width = aligned.evc.shape[1]
    rows = (0, nfd // 2, nfd - 1)
    for n in rows:
        n_valid = int(aligned.Nfsam[n])
        # representative left-padding reads
        assert aligned.evc[n, 0] == 0.0
        assert aligned.evs[n, 0] == 0.0
        assert aligned.evc[n, pad - 1] == 0.0
        # representative right-padding reads
        assert aligned.evc[n, pad + n_valid] == 0.0
        assert aligned.evs[n, pad + n_valid] == 0.0
        assert aligned.evc[n, width - 1] == 0.0


def _classify_stripe_drops(
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    wc: WDMWaveletConstants,
    table: WaveletTaylorTimeCoeffs,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Locate active, bandwidth-dropped, and table-overflow-dropped stripe cells.

    This re-derives the oracle index arithmetic only to *classify* cells; the
    expected values still come from wavemaket. Returns boolean masks of shape
    (Nt, stripe_height, Nc): active (a contribution is written), bandwidth-dropped
    (k outside [kmin, kmax]), and overflow-dropped (k inside bandwidth but the
    jj1/jj2 table-overflow guard fails).
    """
    nc = waveform.AT.shape[0]
    active = np.zeros((wc.Nt, stripe_height, nc), dtype=np.bool_)
    bw_drop = np.zeros((wc.Nt, stripe_height, nc), dtype=np.bool_)
    overflow_drop = np.zeros((wc.Nt, stripe_height, nc), dtype=np.bool_)
    for itrc in range(nc):
        for j in range(nt_lim.nx_min, nt_lim.nx_max):
            y0 = waveform.FTd[itrc, j] / wc.dfd
            ny = int(np.floor(y0))
            n_ind = ny + wc.Nfd_negative
            if not (0 <= n_ind < wc.Nfd - 1):
                continue
            fa = waveform.FT[itrc, j]
            za = fa / wc.df_bw
            nfsam1 = int(table.Nfsam[n_ind])
            nfsam2 = int(table.Nfsam[n_ind + 1])
            half_bandwidth = (min(nfsam1, nfsam2) - 1) * wc.df_bw / 2
            kmin = max(0, int(np.ceil((fa - half_bandwidth) / wc.DF)))
            kmax = min(wc.Nf - 1, int(np.floor((fa + half_bandwidth) / wc.DF)))
            for k in range(nf_start, nf_start + stripe_height):
                col = k - nf_start
                if not (kmin <= k <= kmax):
                    bw_drop[j, col, itrc] = True
                    continue
                zmid = (wc.DF / wc.df_bw) * k
                if za < zmid:
                    zmid = za - abs(za - zmid)
                kk = int(np.floor(za - zmid - 0.5))
                jj1 = kk + nfsam1 // 2
                jj2 = kk + nfsam2 // 2
                if (0 <= jj1 < nfsam1 - 1) and (0 <= jj2 < nfsam2 - 1):
                    active[j, col, itrc] = True
                else:
                    overflow_drop[j, col, itrc] = True
    return active, bw_drop, overflow_drop


def _aligned_padding_only_stripe(
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    wc: WDMWaveletConstants,
    aligned: WaveletTaylorTimeCoeffsAligned,
) -> NDArray[np.floating]:
    """Compute the rejected padding-only aligned result (no table-overflow validity factor).

    Used only to demonstrate that relying on zero padding alone (blending the
    non-negligible edge coefficient with zero padding) would produce a different,
    nonzero contribution at table-overflow cells than the oracle's exact zero.
    """
    nc = waveform.AT.shape[0]
    out = np.zeros((wc.Nt, stripe_height, nc))
    pad = aligned.pad
    for itrc in range(nc):
        for j in range(nt_lim.nx_min, nt_lim.nx_max):
            y0 = waveform.FTd[itrc, j] / wc.dfd
            ny = int(np.floor(y0))
            n_ind = ny + wc.Nfd_negative
            if not (0 <= n_ind < wc.Nfd - 1):
                continue
            cval = np.cos(waveform.PT[itrc, j])
            sval = np.sin(waveform.PT[itrc, j])
            dy = y0 - ny
            fa = waveform.FT[itrc, j]
            za = fa / wc.df_bw
            nfsam1 = int(aligned.Nfsam[n_ind])
            nfsam2 = int(aligned.Nfsam[n_ind + 1])
            half_bandwidth = (min(nfsam1, nfsam2) - 1) * wc.df_bw / 2
            kmin = max(nf_start, int(np.ceil((fa - half_bandwidth) / wc.DF)))
            kmax = min(nf_start + stripe_height - 1, int(np.floor((fa + half_bandwidth) / wc.DF)))
            mult1 = waveform.AT[itrc, j]
            for k in range(kmin, kmax + 1):
                zmid = (wc.DF / wc.df_bw) * k
                if za < zmid:
                    zmid = za - abs(za - zmid)
                kk = int(np.floor(za - zmid - 0.5))
                dx = za - (zmid + kk + 0.5)
                idx1 = kk + nfsam1 // 2 + pad
                idx2 = kk + nfsam2 // 2 + pad
                y = (1.0 - dx) * aligned.evc[n_ind, idx1] + dx * aligned.evc[n_ind, idx1 + 1]
                yy = (1.0 - dx) * aligned.evc[n_ind + 1, idx2] + dx * aligned.evc[n_ind + 1, idx2 + 1]
                z = (1.0 - dx) * aligned.evs[n_ind, idx1] + dx * aligned.evs[n_ind, idx1 + 1]
                zz = (1.0 - dx) * aligned.evs[n_ind + 1, idx2] + dx * aligned.evs[n_ind + 1, idx2 + 1]
                y1 = ((1.0 - dy) * y + dy * yy) * mult1
                z1 = ((1.0 - dy) * z + dy * zz) * mult1
                if (j + k) % 2:
                    out[j, k - nf_start, itrc] += -(cval * z1 + sval * y1)
                else:
                    out[j, k - nf_start, itrc] += cval * y1 - sval * z1
    return out


def test_bandwidth_and_overflow_drops(setup1: _Setup) -> None:
    """The bandwidth drop and the table-overflow drop are both exercised and reproduced.

    Verifies TR8/TR26: a case that drops cells by both mechanisms; all three
    functions reproduce the exact oracle zeros; and at the overflow-dropped cells
    the rejected padding-only result is non-negligible, so the test cannot pass
    merely because padded reads blend a nonzero edge coefficient with zero padding.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    nf_start, stripe_height = 100, 2
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, -0.25, -1.99, nc)
    oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    amp = _amplitude_source(waveform, nt_lim)
    atol = 1e-10 * amp

    active, bw_drop, overflow_drop = _classify_stripe_drops(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    assert bw_drop.any(), 'expected at least one bandwidth-dropped cell'
    assert overflow_drop.any(), 'expected at least one table-overflow-dropped cell'
    assert active.any(), 'expected at least one active cell'

    # the oracle drops both classes to exactly zero
    assert np.all(oracle[bw_drop] == 0.0)
    assert np.all(oracle[overflow_drop] == 0.0)

    fill_value = 2.0
    for name in ('sparse', 'dense', 'aligned'):
        stripe = np.full((wc.Nt, stripe_height, nc), fill_value)
        _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
        assert_allclose(stripe, fill_value + oracle, atol=atol, rtol=1e-9, err_msg=name)
        # dropped cells keep exactly the pre-existing value
        assert np.all(np.abs(stripe[bw_drop] - fill_value) <= atol)
        assert np.all(np.abs(stripe[overflow_drop] - fill_value) <= atol)

    # the padding-only approach would write a non-negligible value at overflow cells,
    # confirming the drop is real and the validity factor (not padding) reproduces it
    padding_only = _aligned_padding_only_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.aligned)
    assert np.max(np.abs(padding_only[overflow_drop])) > 1e-3 * amp


def test_derivative_index_drop_out_of_domain(setup1: _Setup) -> None:
    """An out-of-supported-domain frequency derivative drops every contribution.

    Verifies TR8/MC-v2-B behaviorally: when ``ny + wc.Nfd_negative`` is outside
    ``[0, wc.Nfd - 1)`` the derivative-index guard drops the pixel, so the oracle
    and all three functions produce exactly zero over the stripe.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    nf_start, stripe_height = 100, 4
    # FTd large and negative so ny = floor(FTd/dfd) drives n_ind below 0
    ftd_const = -(wc.Nfd_negative + 20) * wc.dfd
    f0 = (nf_start + (stripe_height - 1) / 2.0) * wc.DF
    waveform = _build_linear_waveform(wc, f0, ftd_const, nc)
    ny = int(np.floor(ftd_const / wc.dfd))
    assert not (0 <= ny + wc.Nfd_negative < wc.Nfd - 1)

    oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    assert np.count_nonzero(oracle) == 0
    for name in ('sparse', 'dense', 'aligned'):
        stripe = np.zeros((wc.Nt, stripe_height, nc))
        _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
        assert np.count_nonzero(stripe) == 0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
