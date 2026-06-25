"""Tests for the dense-stripe Taylor-time wavelet coaddition functions.

The numerical oracle for every correctness test is the existing
:func:`WaveletWaveforms.taylor_time_wavelet_funcs.wavemaket` called with
``force_nulls=0`` and ``amplitude_order=0``, decoded to a dense
``(wc.Nt, wc.Nf, Nc)`` array with
:func:`WaveletWaveforms.sparse_waveform_functions.wavelet_sparse_to_dense` and
sliced to the stripe window. The oracle is independent of the implementation
under test.

``wavemaket_stripe_dense_aligned`` uses mandatory integer-stride ``R * k``
arithmetic (Contract Amendment 1) instead of the oracle's compiled ``fastmath``
``(wc.DF / wc.df_bw) * k``, so the two paths can disagree on a table-overflow
keep/drop decision at rare ULP-boundary cells. Those *aligned keep/drop boundary
cells* are classified by two compiled helpers -- a fastmath oracle-decision
helper (:func:`_oracle_decision_helper`) and an independent integer-stride
reference (:func:`_int_stride_reference_helper`) that does not call or reuse the
aligned implementation -- and at those cells the aligned output is checked
against the integer-stride reference instead of the oracle.

A small amount of the oracle's index arithmetic is also re-derived in
``_classify_stripe_drops`` and ``_aligned_padding_only_stripe``, but only to
*locate* which cells the oracle drops and to demonstrate the rejected
padding-only behavior. The expected *values* always come from ``wavemaket`` plus
``wavelet_sparse_to_dense`` (non-boundary cells) or the independent
integer-stride reference (boundary cells).
"""

import ast
import inspect
import textwrap
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import tomllib
from numba import njit
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

# All three supported public functions, used to parametrize shared-behavior tests.
FUNCTION_NAMES: tuple[str, ...] = ('sparse', 'dense', 'aligned')


class _Setup(NamedTuple):
    """Bundle of wavelet constants and Taylor tables shared across tests."""

    wc: WDMWaveletConstants
    table: WaveletTaylorTimeCoeffs
    aligned: WaveletTaylorTimeCoeffsAligned


class _AlignedRef(NamedTuple):
    """Boundary-cell classification for ``wavemaket_stripe_dense_aligned``.

    All arrays have shape ``(wc.Nt, stripe_height, Nc)``.
    """

    values: NDArray[np.floating]  # integer-stride reference contribution (0.0 where it drops)
    oracle_keep: NDArray[np.bool_]  # compiled fastmath oracle combined keep/drop
    reference_keep: NDArray[np.bool_]  # integer-stride reference combined keep/drop
    boundary: NDArray[np.bool_]  # aligned keep/drop boundary cells (table-overflow disagreement)


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


@njit(fastmath=True)
def _oracle_decision_helper(
    deriv: NDArray[np.bool_],
    bw: NDArray[np.bool_],
    overflow: NDArray[np.bool_],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    wc: WDMWaveletConstants,
    table: WaveletTaylorTimeCoeffs,
) -> None:
    """Fill per-cell compiled fastmath oracle keep/drop booleans, mirroring ``wavemaket``.

    ``deriv``, ``bw``, and ``overflow`` are filled in place with the derivative-index,
    bandwidth, and table-overflow guard outcomes computed with the same
    float-expression ``zmid = (wc.DF / wc.df_bw) * k`` as ``wavemaket`` under
    ``fastmath``. This is the observed compiled oracle keep/drop classifier required by
    Contract Amendment 1; it does not call any stripe implementation.
    """
    nf_top = nf_start + stripe_height - 1
    nc = waveform.AT.shape[0]
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
            for k in range(nf_start, nf_top + 1):
                col = k - nf_start
                deriv[j, col, itrc] = True
                bw[j, col, itrc] = kmin <= k <= kmax
                zmid = (wc.DF / wc.df_bw) * k
                if za < zmid:
                    zmid = za - np.abs(za - zmid)
                kk = int(np.floor(za - zmid - 0.5))
                jj1 = kk + nfsam1 // 2
                jj2 = kk + nfsam2 // 2
                overflow[j, col, itrc] = (0 <= jj1 < nfsam1 - 1) and (0 <= jj2 < nfsam2 - 1)


@njit(fastmath=True)
def _int_stride_reference_helper(
    values: NDArray[np.floating],
    keep: NDArray[np.bool_],
    deriv: NDArray[np.bool_],
    bw: NDArray[np.bool_],
    overflow: NDArray[np.bool_],
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    wc: WDMWaveletConstants,
    aligned: WaveletTaylorTimeCoeffsAligned,
) -> None:
    """Fill the independent integer-stride aligned reference value and keep/drop booleans.

    This is a per-cell reference written independently of
    ``wavemaket_stripe_dense_aligned`` (it does not call it or reuse its loop body):
    it evaluates each ``(j, k, c)`` directly with integer-stride ``R * k`` arithmetic
    and the same per-cell bandwidth and table-overflow guards as the oracle, emitting
    both the kept contribution value and the keep/drop booleans required by Contract
    Amendment 1.
    """
    nf_top = nf_start + stripe_height - 1
    nc = waveform.AT.shape[0]
    r_stride = aligned.R
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
            kmin = max(0, int(np.ceil((fa - half_bandwidth) / wc.DF)))
            kmax = min(wc.Nf - 1, int(np.floor((fa + half_bandwidth) / wc.DF)))
            kk_below = int(np.floor(za - 0.5))
            dx_below = (za - 0.5) - kk_below
            kk_above = int(np.floor(-za - 0.5))
            dx_above = (-za - 0.5) - kk_above
            k_high_start = int(np.floor(za / r_stride)) + 1
            mult1 = waveform.AT[itrc, j]
            for k in range(nf_start, nf_top + 1):
                col = k - nf_start
                deriv[j, col, itrc] = True
                bw[j, col, itrc] = kmin <= k <= kmax
                if k < k_high_start:
                    kk = kk_below - r_stride * k
                    dx = dx_below
                else:
                    kk = kk_above + r_stride * k
                    dx = dx_above
                jj1 = kk + nfsam1 // 2
                jj2 = kk + nfsam2 // 2
                overflow[j, col, itrc] = (0 <= jj1 < nfsam1 - 1) and (0 <= jj2 < nfsam2 - 1)
                if bw[j, col, itrc] and overflow[j, col, itrc]:
                    keep[j, col, itrc] = True
                    idx1 = jj1 + pad
                    idx2 = jj2 + pad
                    y = (1.0 - dx) * aligned.evc[n_ind, idx1] + dx * aligned.evc[n_ind, idx1 + 1]
                    yy = (1.0 - dx) * aligned.evc[n_ind + 1, idx2] + dx * aligned.evc[n_ind + 1, idx2 + 1]
                    z = (1.0 - dx) * aligned.evs[n_ind, idx1] + dx * aligned.evs[n_ind, idx1 + 1]
                    zz = (1.0 - dx) * aligned.evs[n_ind + 1, idx2] + dx * aligned.evs[n_ind + 1, idx2 + 1]
                    y1 = ((1.0 - dy) * y + dy * yy) * mult1
                    z1 = ((1.0 - dy) * z + dy * zz) * mult1
                    if (j + k) % 2:
                        values[j, col, itrc] = -(cval * z1 + sval * y1)
                    else:
                        values[j, col, itrc] = cval * y1 - sval * z1


def _aligned_reference(
    waveform: StationaryWaveformTime,
    nf_start: int,
    stripe_height: int,
    nt_lim: PixelGenericRange,
    setup: _Setup,
) -> _AlignedRef:
    """Classify aligned keep/drop boundary cells via the two independent compiled helpers.

    A boundary cell is one where the compiled fastmath oracle and the integer-stride
    reference agree on the derivative-index, bandwidth, and stripe-window guards but
    differ on the table-overflow keep/drop decision (Contract Amendment 1).
    """
    wc = setup.wc
    nc = waveform.AT.shape[0]
    shape = (wc.Nt, stripe_height, nc)
    o_deriv = np.zeros(shape, np.bool_)
    o_bw = np.zeros(shape, np.bool_)
    o_overflow = np.zeros(shape, np.bool_)
    _oracle_decision_helper(o_deriv, o_bw, o_overflow, waveform, nf_start, stripe_height, nt_lim, wc, setup.table)

    r_values = np.zeros(shape)
    r_keep = np.zeros(shape, np.bool_)
    r_deriv = np.zeros(shape, np.bool_)
    r_bw = np.zeros(shape, np.bool_)
    r_overflow = np.zeros(shape, np.bool_)
    _int_stride_reference_helper(
        r_values, r_keep, r_deriv, r_bw, r_overflow, waveform, nf_start, stripe_height, nt_lim, wc, setup.aligned
    )

    boundary = o_deriv & r_deriv & (o_bw == r_bw) & o_bw & (o_overflow != r_overflow)
    oracle_keep = o_deriv & o_bw & o_overflow
    return _AlignedRef(r_values, oracle_keep, r_keep, boundary)


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


def _assert_contribution(
    name: str,
    out: NDArray[np.floating],
    oracle: NDArray[np.floating],
    ref: _AlignedRef,
    atol: float,
    base: float,
    context: str,
) -> None:
    """Assert ``out - base`` matches the oracle, applying the aligned boundary exception.

    For ``sparse`` and ``dense`` the whole stripe must match the oracle. For ``aligned``,
    non-boundary cells must match the oracle and aligned keep/drop boundary cells must
    instead match the independent integer-stride reference value (Contract Amendment 1).
    """
    contrib = out - base
    if name == 'aligned':
        non_boundary = ~ref.boundary
        assert_allclose(contrib[non_boundary], oracle[non_boundary], atol=atol, rtol=1e-9, err_msg=f'aligned non-boundary {context}')
        if ref.boundary.any():
            assert_allclose(contrib[ref.boundary], ref.values[ref.boundary], atol=atol, rtol=1e-9, err_msg=f'aligned boundary {context}')
    else:
        assert_allclose(contrib, oracle, atol=atol, rtol=1e-9, err_msg=f'{name} {context}')


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
    required slopes. Aligned comparisons apply the Contract Amendment 1 keep/drop
    boundary exception (AM1-R5/AM1-R14).
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
        ref = _aligned_reference(waveform, nf_start, stripe_height, nt_lim, setup1)
        context = f'offset={offset_pix} slope={slope_pix} nf_start={nf_start} sh={stripe_height}'

        outputs: dict[str, NDArray[np.floating]] = {}
        for name in FUNCTION_NAMES:
            stripe = np.zeros((wc.Nt, stripe_height, nc))
            _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
            outputs[name] = stripe
            _assert_contribution(name, stripe, oracle, ref, atol, 0.0, context)

        # pairwise agreement: sparse vs dense exactly; aligned vs others off the
        # boundary cells (AM1-R14 extends the boundary exception to pairwise tests).
        assert_allclose(outputs['sparse'], outputs['dense'], atol=atol, rtol=1e-9)
        non_boundary = ~ref.boundary
        assert_allclose(outputs['aligned'][non_boundary], outputs['sparse'][non_boundary], atol=atol, rtol=1e-9)
        assert_allclose(outputs['aligned'][non_boundary], outputs['dense'][non_boundary], atol=atol, rtol=1e-9)


@pytest.mark.parametrize(('nf_start', 'stripe_height'), STRIPE_CASES)
def test_pixel_boundary_and_halfway(setup1: _Setup, nf_start: int, stripe_height: int) -> None:
    """Frequencies exactly on a pixel boundary and exactly halfway between boundaries.

    Verifies the on-boundary and halfway required-coverage cases (TR15) for all
    three functions against the oracle, applying the aligned boundary exception.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    k0 = nf_start + stripe_height // 2
    for f0 in (k0 * wc.DF, (k0 + 0.5) * wc.DF):
        waveform = _build_linear_waveform(wc, f0, 0.0, nc)
        oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
        atol = 1e-10 * _amplitude_source(waveform, nt_lim)
        ref = _aligned_reference(waveform, nf_start, stripe_height, nt_lim, setup1)
        # the source sits inside the stripe, so the oracle must place real power there
        assert np.count_nonzero(oracle) > 0
        for name in FUNCTION_NAMES:
            stripe = np.zeros((wc.Nt, stripe_height, nc))
            _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
            _assert_contribution(name, stripe, oracle, ref, atol, 0.0, f'f0={f0}')


@pytest.mark.parametrize('layout', ['C', 'F'])
def test_nt_2048_matches_oracle(setup1: _Setup, layout: str) -> None:
    """A representative wc.Nt == 2048 case for C- and F-contiguous stripes.

    Verifies TR16 (representative Nt == 2048) and the contiguity-layout coverage.
    The Taylor table is independent of Nt, so the cached config1 table is reused
    with an Nt == 2048 wavelet-constants object.
    """
    wc2048 = setup1.wc._replace(Nt=2048)
    setup = _Setup(wc2048, setup1.table, setup1.aligned)
    nc = 3
    nt_lim = PixelGenericRange(0, wc2048.Nt, wc2048.DT, 0.0)
    nf_start, stripe_height = 100, 4
    for offset_pix, slope_pix in ((0.25, 1.0), (-0.3, -1.99), (0.0, 0.0)):
        waveform = _waveform_from_offset_slope(wc2048, nf_start, stripe_height, offset_pix, slope_pix, nc)
        oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc2048, setup.table)
        atol = 1e-10 * _amplitude_source(waveform, nt_lim)
        ref = _aligned_reference(waveform, nf_start, stripe_height, nt_lim, setup)
        for name in FUNCTION_NAMES:
            stripe = np.zeros((wc2048.Nt, stripe_height, nc))
            if layout == 'F':
                stripe = np.asfortranarray(stripe)
            _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup)
            _assert_contribution(name, stripe, oracle, ref, atol, 0.0, f'{layout} offset={offset_pix} slope={slope_pix}')


@pytest.mark.parametrize('name', FUNCTION_NAMES)
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
@pytest.mark.parametrize('name', FUNCTION_NAMES)
def test_additive_coaddition(setup1: _Setup, name: str, nf_start: int, stripe_height: int) -> None:
    """Pre-filled stripes accumulate additively: result == B + oracle, not oracle.

    Verifies TR12/R5 (``+=`` coaddition) and that cells the function drops keep the
    pre-existing value B exactly. Aligned applies the keep/drop boundary exception.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.3, 1.0, nc)
    oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    atol = 1e-10 * _amplitude_source(waveform, nt_lim)
    ref = _aligned_reference(waveform, nf_start, stripe_height, nt_lim, setup1)

    fill_value = -3.5
    stripe = np.full((wc.Nt, stripe_height, nc), fill_value)
    _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)

    # additive, not overwrite: B + contribution differs from the bare contribution
    _assert_contribution(name, stripe, oracle, ref, atol, fill_value, f'nf_start={nf_start} sh={stripe_height}')
    assert not np.allclose(stripe, oracle)
    # cells the function drops keep exactly the pre-existing B value. For sparse/dense
    # the drop set is the oracle's; for aligned it is the integer-stride drop set.
    dropped = (~ref.reference_keep) if name == 'aligned' else (oracle == 0.0)
    assert np.all(np.abs(stripe[dropped] - fill_value) <= atol)


@pytest.mark.parametrize('name', FUNCTION_NAMES)
def test_invalid_inputs_assert(setup1: _Setup, name: str) -> None:
    """Each required runtime validation assertion fires for invalid input.

    Verifies TR17 for all three public functions (regression for PR #40 finding
    F002): the shape, stripe_height, nf_start, channel-count, and time-range
    assertions are exercised through each function's own validation block, not only
    through ``wavemaket_stripe_dense``.
    """
    wc = setup1.wc
    nc = 3
    nf_start, stripe_height = 100, 4
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, 0.0, 0.0, nc)

    def call(stripe: NDArray[np.floating], nf: int, sh: int, lim: PixelGenericRange) -> None:
        _run_function(name, stripe, waveform, nf, sh, lim, setup1)

    # wrong ndim
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height)), nf_start, stripe_height, nt_lim)
    # shape[1] != stripe_height
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height + 1, nc)), nf_start, stripe_height, nt_lim)
    # invalid stripe_height
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, 6, nc)), nf_start, 6, nt_lim)
    # negative nf_start
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height, nc)), -1, stripe_height, nt_lim)
    # stripe extends beyond the full band
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height, nc)), wc.Nf - 1, stripe_height, nt_lim)
    # mismatched channel count
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height, nc - 1)), nf_start, stripe_height, nt_lim)
    # invalid time range: nx_min < 0
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height, nc)), nf_start, stripe_height, PixelGenericRange(-1, wc.Nt, wc.DT, 0.0))
    # invalid time range: nx_max > Nt
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height, nc)), nf_start, stripe_height, PixelGenericRange(0, wc.Nt + 1, wc.DT, 0.0))
    # invalid time range: nx_min > nx_max
    with pytest.raises(AssertionError):
        call(np.zeros((wc.Nt, stripe_height, nc)), nf_start, stripe_height, PixelGenericRange(10, 5, wc.DT, 0.0))


def test_aligned_precondition_assertions(setup1: _Setup) -> None:
    """The aligned function rejects non-integer-aligned configurations.

    Verifies TR14/AM1-R7 (additional aligned-path assertions): a configuration with
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

    Verifies TR14/AM1-R7 structural precondition in the cross-multiplied form:
    ``2 * wc.Nsf % 3 == 0``, ``R == 2 * wc.Nsf // 3``,
    ``abs(wc.DF - R * wc.df_bw) <= 1e-15 * max(1, R) * wc.df_bw``, and that the
    repository Nsf = 150 configuration gives R == 100.
    """
    wc = setup1.wc
    assert 2 * wc.Nsf % 3 == 0
    r_stride = 2 * wc.Nsf // 3
    assert r_stride == 100
    assert setup1.aligned.R == r_stride
    # cross-multiplied ratio form mandated by Contract Amendment 1 (AM1-R7)
    assert abs(wc.DF - r_stride * wc.df_bw) <= 1e-15 * max(1.0, abs(r_stride)) * wc.df_bw


def _aligned_function_ast() -> ast.FunctionDef:
    """Parse ``wavemaket_stripe_dense_aligned`` to its FunctionDef (comments dropped)."""
    # the numba dispatcher keeps the original Python function under .py_func
    func = wavemaket_stripe_dense_aligned.py_func
    source = textwrap.dedent(inspect.getsource(func))
    module = ast.parse(source)
    func_def = module.body[0]
    assert isinstance(func_def, ast.FunctionDef)
    return func_def


def _per_k_loops(func_def: ast.FunctionDef) -> list[ast.For]:
    """Return the per-``k`` ``for`` loops (target named ``k``) in the function."""
    return [
        node
        for node in ast.walk(func_def)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'k'
    ]


def test_aligned_source_uses_integer_stride() -> None:
    """The aligned function uses integer-stride zmid and forms no per-k floating ratio.

    Verifies AM1-R2/AM1-R15/AM1-R17: no ``(wc.DF / wc.df_bw) * k`` or equivalent
    floating-ratio product appears, and the body carries no ``wc.DF / wc.df_bw``
    division at all (the precondition uses the cross-multiplied form). Comments are
    dropped because the check runs against the parsed AST.
    """
    func_def = _aligned_function_ast()

    # No division of wc.DF by wc.df_bw anywhere (covers the division-form assertion
    # and the (wc.DF / wc.df_bw) * k loop product and its k * (...) reorderings).
    for node in ast.walk(func_def):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator = ast.unparse(node.left)
            denominator = ast.unparse(node.right)
            assert not ('wc.DF' in numerator and 'wc.df_bw' in denominator), (
                f'forbidden floating ratio wc.DF / wc.df_bw: {ast.unparse(node)}'
            )

    # The mandated integer stride and reflection split are present.
    body_src = ast.unparse(func_def)
    assert '2 * wc.Nsf // 3' in body_src
    assert 'kstep * k_start' in body_src
    assert 'jj1 += 2 * kstep' in body_src
    assert 'jj2 += 2 * kstep' in body_src
    assert 'int(np.floor(za / r_stride)) + 1' in body_src


def test_aligned_source_per_k_loop_is_branchless() -> None:
    """The per-``k`` loop body contains no keep/drop, reflection, or parity branch.

    Verifies AM1-R10/AM1-R18: there is exactly one per-``k`` loop and its body has no
    ``if``, ``continue``, ``break``, ``return``, or conditional-expression node, so the
    reflection, bandwidth, table-overflow, and ``(j_ind + k) % 2`` decisions are all
    handled by per-regime loop bounds and per-parity sub-loops outside the hot loop.
    """
    func_def = _aligned_function_ast()
    loops = _per_k_loops(func_def)
    assert len(loops) == 1, f'expected exactly one per-k loop, found {len(loops)}'
    forbidden = (ast.If, ast.IfExp, ast.Continue, ast.Break, ast.Return, ast.While)
    offending = [type(node).__name__ for node in ast.walk(loops[0]) if isinstance(node, forbidden)]
    assert not offending, f'per-k loop body must be branchless, found {offending}'


def test_aligned_source_dx_constant_per_regime() -> None:
    """``dx`` is fixed at regime entry and the inner loop only advances the base indices.

    Verifies AM1-R9: the per-``k`` loop body does not reassign ``dx`` (it is selected
    once per regime), and it advances ``jj1``/``jj2`` by the integer stride.
    """
    func_def = _aligned_function_ast()
    loop = _per_k_loops(func_def)[0]
    assigned_names: set[str] = set()
    for node in ast.walk(loop):
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        assigned_names.update(target.id for target in targets if isinstance(target, ast.Name))
    assert 'dx' not in assigned_names, 'dx must not be recomputed inside the per-k loop'
    # the base indices are advanced inside the loop (by the integer stride)
    assert 'jj1' in assigned_names
    assert 'jj2' in assigned_names


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
    """Compute the rejected padding-only aligned result (no table-overflow loop bound).

    Used only to demonstrate that relying on zero padding alone (blending the
    non-negligible edge coefficient with zero padding) would produce a different,
    nonzero contribution at table-overflow cells than the exact zero the oracle and
    the loop-bounded aligned function produce.
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

    Verifies TR8/TR26: a case that drops cells by both mechanisms; sparse and dense
    reproduce the exact oracle zeros; aligned reproduces them off its keep/drop
    boundary cells; and at the overflow-dropped cells the rejected padding-only result
    is non-negligible, so the test cannot pass merely because padded reads blend a
    nonzero edge coefficient with zero padding.
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    nf_start, stripe_height = 100, 2
    waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, -0.25, -1.99, nc)
    oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    amp = _amplitude_source(waveform, nt_lim)
    atol = 1e-10 * amp
    ref = _aligned_reference(waveform, nf_start, stripe_height, nt_lim, setup1)

    active, bw_drop, overflow_drop = _classify_stripe_drops(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
    assert bw_drop.any(), 'expected at least one bandwidth-dropped cell'
    assert overflow_drop.any(), 'expected at least one table-overflow-dropped cell'
    assert active.any(), 'expected at least one active cell'

    # the oracle drops both classes to exactly zero
    assert np.all(oracle[bw_drop] == 0.0)
    assert np.all(oracle[overflow_drop] == 0.0)

    fill_value = 2.0
    for name in FUNCTION_NAMES:
        stripe = np.full((wc.Nt, stripe_height, nc), fill_value)
        _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
        _assert_contribution(name, stripe, oracle, ref, atol, fill_value, name)
        # off the aligned boundary cells, dropped cells keep exactly the pre-existing value
        keep_check = (bw_drop | overflow_drop) & ~ref.boundary
        assert np.all(np.abs(stripe[keep_check] - fill_value) <= atol)

    # the padding-only approach would write a non-negligible value at overflow cells,
    # confirming the drop is real and the loop bounds (not padding) reproduce it
    padding_only = _aligned_padding_only_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.aligned)
    assert np.max(np.abs(padding_only[overflow_drop])) > 1e-3 * amp


def test_aligned_keep_drop_boundary_exception(setup1: _Setup) -> None:
    """The aligned keep/drop boundary exception is correctly classified and applied.

    Verifies AM1-R3/AM1-R4/AM1-R6 and the amendment's boundary-test verification
    bullets. Sweeps the required coverage matrix; for every case all non-boundary
    cells match the oracle at standard tolerance, and any aligned keep/drop boundary
    cell matches the independent integer-stride reference (``0.0`` when that reference
    drops, the reference contribution when it keeps) while genuinely differing from the
    oracle. The exception is never applied to a cell where the oracle and the reference
    both keep. If the compiled environment produces no boundary cells the test still
    passes by verifying the standard path for all cells (AM1-R16).
    """
    wc = setup1.wc
    nc = 3
    nt_lim = PixelGenericRange(0, wc.Nt, wc.DT, 0.0)
    total_boundary = 0
    total_int_keep = 0
    total_int_drop = 0
    for nf_start, stripe_height in STRIPE_CASES:
        for offset_pix in OFFSETS:
            for slope_pix in SLOPES:
                waveform = _waveform_from_offset_slope(wc, nf_start, stripe_height, offset_pix, slope_pix, nc)
                oracle = _oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, setup1.table)
                atol = 1e-10 * _amplitude_source(waveform, nt_lim)
                ref = _aligned_reference(waveform, nf_start, stripe_height, nt_lim, setup1)

                aligned_out = np.zeros((wc.Nt, stripe_height, nc))
                wavemaket_stripe_dense_aligned(aligned_out, waveform, nf_start, stripe_height, nt_lim, wc, setup1.aligned)

                non_boundary = ~ref.boundary
                # non-boundary cells (including all both-keep cells) use the standard oracle tolerance
                assert_allclose(aligned_out[non_boundary], oracle[non_boundary], atol=atol, rtol=1e-9)
                # the exception is never applied where the oracle and the reference both keep
                both_keep = ref.oracle_keep & ref.reference_keep
                assert not np.any(both_keep & ref.boundary)

                if ref.boundary.any():
                    # boundary cells take the integer-stride reference value and genuinely
                    # differ from the oracle (a real keep/drop flip, not a silent no-op)
                    assert_allclose(aligned_out[ref.boundary], ref.values[ref.boundary], atol=atol, rtol=1e-9)
                    assert np.all(np.abs(aligned_out[ref.boundary] - oracle[ref.boundary]) > atol)
                    boundary_drop = ref.boundary & ~ref.reference_keep
                    boundary_keep = ref.boundary & ref.reference_keep
                    # integer-stride drop -> exactly 0.0; integer-stride keep -> reference contribution
                    assert np.all(aligned_out[boundary_drop] == 0.0)
                    assert np.all(np.abs(aligned_out[boundary_keep] - ref.values[boundary_keep]) <= atol)
                    total_int_drop += int(boundary_drop.sum())
                    total_int_keep += int(boundary_keep.sum())
                total_boundary += int(ref.boundary.sum())

    # Either boundary cells were produced (and were exercised above) or none were, in
    # which case the standard-tolerance path was verified for every cell. The split of
    # integer-stride keeps/drops must account for all boundary cells.
    assert total_int_keep + total_int_drop == total_boundary


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
    for name in FUNCTION_NAMES:
        stripe = np.zeros((wc.Nt, stripe_height, nc))
        _run_function(name, stripe, waveform, nf_start, stripe_height, nt_lim, setup1)
        assert np.count_nonzero(stripe) == 0


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
