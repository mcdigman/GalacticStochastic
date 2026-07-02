"""Speed benchmark/demo for the dense-stripe Taylor-time wavelet coaddition functions.

This script benchmarks the median per-call wall time of the full-band sparse
``wavemaket`` against the three dense-stripe functions
(``wavemaket_stripe_sparse``, ``wavemaket_stripe_dense``,
``wavemaket_stripe_dense_aligned``) over a narrow frequency stripe.

The benchmark:

- excludes waveform/source construction and stripe allocation from the timed
  sections (only the function call itself is timed),
- warms up numba compilation before recording any timing,
- verifies every benchmarked float64 stripe output matches the ``wavemaket`` +
  ``wavelet_sparse_to_dense`` oracle within the faithful tolerance
  (``atol = 1e-10 * amplitude_source``, ``rtol = 1e-9``) before reporting speed,
  applying the Contract Amendment 1 aligned keep/drop boundary exception (at the
  rare ULP table-overflow boundary cells the integer-stride aligned path is
  checked against an independent integer-stride reference instead of the oracle),
- covers ``wc.Nt == 2048``, ``Nc == 3``, stripe heights 2, 3, 4, 5, and both
  C-contiguous and F-contiguous stripe layouts.

There is no required speed threshold; this is an evidence-producing artifact.

Run from the repository root with the project on the path, e.g.::

    python -m speed_tests.taylor_time_wavelet_optimized_speed_demo
"""

import time
from pathlib import Path

import numpy as np
import tomllib
from numba import njit

from LisaWaveformTools.stationary_source_waveform import StationaryWaveformTime
from WaveletWaveforms.sparse_waveform_functions import PixelGenericRange, wavelet_sparse_to_dense
from WaveletWaveforms.taylor_time_coefficients import (
    get_empty_sparse_taylor_time_waveform,
    get_taylor_table_time,
)
from WaveletWaveforms.taylor_time_wavelet_funcs import wavemaket
from WaveletWaveforms.taylor_time_wavelet_optimized import (
    build_aligned_taylor_time_table,
    wavemaket_stripe_dense,
    wavemaket_stripe_dense_aligned,
    wavemaket_stripe_sparse,
)
from WaveletWaveforms.wdm_config import get_wavelet_model

N_TIMING = 21
NF_START = 100
OFFSET_PIX = 0.25
SLOPE_PIX = 1.0


def build_waveform(wc, nf_start, stripe_height, nc):
    """Build a supported-domain linear-frequency waveform (excluded from timing)."""
    stripe_center_layer = nf_start + (stripe_height - 1) / 2.0
    f0 = (stripe_center_layer + OFFSET_PIX) * wc.DF
    ftd_const = SLOPE_PIX * wc.DF / wc.Tobs
    t_grid = np.arange(wc.Nt) * wc.DT
    pt = np.zeros((nc, wc.Nt))
    ft = np.zeros((nc, wc.Nt))
    ftd = np.zeros((nc, wc.Nt))
    at = np.zeros((nc, wc.Nt))
    for itrc in range(nc):
        ft[itrc] = f0 + ftd_const * t_grid
        ftd[itrc] = ftd_const
        pt[itrc] = 2 * np.pi * (f0 * t_grid + 0.5 * ftd_const * t_grid**2) + 0.3 * itrc
        at[itrc] = (1.0 + 0.2 * itrc) * (1.0 + 0.1 * np.sin(2 * np.pi * t_grid / wc.Tobs))
    return StationaryWaveformTime(t_grid, pt, ft, ftd, at)


def oracle_stripe(waveform, nf_start, stripe_height, nt_lim, wc, table):
    """Reference stripe slice from wavemaket + wavelet_sparse_to_dense."""
    nc = waveform.AT.shape[0]
    sparse = get_empty_sparse_taylor_time_waveform(nc, wc)
    wavemaket(sparse, waveform, nt_lim, wc, table, force_nulls=0, amplitude_order=0)
    dense = wavelet_sparse_to_dense(sparse, wc)
    return np.ascontiguousarray(dense[:, nf_start:nf_start + stripe_height, :])


def make_stripe(nt, stripe_height, nc, layout):
    """Allocate a zeroed stripe array with the requested contiguity (excluded from timing)."""
    stripe = np.zeros((nt, stripe_height, nc))
    if layout == 'F':
        return np.asfortranarray(stripe)
    return stripe


@njit(fastmath=True)
def aligned_int_stride_reference(out, waveform, nf_start, stripe_height, nt_lim, wc, aligned):
    """Independent per-cell integer-stride aligned reference (mirrors the keep/drop, not the loop).

    Written independently of ``wavemaket_stripe_dense_aligned`` (per-cell guards rather
    than the implementation's per-regime loop bounds), this evaluates each cell with the
    same integer-stride ``R * k`` arithmetic so the benchmark can verify the aligned
    output at Contract Amendment 1 keep/drop boundary cells against an independent
    reference rather than the oracle.
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
                if not (kmin <= k <= kmax):
                    continue
                if k < k_high_start:
                    kk = kk_below - r_stride * k
                    dx = dx_below
                else:
                    kk = kk_above + r_stride * k
                    dx = dx_above
                jj1 = kk + nfsam1 // 2
                jj2 = kk + nfsam2 // 2
                if (0 <= jj1 < nfsam1 - 1) and (0 <= jj2 < nfsam2 - 1):
                    idx1 = jj1 + pad
                    idx2 = jj2 + pad
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


def aligned_correctness_ok(aligned_out, oracle, reference, atol):
    """Aligned matches the oracle off its keep/drop boundary cells and the reference on them.

    Boundary cells are those where the independent integer-stride reference differs from
    the oracle beyond tolerance (Contract Amendment 1). Returns the boolean result and the
    number of boundary cells.
    """
    boundary = ~np.isclose(reference, oracle, atol=atol, rtol=1e-9)
    off_boundary_ok = np.allclose(aligned_out[~boundary], oracle[~boundary], atol=atol, rtol=1e-9)
    on_boundary_ok = np.allclose(aligned_out[boundary], reference[boundary], atol=atol, rtol=1e-9)
    return bool(off_boundary_ok and on_boundary_ok), int(boundary.sum())


def median_stripe_time(func, stripe, waveform, nf_start, stripe_height, nt_lim, wc, tbl, n_repeat):
    """Median wall time (seconds) of a stripe function over n_repeat calls."""
    times = np.empty(n_repeat)
    for i in range(n_repeat):
        t0 = time.perf_counter()
        func(stripe, waveform, nf_start, stripe_height, nt_lim, wc, tbl)
        times[i] = time.perf_counter() - t0
    return float(np.median(times))


def median_wavemaket_time(sparse, waveform, nt_lim, wc, table, n_repeat):
    """Median wall time (seconds) of the full-band sparse wavemaket over n_repeat calls."""
    times = np.empty(n_repeat)
    for i in range(n_repeat):
        t0 = time.perf_counter()
        wavemaket(sparse, waveform, nt_lim, wc, table, force_nulls=0, amplitude_order=0)
        times[i] = time.perf_counter() - t0
    return float(np.median(times))


def nt_lim_of(wc):
    """Full time-pixel range for the given wavelet constants."""
    return PixelGenericRange(0, wc.Nt, wc.DT, 0.0)


def benchmark_layout(wc, table, aligned, waveform, oracle, stripe_height, layout, atol, nc):
    """Warm up, verify correctness, and time the three stripe functions for one layout.

    Returns the per-function median timings and the number of aligned keep/drop
    boundary cells exempted from the oracle comparison for this layout.
    """
    timings = {}
    n_boundary = 0
    for name, func, tbl in (
        ('sparse', wavemaket_stripe_sparse, table),
        ('dense', wavemaket_stripe_dense, table),
        ('aligned', wavemaket_stripe_dense_aligned, aligned),
    ):
        # warm up compilation for this layout/shape, then verify correctness before timing
        warm_stripe = make_stripe(wc.Nt, stripe_height, nc, layout)
        func(warm_stripe, waveform, NF_START, stripe_height, nt_lim_of(wc), wc, tbl)
        if name == 'aligned':
            # aligned uses integer-stride arithmetic; apply the Contract Amendment 1
            # keep/drop boundary exception, checking boundary cells against an
            # independent integer-stride reference instead of the oracle.
            reference = make_stripe(wc.Nt, stripe_height, nc, 'C')
            aligned_int_stride_reference(reference, waveform, NF_START, stripe_height, nt_lim_of(wc), wc, aligned)
            ok, n_boundary = aligned_correctness_ok(warm_stripe, oracle, reference, atol)
        else:
            ok = bool(np.allclose(warm_stripe, oracle, atol=atol, rtol=1e-9))
        if not ok:
            max_err = float(np.max(np.abs(warm_stripe - oracle)))
            msg = (
                f'{name} ({layout}-contiguous, stripe_height={stripe_height}) does not match the '
                f'oracle: max abs error {max_err:.3e} > atol {atol:.3e}'
            )
            raise AssertionError(msg)

        timed_stripe = make_stripe(wc.Nt, stripe_height, nc, layout)
        timings[name] = median_stripe_time(
            func, timed_stripe, waveform, NF_START, stripe_height, nt_lim_of(wc), wc, tbl, N_TIMING,
        )
    return timings, n_boundary


def main():
    """Run the benchmark and print median per-call wall times."""
    with Path('tests/wavemaket_test_config1.toml').open('rb') as f:
        config = tomllib.load(f)

    # The Taylor table is independent of Nt, so load it from the cached Nt=1024
    # configuration (cache hit, no recomputation) and reuse it for the Nt=2048 run.
    wc_table = get_wavelet_model(config)
    table = get_taylor_table_time(wc_table, cache_mode='check', output_mode='skip', grid_check_mode=0)

    config['wavelet_constants']['Nt'] = 2048
    wc = get_wavelet_model(config)
    aligned = build_aligned_taylor_time_table(table, wc)

    nc = 3
    nt_lim = nt_lim_of(wc)

    print(f'Configuration: Nf={wc.Nf} Nt={wc.Nt} Nc={nc} nf_start={NF_START} R={aligned.R}')
    print(f'Timed runs per measurement (median reported): {N_TIMING}')
    print()

    # Full-band sparse wavemaket is independent of the stripe; build/warm/time once.
    waveform = build_waveform(wc, NF_START, 4, nc)
    sparse = get_empty_sparse_taylor_time_waveform(nc, wc)
    wavemaket(sparse, waveform, nt_lim, wc, table, force_nulls=0, amplitude_order=0)  # warm-up/compile
    wavemaket_full_time = median_wavemaket_time(sparse, waveform, nt_lim, wc, table, N_TIMING)
    print(f'wavemaket (full-band sparse, force_nulls=0): {wavemaket_full_time * 1e3:.4f} ms/call')
    print()

    header = f'{"stripe_height":>13} {"layout":>7} {"sparse[ms]":>12} {"dense[ms]":>12} {"aligned[ms]":>12}'
    print(header)
    print('-' * len(header))

    total_boundary = 0
    for stripe_height in (2, 3, 4, 5):
        waveform = build_waveform(wc, NF_START, stripe_height, nc)
        oracle = oracle_stripe(waveform, NF_START, stripe_height, nt_lim, wc, table)
        amp = float(np.max(np.abs(waveform.AT[:, nt_lim.nx_min:nt_lim.nx_max])))
        atol = 1e-10 * amp

        for layout in ('C', 'F'):
            timings, n_boundary = benchmark_layout(wc, table, aligned, waveform, oracle, stripe_height, layout, atol, nc)
            total_boundary += n_boundary
            print(
                f'{stripe_height:>13} {layout:>7} '
                f'{timings["sparse"] * 1e3:>12.4f} {timings["dense"] * 1e3:>12.4f} {timings["aligned"] * 1e3:>12.4f}'
            )

    print()
    print(
        'All benchmarked float64 stripe outputs matched the oracle within the faithful tolerance '
        f'({total_boundary} aligned keep/drop boundary cell(s) checked against the integer-stride reference).'
    )
    print('No speed threshold is enforced; results are evidence for human review.')


if __name__ == '__main__':
    main()
