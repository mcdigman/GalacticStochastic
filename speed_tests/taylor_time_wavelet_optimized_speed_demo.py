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
    """Warm up, verify correctness, and time the three stripe functions for one layout."""
    timings = {}
    for name, func, tbl in (
        ('sparse', wavemaket_stripe_sparse, table),
        ('dense', wavemaket_stripe_dense, table),
        ('aligned', wavemaket_stripe_dense_aligned, aligned),
    ):
        # warm up compilation for this layout/shape, then verify correctness before timing
        warm_stripe = make_stripe(wc.Nt, stripe_height, nc, layout)
        func(warm_stripe, waveform, NF_START, stripe_height, nt_lim_of(wc), wc, tbl)
        if not np.allclose(warm_stripe, oracle, atol=atol, rtol=1e-9):
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
    return timings


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

    for stripe_height in (2, 3, 4, 5):
        waveform = build_waveform(wc, NF_START, stripe_height, nc)
        oracle = oracle_stripe(waveform, NF_START, stripe_height, nt_lim, wc, table)
        amp = float(np.max(np.abs(waveform.AT[:, nt_lim.nx_min:nt_lim.nx_max])))
        atol = 1e-10 * amp

        for layout in ('C', 'F'):
            timings = benchmark_layout(wc, table, aligned, waveform, oracle, stripe_height, layout, atol, nc)
            print(
                f'{stripe_height:>13} {layout:>7} '
                f'{timings["sparse"] * 1e3:>12.4f} {timings["dense"] * 1e3:>12.4f} {timings["aligned"] * 1e3:>12.4f}'
            )

    print()
    print('All benchmarked float64 stripe outputs matched the oracle within the faithful tolerance.')
    print('No speed threshold is enforced; results are evidence for human review.')


if __name__ == '__main__':
    main()
