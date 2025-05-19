"""helper functions for the iterative fit loops"""

from collections import namedtuple
from time import perf_counter

import numpy as np

from GalacticStochastic.galactic_fit_helpers import get_S_cyclo
from LisaWaveformTools.instrument_noise import DiagonalNonstationaryDenseInstrumentNoiseModel

IterationConfig = namedtuple('IterationConfig', ['max_iterations', 'snr_thresh', 'snr_min', 'snr_cut_bright', 'smooth_lengthf'])


def run_binary_coadd(itrb, faints_in, waveform_model, noise_upper, snrs_upper, snrs_tot_upper, itrn, galactic_below, galactic_undecided, brights, wc, params_gb, snr_min, snr_cut_bright) -> None:
    if not faints_in[itrb]:
        waveform_model.update_params(params_gb[itrb].copy())
        wavelet_waveform = waveform_model.get_unsorted_coeffs()
        snrs_upper[itrn, itrb] = noise_upper.get_sparse_snrs(wavelet_waveform)
        snrs_tot_upper[itrn, itrb] = np.linalg.norm(snrs_upper[itrn, itrb])
        if itrn == 0 and snrs_tot_upper[0, itrb] < snr_min:
            faints_in[itrb] = True
            for itrc in range(wc.NC):
                galactic_below[wavelet_waveform.pixel_index[itrc, :wavelet_waveform.N_set[itrc]], itrc] += wavelet_waveform.wave_value[itrc, :wavelet_waveform.N_set[itrc]]
        elif snrs_tot_upper[itrn, itrb] < snr_cut_bright:
            for itrc in range(wc.NC):
                galactic_undecided[wavelet_waveform.pixel_index[itrc, :wavelet_waveform.N_set[itrc]], itrc] += wavelet_waveform.wave_value[itrc, :wavelet_waveform.N_set[itrc]]
        else:
            brights[itrn, itrb] = True


def do_preliminary_loop(wc, ic, SAET_tot, n_bin_use, faints_in, waveform_model, params_gb, snrs_tot_upper, galactic_below, S_inst_m):
    # TODO make snr_cut_bright and smooth_lengthf an array as a function of iteration
    # TODO make NC controllable; probably not much point in getting T channel snrs
    snrs_upper = np.zeros((ic.max_iterations, n_bin_use, wc.NC))
    brights = np.zeros((ic.max_iterations, n_bin_use), dtype=np.bool_)

    for itrn in range(ic.max_iterations):
        galactic_undecided = np.zeros((wc.Nt * wc.Nf, wc.NC))
        noise_upper = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot[itrn], wc, prune=True)

        t0n = perf_counter()

        for itrb in range(n_bin_use):
            if itrb % 10000 == 0 and itrn == 0:
                tin = perf_counter()
                print("Starting binary # %11d of %11d at t=%9.2f s at iteration %4d" % (itrb, n_bin_use, (tin - t0n), itrn))

            run_binary_coadd(itrb, faints_in, waveform_model, noise_upper, snrs_upper, snrs_tot_upper, itrn, galactic_below, galactic_undecided, brights, wc, params_gb, ic.snr_min[itrn], ic.snr_cut_bright[itrn])

            assert np.all(np.isfinite(snrs_upper[itrn, itrb]))

        t1n = perf_counter()

        print('Finished coadd for iteration %4d at time %9.2f s' % ((itrn, t1n - t0n)))

        galactic_below_high = (galactic_undecided + galactic_below).reshape((wc.Nt, wc.Nf, wc.NC))

        SAET_tot[itrn + 1], _, _, _, _ = get_S_cyclo(galactic_below_high, S_inst_m, wc, ic.smooth_lengthf[itrn], False, period_list=())

    return galactic_below_high, galactic_below, SAET_tot, brights, snrs_upper, snrs_tot_upper, noise_upper
