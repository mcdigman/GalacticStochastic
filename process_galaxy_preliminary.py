"""Get the waveform for a galaxy of galactic binaries"""

import configparser

import numpy as np

import global_const as gc
import global_file_index as gfi
from instrument_noise import (DiagonalStationaryDenseInstrumentNoiseModel,
                              instrument_noise_AET_wdm_m)
from iterative_fit_helpers import IterationConfig, do_preliminary_loop
from lisa_config import get_lisa_constants
from wavelet_detector_waveforms import BinaryWaveletAmpFreqDT
from wdm_config import get_wavelet_model

if __name__=='__main__':
    config = configparser.ConfigParser()
    config.read('default_parameters.ini')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    params_gb, _, _, _, n_tot = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

    params0 = params_gb[0].copy()

    waveT_ini = BinaryWaveletAmpFreqDT(params0.copy(), wc, lc)
    listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    noise_floor = DiagonalStationaryDenseInstrumentNoiseModel(SAET_m, wc, prune=False)

    noise_realization = noise_floor.generate_dense_noise()

    n_bin_use = n_tot

    n_iterations = 2
    SAET_tot = np.zeros((n_iterations+1, wc.Nt, wc.Nf, wc.NC))
    SAET_tot[0] = noise_floor.SAET.copy()

    snr_thresh = 7.
    snr_min = np.full(n_iterations, snr_thresh)
    snr_cut_bright = np.full(n_iterations, snr_thresh)
    snr_cut_bright[0] = 500.
    snrs_tot_upper = np.zeros((n_iterations, n_bin_use))

    faints_in = np.zeros(n_bin_use, dtype=np.bool_)

    galactic_below = np.zeros((wc.Nt*wc.Nf, wc.NC))

    smooth_lengthf = np.full(n_iterations, 8)

    ic = IterationConfig(n_iterations, snr_thresh, snr_min, snr_cut_bright, smooth_lengthf)

    galactic_below_high, galactic_below, signal_full, SAET_tot, brights, snrs_upper, snrs_tot_upper, noise_upper = do_preliminary_loop(wc, ic, SAET_tot, n_bin_use, faints_in, waveT_ini, params_gb, snrs_tot_upper, galactic_below, noise_realization, SAET_m)

    do_hf_write = True
    if do_hf_write:
        gfi.store_preliminary_gb_file(galaxy_dir, galaxy_file, wc, lc, ic, galactic_below, noise_realization, n_bin_use, SAET_m, snrs_tot_upper)

    plot_noise_spectrum_evolve = True
    if plot_noise_spectrum_evolve:
        import matplotlib.pyplot as plt

        plt.loglog(np.arange(0, wc.Nf)*wc.DF, SAET_m[:, 0])
        plt.loglog(np.arange(0, wc.Nf)*wc.DF, np.mean(SAET_tot[:, :, :, 0], axis=1).T)
        plt.xlabel('f (Hz)')
        plt.show()

    plot_bg_smooth = True
    if plot_bg_smooth:
        import matplotlib.pyplot as plt
        res_A = np.sqrt(SAET_tot[-1, :, :, 0]/SAET_m[:, 0])
        plt.imshow(np.rot90(np.log10(res_A[:, 0:wc.Nf//2])), aspect='auto', extent=[0, wc.Nt*wc.DT/gc.SECSYEAR, 0, wc.Nf//2*wc.DF])
        plt.xlabel('t (yr)')
        plt.ylabel('f (Hz)')
        plt.title(r"log10($S_A$), snr threshold="+str(snr_thresh))
        plt.show()

        res_E = np.sqrt(SAET_tot[-1, :, :, 1]/SAET_m[:, 1])
        plt.imshow(np.rot90(np.log10(res_E[:, 0:wc.Nf//2])), aspect='auto', extent=[0, wc.Nt*wc.DT/gc.SECSYEAR, 0, wc.Nf//2*wc.DF])
        plt.xlabel('t (yr)')
        plt.ylabel('f (Hz)')
        plt.title(r"log10($S_E$), snr threshold="+str(snr_thresh))
        plt.show()

    plot_bg = False
    if plot_bg:
        res = galactic_below_high[:, :, 0]
        res = res[:, :wc.Nf//2]
        mask = (res==0.)
        res[mask] = np.nan
        import matplotlib.pyplot as plt
        plt.imshow(np.rot90(res), aspect='auto')
        plt.show()

    plot_realization_im = False
    if plot_realization_im:
        import matplotlib.pyplot as plt
        mask = (galactic_below_high==0.)
        res = np.zeros_like(galactic_below_high)
        res[~mask] = np.log10(np.abs(galactic_below_high[~mask]))
        plt.imshow(np.rot90(res[:, :, 0][:, 1:200]), aspect='auto')
        plt.ylabel('frequency')
        plt.xlabel('time')
        plt.show()


    do_hist_plots = False
    if do_hist_plots:
        import matplotlib.pyplot as plt
        plt.hist(np.log10(params_gb[:, 0]), 100)
        plt.xlabel('log10(Amplitude)')
        plt.show()

        plt.hist(np.cos(params_gb[:, 1]), 100)
        plt.xlabel('cos(EclipticLatitude)')
        plt.show()

        plt.hist(params_gb[:, 2], 100)
        plt.xlabel('EclipticLongitude')
        plt.show()

        plt.hist(np.log10(params_gb[:, 3]), 100)
        plt.xlabel('log10(Frequency)')
        plt.show()

        plt.hist(np.log10(np.abs(params_gb[:, 4])), 100)
        plt.xlabel('log10(abs(FrequencyDerivative))')
        plt.show()

        plt.hist(np.cos(params_gb[:, 5]), 100)
        plt.xlabel('cosi')
        plt.show()

        plt.hist(params_gb[:, 6], 100)
        plt.xlabel('InitialPhase')
        plt.show()

        plt.hist(params_gb[:, 7], 100)
        plt.xlabel('Polarization')
        plt.show()
