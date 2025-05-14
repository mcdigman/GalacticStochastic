"""second step in processing the waveform from a galaxy of binaries"""

import configparser

import numpy as np

import GalacticStochastic.global_const as gc
import GalacticStochastic.global_file_index as gfi
from GalacticStochastic.iterative_fit_helpers import (IterationConfig,
                                                      do_preliminary_loop)
from LisaWaveformTools.instrument_noise import (
    DiagonalStationaryDenseInstrumentNoiseModel, instrument_noise_AET_wdm_m)
from LisaWaveformTools.lisa_config import get_lisa_constants
from WaveletWaveforms.wavelet_detector_waveforms import BinaryWaveletAmpFreqDT
from WaveletWaveforms.wdm_config import get_wavelet_model

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('default_parameters.ini')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    snr_thresh = 7.

    params_gb, _, _, _, n_tot = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

    params0 = params_gb[0].copy()

    galactic_below_in, noise_realization_common, snrs_tot_upper_in, _, lc = gfi.load_preliminary_galactic_file(galaxy_file, galaxy_dir, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    waveT_ini = BinaryWaveletAmpFreqDT(params0.copy(), wc, lc)
    listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    noise_floor = DiagonalStationaryDenseInstrumentNoiseModel(SAET_m, wc, prune=False)

    noise_realization = noise_realization_common[:wc.Nt, :, :].copy()

    n_bin_use = n_tot

    max_iterations = 2
    SAET_tot = np.zeros((max_iterations+1, wc.Nt, wc.Nf, wc.NC))
    SAET_tot[0] = noise_floor.SAET.copy()

    snr_min = np.full(max_iterations, snr_thresh)

    snr_cut_bright = np.full(max_iterations, snr_thresh)
    snr_cut_bright[0] = 500.

    snrs_tot_upper = np.zeros((max_iterations, n_bin_use))
    snrs_tot_upper[:] = snrs_tot_upper_in

    galactic_below = galactic_below_in.reshape(wc.Nt, wc.Nf, wc.NC)[:wc.Nt].reshape(wc.Nt*wc.Nf, wc.NC)

    galactic_below_in = None

    smooth_lengthf = np.full(max_iterations, 8)

    ic = IterationConfig(max_iterations, snr_thresh, snr_min, snr_cut_bright, smooth_lengthf)

    faints_in = snrs_tot_upper_in < ic.snr_min[0]

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

    plot_bg = True
    if plot_bg:
        res = galactic_below_high[:, :, 0]
        res = res[:, :wc.Nf//2]
        mask = res == 0.
        res[mask] = np.nan
        import matplotlib.pyplot as plt
        plt.imshow(np.rot90(res), aspect='auto')
        plt.show()
