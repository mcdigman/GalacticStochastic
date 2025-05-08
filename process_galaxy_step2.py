"""second step in processing the waveform from a galaxy of binaries"""

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

    snr_thresh = 7.

    params_gb, _, _, _, n_tot = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

    params0 = params_gb[0].copy()


    galactic_bg_const_in, noise_realization_common, snrs_tot_in, _, lc = gfi.load_preliminary_galactic_file(galaxy_file, galaxy_dir, snr_thresh, wc.Nf, wc.Nt, wc.dt)



    waveT_ini = BinaryWaveletAmpFreqDT(params0.copy(), wc, lc)
    listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    noise_AET_dense_pure = DiagonalStationaryDenseInstrumentNoiseModel(SAET_m, wc, prune=False)

    noise_realization = noise_realization_common[:wc.Nt, :, :].copy()

    n_bin_use = n_tot

    n_iterations = 2
    SAET_tot = np.zeros((n_iterations+1, wc.Nt, wc.Nf, wc.NC))
    SAET_tot[0] = noise_AET_dense_pure.SAET.copy()

    snr_min = np.full(n_iterations, snr_thresh)

    snr_autosuppress = np.full(n_iterations, snr_thresh)
    snr_autosuppress[0] = 500.

    snrs_tot = np.zeros((n_iterations, n_bin_use))
    snrs_tot[:] = snrs_tot_in

    galactic_bg_const = galactic_bg_const_in.reshape(wc.Nt, wc.Nf, wc.NC)[:wc.Nt].reshape(wc.Nt*wc.Nf, wc.NC)

    smooth_lengthf = np.full(n_iterations, 8)
    smooth_lengtht = np.full(n_iterations, 84*2)

    ic = IterationConfig(n_iterations, snr_thresh, snr_min, snr_autosuppress, smooth_lengthf, smooth_lengtht)

    const_suppress_in = snrs_tot_in < ic.snr_min[0]

    galactic_bg_full, galactic_bg_const, signal_full, SAET_tot, var_suppress, snrs, snrs_tot, noise_AET_dense = do_preliminary_loop(wc, ic, SAET_tot, n_bin_use, const_suppress_in, waveT_ini, params_gb, snrs_tot, galactic_bg_const, noise_realization, SAET_m)

    do_hf_write = True
    if do_hf_write:
        gfi.store_preliminary_gb_file(galaxy_dir, galaxy_file, wc, lc, ic, galactic_bg_const, noise_realization, n_bin_use, SAET_m, snrs_tot)
        #gfi.store_init_gb_file(galaxy_dir, galaxy_file, wc, lc, snr_thresh, snr_min, galactic_bg_const, noise_realization, smooth_lengthf, smooth_lengtht, n_iterations, n_bin_use, SAET_m, snrs_tot)

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
        res = galactic_bg_full[:, :, 0]
        res = res[:, :wc.Nf//2]
        mask = (res==0.)
        res[mask] = np.nan
        import matplotlib.pyplot as plt
        plt.imshow(np.rot90(res), aspect='auto')
        plt.show()
