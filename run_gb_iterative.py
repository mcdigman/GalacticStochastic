"""run iterative processing of galactic background"""

import configparser

import matplotlib.pyplot as plt
import numpy as np

import global_file_index as gfi
from iterative_fit_helpers import IterationConfig
from iterative_fit_manager import IterativeFitManager
from lisa_config import get_lisa_constants
from testing_tools import unit_normal_battery
from wdm_config import get_wavelet_model

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('default_parameters.ini')

    galaxy_file = config['files']['galaxy_file']
    galaxy_dir = config['files']['galaxy_dir']

    wc = get_wavelet_model(config)

    lc = get_lisa_constants(config)

    snr_thresh = 7.

    galactic_below_in, _, snr_tots_in, SAET_m, _, _, snr_min_in = gfi.load_init_galactic_file(galaxy_dir, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    for itrm in range(0,1):
        const_only = False
        nt_min = 256*(7-itrm)
        nt_max = nt_min+512*(itrm+1)
        print(nt_min, nt_max, wc.Nt, wc.Nf, const_only)

        params_gb, _, _, _, _ = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)

        # iteration to switch to cyclostationary noise model (using it too early may be noisy)
        n_cyclo_switch = 3

        smooth_lengthf_fix = 0.25
        fsmooth_settle_mult = 6
        fsmooth_settle_scale = 1.
        fsmooth_settle_offset = 0.
        fsmooth_fix_itr = 4

        snr_high_initial = 30.
        snr_high_fix = snr_thresh
        snr_high_settle_mult = 7.
        snr_high_settle_scale = 2./n_cyclo_switch
        snr_high_settle_offset = 4./n_cyclo_switch
        snr_high_fix_itr = 4

        snr_low_initial = snr_thresh
        snr_low_mult = 0.999

        n_iterations = 40
        n_const_force = 6

        const_converge_change_thresh = 3

        snr_cut_bright = np.zeros(n_iterations) + snr_high_fix
        for itrn in range(snr_high_fix_itr):
            snr_cut_bright[itrn] += snr_high_settle_mult*np.exp(-snr_high_settle_scale*itrn - snr_high_settle_offset)

        snr_cut_bright[0] = snr_high_initial

        # phase in the frequency smoothing length gradually
        # give absorbing constants a relative advantage on early iterations
        # because for the first iteration we included no galactic background

        smooth_lengthfs = np.zeros(n_iterations) + smooth_lengthf_fix
        for itrn in range(fsmooth_fix_itr):
            smooth_lengthfs[itrn] += fsmooth_settle_mult*np.exp(-fsmooth_settle_scale*itrn - fsmooth_settle_offset)


        snr_min = np.zeros(n_iterations)
        snr_min[0] = snr_low_initial               # for first iteration set to thresh because spectrum is just instrument noise
        snr_min[1:] = snr_low_mult * snr_high_fix  # for subsequent, choose value to ensure almost nothing gets decided as constant because of its own power alone

        if const_only:
            period_list = np.array([])
        else:
            period_list = np.array([1, 2, 3, 4, 5])

        # TODO move snr_min, snr_thresh, period_list, etc to init file


        ic = IterationConfig(n_iterations, snr_thresh, snr_min, snr_cut_bright, smooth_lengthfs)

        ifm = IterativeFitManager(lc, wc, ic, SAET_m, n_iterations, galactic_below_in, snr_tots_in, snr_min_in, params_gb, period_list, nt_min, nt_max, n_cyclo_switch, const_only, n_const_force, const_converge_change_thresh, smooth_lengthf_fix)

        params_gb = None

        ifm.do_loop()

        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(galaxy_dir, galaxy_file, ifm.wc, ifm.lc, ifm.ic, ifm.nt_min, ifm.nt_max, ifm.bgd, ifm.period_list, ifm.n_bin_use, ifm.SAET_m, ifm.SAET_fin, ifm.const_only, ifm.bis.snrs_tot_upper, ifm.n_full_converged, ifm.argbinmap, ifm.faints_old, ifm.bis.faints_cur, ifm.bis.brights, snr_min_in)



do_parseval_plot = False
if do_parseval_plot:
    plt.plot(ifm.parseval_const[1:itrn+1]/ifm.parseval_tot[1:itrn+1])
    plt.plot(ifm.parseval_bg[1:itrn+1]/ifm.parseval_tot[1:itrn+1])
    plt.plot(ifm.parseval_sup[1:itrn+1]/ifm.parseval_tot[1:itrn+1])
    plt.show()

plot_noise_spectrum_ambiguity = True
if plot_noise_spectrum_ambiguity:
    SAET_m_shift = SAET_m
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_upper.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_lower.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m_shift[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    #plt.legend(['initial', '1', '2', '3', '4', '5', '6', 'final'])
    # TODO handle if not all iterations complete
    plt.legend(['initial', '1', '2', '3', '4', '5', 'base'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()


plot_noise_spectrum_evolve = True
if plot_noise_spectrum_evolve:
    SAET_m_shift = SAET_m
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, (ifm.galactic_total.reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2).mean(axis=0).mean(axis=1)+SAET_m[1:, 0], 'k', alpha=0.3, zorder=-90)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.SAET_tots[[1, 2, 3, 4], :, 1:, 0], axis=1).T, '--', alpha=0.7)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(ifm.noise_upper.SAET[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m_shift[1:, 0], 'k--', zorder=-100)
    ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
    #plt.legend(['initial', '1', '2', '3', '4', '5', '6', 'final'])
    # TODO handle if not all iterations complete
    plt.legend(['initial', '1', '2', '3', '4', '5', 'base'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()

res_mask = (ifm.noise_upper.SAET[:, :, 0]-SAET_m[:, 0]).mean(axis=0) > 0.1*SAET_m[:, 0]
galactic_below_high = ifm.bgd.get_galactic_below_high()
unit_normal_res, _, _, _ = unit_normal_battery((galactic_below_high.reshape(wc.Nt, wc.Nf, wc.NC)[nt_min:nt_max, res_mask, 0:2]/np.sqrt(ifm.noise_upper.SAET[nt_min:nt_max, res_mask, 0:2]-SAET_m[res_mask, 0:2])).flatten(), A2_cut=10., sig_thresh=10.,do_assert=False)
if unit_normal_res:
    print('After iteration, final background PASSES normality tests')
else:
    print('After iteration, final background FAILS  normality tests')
