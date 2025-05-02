"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

from wavelet_detector_waveforms import BinaryWaveletAmpFreqDT
from instrument_noise import instrument_noise_AET_wdm_m, DiagonalNonstationaryDenseInstrumentNoiseModel

import global_file_index as gfi

import global_const as gc

from iterative_fit_helpers import IterationConfig, unit_normal_battery, sustain_snr_helper, run_binary_coadd2, total_signal_consistency_check, subtraction_convergence_decision, addition_convergence_decision, BGDecomposition


if __name__ == '__main__':

    galaxy_file = 'galaxy_binaries.hdf5'
    galaxy_dir = 'Galaxies/Galaxy4/'

    snr_thresh = 7

    from wdm_const import wdm_const as wc

    galactic_bg_const_in, noise_realization_got, snr_tots_in, SAET_m, wc, lc, ic_preliminary = gfi.load_init_galactic_file(galaxy_dir, snr_thresh, wc.Nf, wc.Nt, wc.dt)

    for itrm in range(1):
        const_only = True
        nt_min = 256*6
        nt_max = nt_min+2*512
        print(nt_min, nt_max, wc.Nt, wc.Nf, const_only)

        params_gb, n_dgb, n_igb, n_vgb, n_tot = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)


        smooth_lengthf_long = 6

        n_iterations = 40
        # iteration to switch to cyclostationary noise model (using it too early may be noisy)
        n_cyclo_switch = 3
        n_const_force = 6

        snr_autosuppresses = np.zeros(n_iterations) + snr_thresh
        snr_autosuppresses[0] = 30
        snr_autosuppresses[1] = (1. + np.exp(-6/n_cyclo_switch))*snr_thresh
        snr_autosuppresses[2] = (1. + np.exp(-8/n_cyclo_switch))*snr_thresh
        snr_autosuppresses[3] = (1. + np.exp(-10/n_cyclo_switch))*snr_thresh

        smooth_lengthf_targ = 0.25

        # give absorbing constants a relative advantage on early iterations
        # because for the first iteration we included no galactic background

        smooth_lengthfs = np.zeros(n_iterations) + smooth_lengthf_targ
        smooth_lengthfs[0] = smooth_lengthf_targ + smooth_lengthf_long*np.exp(-0)
        smooth_lengthfs[1] = smooth_lengthf_targ + smooth_lengthf_long*np.exp(-1)
        smooth_lengthfs[2] = smooth_lengthf_targ + smooth_lengthf_long*np.exp(-2)
        smooth_lengthfs[3] = smooth_lengthf_targ + smooth_lengthf_long*np.exp(-3)

        smooth_lengthts = np.zeros(n_iterations)


        snr_min = np.zeros(n_iterations)
        snr_min[0] = 7.#snr_thresh  # for first iteration set to thresh because spectrum is pure noise
        snr_min[1:] = 0.999*snr_min[0]#snr_thresh  # for subsequent, choose value to ensure almost nothing gets suppressed as constant because of its own power alone

        # TODO move snr_min, snr_thresh, period_list, etc to init file

        const_converge_change_thresh = 3

        ic = IterationConfig(n_iterations, snr_thresh, snr_min, snr_autosuppresses, smooth_lengthfs, smooth_lengthts)

        common_noise = True
        noise_realization_common = gfi.get_noise_common(galaxy_dir, ic_preliminary.snr_thresh, wc, lc)
        if common_noise:
            # get a common noise realization so results at different lengths are comparable
            assert wc.Nt <= noise_realization_common.shape[0]

        SAET_m_alt = np.zeros((wc.Nf, wc.NC))
        for itrc in range(3):
            SAET_m_alt[:, itrc] = (noise_realization_common[:, :, itrc]**2).mean(axis=0)
        # first element isn't validated so don't necessarily expect it to be correct
        assert np.allclose(SAET_m[1:], SAET_m_alt[1:], atol=1.e-80, rtol=4.e-1)

        SAET_m_alt2 = instrument_noise_AET_wdm_m(lc, wc)
        assert np.allclose(SAET_m[1:], SAET_m_alt2[1:], atol=1.e-80, rtol=4.e-1)

        # check input SAET makes sense with noise realization

        galactic_bg_const = np.zeros_like(galactic_bg_const_in)

        if common_noise:
            noise_realization = noise_realization_common[0:wc.Nt, :, :].copy() # TODO possibly needs to be an nt_min offset?
        else:
            noise_realization = noise_realization_got.copy()

        if const_only:
            period_list1 = np.array([])
        else:
            period_list1 = np.array([1, 2, 3, 4, 5])
        #iteration to switch to fitting spectrum fully

        #TODO eliminate if vgb included
        const_suppress_in = (snr_tots_in < ic_preliminary.snr_min[0]) | (params_gb[:, 3] >= (wc.Nf-1)*wc.DF)
        argbinmap = np.argwhere(~const_suppress_in).flatten()
        const_suppress = const_suppress_in[argbinmap]
        params_gb = params_gb[argbinmap]
        n_bin_use = argbinmap.size

        snrs_tot = np.zeros((n_iterations, n_bin_use))
        idx_SAE_save = np.hstack([np.arange(0, min(10, n_iterations)), np.arange(min(10, n_iterations), 4), n_iterations-1])
        itr_save = 0

        SAE_tots = np.zeros((idx_SAE_save.size, wc.Nt, wc.Nf, 2))
        SAE_fin = np.zeros((wc.Nt, wc.Nf, 2))

        snrs = np.zeros((n_iterations, n_bin_use, wc.NC))
        snrs_base = np.zeros((n_iterations, n_bin_use, wc.NC))
        snrs_tot_base = np.zeros((n_iterations, n_bin_use))
        var_suppress = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)
        suppressed = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)

        const_suppress2 = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)
        parseval_const = np.zeros(n_iterations)
        parseval_bg = np.zeros(n_iterations)
        parseval_sup = np.zeros(n_iterations)
        parseval_tot = np.zeros(n_iterations)


        params0 = params_gb[0].copy()
        waveT_ini = BinaryWaveletAmpFreqDT(params0.copy(), wc, lc)
        listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()

        SAET_tot_cur = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_tot_cur[:] = SAET_m

        SAET_tot_base = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_tot_base[:] = SAET_m
        if idx_SAE_save[itr_save] == 0:
            SAE_tots[0] = SAET_tot_cur[:, :, :2]
            itr_save += 1
        SAET_tot_base = np.min([SAET_tot_base, SAET_tot_cur], axis=0)

        noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_cur, wc, prune=True)
        noise_AET_dense_base = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_base, wc, prune=True)

        galactic_bg_const_base = galactic_bg_const_in.copy()
        galactic_full_signal = np.zeros((wc.Nt*wc.Nf, wc.NC))
        galactic_bg_suppress = np.zeros((wc.Nt*wc.Nf, wc.NC))
        galactic_bg = np.zeros((wc.Nt*wc.Nf, wc.NC))
        n_const_suppressed = np.zeros(ic.n_iterations+1, dtype=np.int64)
        n_const_suppressed[1] = const_suppress2[0].sum()
        n_var_suppressed = np.zeros(ic.n_iterations+1, dtype=np.int64)
        n_var_suppressed[1] = var_suppress[0].sum()
        var_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        const_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        switch_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        switchf_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        force_converge = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        n_full_converged = ic.n_iterations-1

        bgd = BGDecomposition(galactic_bg_const_base, galactic_bg_const, galactic_bg, galactic_bg_suppress)


        print('entered loop', itrm)
        ti = perf_counter()
        for itrn in range(1, ic.n_iterations):
            if switchf_next[itrn]:
                bgd.galactic_bg_const[:] = 0.
                const_suppress2[itrn] = False
            else:
                #bgd.galactic_bg_const = galactic_bg_const
                const_suppress2[itrn] = const_suppress2[itrn-1]

            if var_converged[itrn]:
                #bgd.galactic_bg = bgd.galactic_bg
                #bgd.galactic_bg_suppress = bgd.galactic_bg_suppress
                var_suppress[itrn] = var_suppress[itrn-1]
            else:
                bgd.galactic_bg[:] = 0.
                if switch_next[itrn]:
                    bgd.galactic_bg_suppress[:] = 0.
                    var_suppress[itrn] = False
                else:
                    #bgd.galactic_bg_suppress = bgd.galactic_bg_suppress
                    var_suppress[itrn] = var_suppress[itrn-1]

            t0n = perf_counter()

            # do the finishing step for itrn=0 to set everything at the end of the loop as it should be

            suppressed[itrn] = var_suppress[itrn] | const_suppress2[itrn] | const_suppress

            idxbs = np.argwhere(~suppressed[itrn]).flatten()
            for itrb in idxbs:
                if not suppressed[itrn, itrb]:
                    run_binary_coadd2(waveT_ini, params_gb, var_suppress, const_suppress, const_suppress2, snrs_base, snrs, snrs_tot, snrs_tot_base, itrn, itrb, noise_AET_dense, noise_AET_dense_base, ic, const_converged, var_converged, nt_min, nt_max, bgd)

            # copy forward prior calculations of snr calculations that were skipped in this loop iteration
            sustain_snr_helper(const_converged, snrs_tot_base, snrs_base, snrs_tot, snrs, itrn, suppressed[itrn], var_converged)

            # sanity check that the total signal does not change regardless of what bucket the binaries are allocated to
            total_signal_consistency_check(galactic_full_signal, bgd, itrn)


            t1n = perf_counter()

            noise_AET_dense, SAET_tot_cur = subtraction_convergence_decision(bgd, var_suppress, itrn, force_converge, n_var_suppressed, switch_next, var_converged, const_converged, SAET_m, wc, ic, period_list1, const_only, noise_AET_dense, n_cyclo_switch, SAET_tot_cur)

            if itr_save < idx_SAE_save.size and itrn == idx_SAE_save[itr_save]:
                SAE_tots[itr_save] = SAET_tot_cur[:, :, :2]
                itr_save += 1

            noise_AET_dense_base, SAET_tot_base = addition_convergence_decision(bgd, itrn, n_const_suppressed, switch_next, var_converged, switchf_next, const_converged, SAET_m, wc, period_list1, const_only, noise_AET_dense_base, SAET_tot_cur, SAET_tot_base, n_const_force, const_converge_change_thresh, const_suppress2, smooth_lengthf_targ)

            if var_converged[itrn]:
                assert np.all(var_suppress[itrn] == var_suppress[itrn-1])

            if const_converged[itrn]:
                assert np.all(const_suppress2[itrn] == const_suppress2[itrn-1])

            if switchf_next[itrn+1]:
                assert not const_converged[itrn+1]
            if switch_next[itrn+1]:
                assert not var_converged[itrn+1]

            #assert not np.any(var_suppress[itrn] & const_suppress2[itrn])

            parseval_tot[itrn] = np.sum((bgd.galactic_bg_const_base+bgd.galactic_bg_const+bgd.galactic_bg+bgd.galactic_bg_suppress).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])
            parseval_bg[itrn] = np.sum((bgd.galactic_bg_const_base+bgd.galactic_bg_const+bgd.galactic_bg).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])
            parseval_const[itrn] = np.sum((bgd.galactic_bg_const_base+bgd.galactic_bg_const).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])
            parseval_sup[itrn] = np.sum((bgd.galactic_bg_suppress).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])

            t2n = perf_counter()
            print('made bg %3d in time %7.3fs fit time %7.3fs' % (itrn, t1n-t0n, t2n-t1n))

            if var_converged[itrn+1] and const_converged[itrn+1]:
                print('result fully converged at '+str(itrn)+', no further iterations needed')
                n_full_converged = itrn
                break

        SAE_fin[:] = SAET_tot_cur[:, :, :2]


        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(galaxy_dir, galaxy_file, wc, lc, ic, nt_min, nt_max, bgd, period_list1, n_bin_use, SAET_m, SAE_fin, const_only, snrs_tot, n_full_converged, argbinmap, const_suppress, const_suppress2, var_suppress, ic_preliminary)

        tf = perf_counter()
        print('loop time = %.3es' % (tf-ti))

        Tobs_consider_yr = (nt_max - nt_min)*wc.DT/gc.SECSYEAR
        n_consider = n_bin_use
        n_faint = const_suppress.sum()
        n_faint2 = const_suppress2[itrn].sum()
        n_bright = var_suppress[itrn].sum()
        n_ambiguous = (~(const_suppress[itrn] | var_suppress[itrn])).sum()
        print('Out of %10d total binaries, %10d were deemed undetectable by a previous evaluation, %10d were considered here.' % (n_tot, n_tot - n_consider, n_consider))
        print('The iterative procedure deemed (%5.3f yr observation at threshold snr=%5.3f):' % (Tobs_consider_yr, snr_thresh))
        print('       %10d undetectable due to instrument noise' % n_faint)
        print('       %10d undetectable dut to galactic confusion' % n_faint2)
        print('       %10d undecided (presumed undetectable)' % n_ambiguous)
        print('       %10d total undetectable' % (n_tot - n_bright))
        print('       %10d total detectable' % n_bright)

import matplotlib.pyplot as plt


do_parseval_plot = False
if do_parseval_plot:
    plt.plot(parseval_const[1:itrn+1]/parseval_tot[1:itrn+1])
    plt.plot(parseval_bg[1:itrn+1]/parseval_tot[1:itrn+1])
    plt.plot(parseval_sup[1:itrn+1]/parseval_tot[1:itrn+1])
    plt.show()

plot_noise_spectrum_ambiguity = True
if plot_noise_spectrum_ambiguity:
    SAET_m_shift = SAET_m
    fig = plt.figure(figsize=(5.4, 3.5))
    ax = fig.subplots(1)
    fig.subplots_adjust(wspace=0., hspace=0., left=0.13, top=0.99, right=0.99, bottom=0.12)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(SAET_tot_cur[:, 1:, 0:2], axis=0).mean(axis=1).T)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(SAET_tot_base[:, 1:, 0:2], axis=0).mean(axis=1).T)
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
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, (galactic_full_signal.reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2).mean(axis=0).mean(axis=1)+SAET_m[1:, 0], 'k', alpha=0.3, zorder=-90)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(SAE_tots[[1, 2, 3, 4], :, 1:, 0], axis=1).T, '--', alpha=0.7)
    ax.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(SAET_tot_cur[:, 1:, 0:2], axis=0).mean(axis=1).T)
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

res_mask = (SAET_tot_cur[:, :, 0]-SAET_m[:, 0]).mean(axis=0) > 0.1*SAET_m[:, 0]
galactic_bg_res = bgd.galactic_bg + bgd.galactic_bg_const + bgd.galactic_bg_const_base
unit_normal_res, _, _, _ = unit_normal_battery((galactic_bg_res.reshape(wc.Nt, wc.Nf, wc.NC)[nt_min:nt_max, res_mask, 0:2]/np.sqrt(SAET_tot_cur[nt_min:nt_max, res_mask, 0:2]-SAET_m[res_mask, 0:2])).flatten(), A2_cut=10., sig_thresh=10.,do_assert=False)
if unit_normal_res:
    print('After iteration, final background PASSES normality tests')
else:
    print('After iteration, final background FAILS  normality tests')
