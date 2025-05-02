"""run iterative processing of galactic background"""

from time import perf_counter

import numpy as np

import scipy.stats

from wavelet_detector_waveforms import BinaryWaveletAmpFreqDT
from instrument_noise import instrument_noise_AET_wdm_m, DiagonalNonstationaryDenseInstrumentNoiseModel

from galactic_fit_helpers import get_SAET_cyclostationary_mean
import global_file_index as gfi

import global_const as gc

from iterative_fit_helpers import IterationConfig, unit_normal_battery


if __name__ == '__main__':

    galaxy_file = 'galaxy_binaries.hdf5'
    galaxy_dir = 'Galaxies/Galaxy1/'

    snr_thresh = 7

    Nf = 2048
    Nt = 4096
    dt = 30.0750732421875


    filename_gb_init, snr_min_got, galactic_bg_const_in, noise_realization_got, smooth_lengthf_got, smooth_lengtht_got, n_iterations_got, snr_tots_in, SAET_m, wc, lc = gfi.load_init_galactic_file(galaxy_dir, snr_thresh, Nf, Nt, dt)

    for itrm in range(0, 1):
        const_only = False
        nt_min = 256*6
        nt_max = nt_min+2*512
        print(nt_min, nt_max, wc.Nt, wc.Nf, const_only)

        params_gb, n_dgb, n_igb, n_vgb, n_tot = gfi.get_full_galactic_params(galaxy_file, galaxy_dir)


        smooth_lengthf = 6
        smooth_lengtht = 0
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
        smooth_lengthfs[0] = smooth_lengthf_targ + smooth_lengthf*np.exp(-0)
        smooth_lengthfs[1] = smooth_lengthf_targ + smooth_lengthf*np.exp(-1)
        smooth_lengthfs[2] = smooth_lengthf_targ + smooth_lengthf*np.exp(-2)
        smooth_lengthfs[3] = smooth_lengthf_targ + smooth_lengthf*np.exp(-3)

        smooth_lengthts = np.zeros(n_iterations)


        snr_min = np.zeros(n_iterations)
        snr_min[0] = snr_thresh  # for first iteration set to thresh because spectrum is pure noise
        snr_min[1:] = 0.999*snr_thresh  # for subsequent, choose value to ensure almost nothing gets suppressed as constant because of its own power alone

        # TODO move snr_min, snr_thresh, period_list, etc to init file

        const_converge_change_thresh = 3

        ic = IterationConfig(n_iterations, snr_thresh, snr_min, snr_autosuppresses, smooth_lengthfs, smooth_lengthts)

        common_noise = True
        filename_gb_common, noise_realization_common = gfi.get_noise_common(galaxy_dir, snr_thresh, wc, lc)
        if common_noise:
            # get a common noise realization so results at different lengths are comparable
            assert wc.Nt <= noise_realization_common.shape[0]

        SAET_m_alt = np.zeros((wc.Nf, wc.NC))
        for itrc in range(0, 3):
            SAET_m_alt[:, itrc] = (noise_realization_common[:, :, itrc]**2).mean(axis=0)
        # first element isn't validated so don't necessarily expect it to be correct
        assert np.allclose(SAET_m[1:], SAET_m_alt[1:], atol=1.e-80, rtol=4.e-1)

        SAET_m_alt2 = instrument_noise_AET_wdm_m(lc, wc)
        assert np.allclose(SAET_m[1:], SAET_m_alt2[1:], atol=1.e-80, rtol=4.e-1)

        # check input SAET makes sense with noise realization

        galactic_bg_const = np.zeros_like(galactic_bg_const_in)#galactic_bg_const_in.copy()

        if common_noise:
            noise_realization = noise_realization_common[0:wc.Nt, :, :].copy() # TODO possibly needs to be an nt_min offset?
        else:
            noise_realization = noise_realization_got.copy()
        old_noise = False
        if old_noise:
            noise_realization = np.sqrt(wc.Tobs/(8*wc.Nt*wc.Nf))*noise_realization
        else:
            pass


        
        if const_only:
            period_list1 = np.array([])
        else:
            period_list1 = np.array([1, 2, 3, 4, 5])
        #iteration to switch to fitting spectrum fully

        #TODO eliminate if vgb included
        const_suppress_in = (snr_tots_in < snr_min_got[0]) | (params_gb[:, 3] >= (wc.Nf-1)*wc.DF)
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
        n_var_suppressed = var_suppress[0].sum()
        n_const_suppressed = const_suppress2[0].sum()
        var_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        const_converged = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        switch_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        switchf_next = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        force_converge = np.zeros(ic.n_iterations+1, dtype=np.bool_)
        n_full_converged = ic.n_iterations-1


        print('entered loop', itrm)
        ti = perf_counter()
        for itrn in range(1, ic.n_iterations):
            if switchf_next[itrn]:
                galactic_bg_const = np.zeros((wc.Nt*wc.Nf, wc.NC))
                const_suppress2[itrn] = False
            else:
                const_suppress2[itrn] = const_suppress2[itrn-1]

            if var_converged[itrn]:
                galactic_bg = galactic_bg.copy()
            else:
                galactic_bg = np.zeros((wc.Nt*wc.Nf, wc.NC))
                if switch_next[itrn]:
                    galactic_bg_suppress = np.zeros((wc.Nt*wc.Nf, wc.NC))
                    var_suppress[itrn] = False#var_suppress[itrn-1]
                else:
                    var_suppress[itrn] = var_suppress[itrn-1]

            t0n = perf_counter()

            # do the finishing step for itrn=0 to set everything at the end of the loop as it should be

            suppressed = var_suppress[itrn] | const_suppress2[itrn] | const_suppress

            idxbs = np.argwhere(~suppressed).flatten()
            for itrb in idxbs:
                if not suppressed[itrb]:
                    waveT_ini.update_params(params_gb[itrb].copy())
                    listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()
                    if not const_converged[itrn]:
                        snrs_base[itrn, itrb] = noise_AET_dense_base.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
                        snrs_tot_base[itrn, itrb] = np.linalg.norm(snrs_base[itrn, itrb])
                        thresh_base = (snrs_tot_base[itrn, itrb] < ic.snr_min[itrn])
                    else:
                        snrs_base[itrn, itrb] = snrs_base[itrn-1, itrb]
                        snrs_tot_base[itrn, itrb] = snrs_tot_base[itrn-1, itrb]
                        thresh_base = False

                    if not var_converged[itrn]:
                        snrs[itrn, itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp, nt_min, nt_max)
                        snrs_tot[itrn, itrb] = np.linalg.norm(snrs[itrn, itrb])
                        thresh_var = snrs_tot[itrn, itrb] >= ic.snr_autosuppress[itrn]
                    else:
                        snrs[itrn, itrb] = snrs[itrn-1, itrb]
                        snrs_tot[itrn, itrb] = snrs_tot[itrn-1, itrb]
                        thresh_var = False
                    if np.isnan(snrs_tot[itrn, itrb]) or np.isnan(snrs_tot_base[itrn, itrb]):
                        var_suppress_loc = False
                        const_suppress_loc = False
                        raise ValueError('nan detected in snr at '+str(itrn)+', ' + str(itrb))
                    elif thresh_var and thresh_base:
                        # satifisfied conditions to be eliminated in both directions so just keep it
                        var_suppress_loc = False
                        const_suppress_loc = False
                    elif thresh_var:
                        if snrs_tot[itrn, itrb] > snrs_tot_base[itrn, itrb]:
                            # handle case where snr ordering is wrong to prevent oscillation
                            var_suppress_loc = False
                        else:
                            var_suppress_loc = True
                        const_suppress_loc = False
                    elif thresh_base:
                        var_suppress_loc = False
                        const_suppress_loc = True
                    else:
                        var_suppress_loc = False
                        const_suppress_loc = False

                    var_suppress[itrn, itrb] = var_suppress_loc
                    const_suppress2[itrn, itrb] = const_suppress_loc

                    if not var_suppress_loc and not const_suppress_loc:
                        for itrc in range(0, 2):
                            galactic_bg[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
                    elif var_suppress_loc and not const_suppress_loc:
                        for itrc in range(0, 2):
                            galactic_bg_suppress[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
                    elif not var_suppress_loc and const_suppress_loc:
                        if itrn == 1:
                            const_suppress2[itrn, itrb] = False
                            const_suppress[itrb] = True
                            for itrc in range(0, 2):
                                galactic_bg_const_base[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
                        else:
                            for itrc in range(0, 2):
                                galactic_bg_const[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
                    else:
                        raise ValueError('impossible state')




            #carry forward any other snr values we still know
            if const_converged[itrn]:
                snrs_tot_base[itrn, suppressed] = snrs_tot_base[itrn-1, suppressed]
                snrs_base[itrn, suppressed] = snrs_base[itrn-1, suppressed]
            if var_converged[itrn]:
                snrs_tot[itrn, suppressed] = snrs_tot[itrn-1, suppressed]
                snrs[itrn, suppressed] = snrs[itrn-1, suppressed]

            if itrn == 1:
                assert np.all(galactic_bg_const == 0.)
                galactic_full_signal[:] = galactic_bg_const_base + galactic_bg_const + galactic_bg + galactic_bg_suppress
            else:
                #check all contributions to the total signal are tracked accurately
                assert np.allclose(galactic_full_signal, galactic_bg_const_base + galactic_bg_const + galactic_bg + galactic_bg_suppress, atol=1.e-300, rtol=1.e-6)


            t1n = perf_counter()

            if not var_converged[itrn]:
                galactic_bg_res = galactic_bg + galactic_bg_const + galactic_bg_const_base
                n_var_suppressed_new = var_suppress[itrn].sum()

                if itrn > 1 and (force_converge[itrn] or (np.all(var_suppress[itrn] == var_suppress[itrn-1]) or np.all(var_suppress[itrn] == var_suppress[itrn-2]) or np.all(var_suppress[itrn] == var_suppress[itrn-3]))):
                    assert n_var_suppressed == n_var_suppressed_new or force_converge[itrn] or np.all(var_suppress[itrn] == var_suppress[itrn-2]) or np.all(var_suppress[itrn] == var_suppress[itrn-3])
                    if switch_next[itrn]:
                        print('subtraction converged at ' + str(itrn))
                        var_converged[itrn+1] = True
                        switch_next[itrn+1] = False
                        const_converged[itrn+1] = True
                    else:
                        if (np.all(var_suppress[itrn] == var_suppress[itrn-2]) or np.all(var_suppress[itrn] == var_suppress[itrn-3])) and not np.all(var_suppress[itrn] == var_suppress[itrn-1]):
                            print('cycling detected at ' + str(itrn) + ', doing final check iteration aborting')
                            force_converge[itrn+1] = True
                        print('subtraction predicted initial converged at ' + str(itrn) + ' next iteration will be check iteration')
                        switch_next[itrn+1] = True
                        var_converged[itrn+1] = False
                else:
                    switch_next[itrn+1] = False

                    if itrn < n_cyclo_switch:
                        # TODO check this is being used appropriately
                        print('here')
                        SAET_tot_cur, _, _ = get_SAET_cyclostationary_mean(galactic_bg_res, SAET_m, wc, ic.smooth_lengthf[itrn], filter_periods=False, period_list=period_list1)
                    else:
                        SAET_tot_cur, _, _ = get_SAET_cyclostationary_mean(galactic_bg_res, SAET_m, wc, ic.smooth_lengthf[itrn], filter_periods=not const_only, period_list=period_list1)

                    noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_cur, wc, prune=True)
                    n_var_suppressed = n_var_suppressed_new
            else:
                switch_next[itrn+1] = False

            if itr_save < idx_SAE_save.size and itrn == idx_SAE_save[itr_save]:
                SAE_tots[itr_save] = SAET_tot_cur[:, :, :2]
                itr_save += 1

            if not const_converged[itrn+1] or switch_next[itrn+1]:
                if itrn < n_const_force:
                    SAET_tot_base, _, _ = get_SAET_cyclostationary_mean(galactic_bg_const + galactic_bg_const_base, SAET_m, wc, smooth_lengthf_targ, filter_periods=not const_only, period_list=period_list1)
                else:
                    SAET_tot_base, _, _ = get_SAET_cyclostationary_mean(galactic_bg_const + galactic_bg_const_base, SAET_m, wc, smooth_lengthf_targ, filter_periods=not const_only, period_list=period_list1)
                    const_converged[itrn+1] = True
                    # need to disable adaption of constant here because after this point the convergence isn't guaranteed to be monotonic
                    print('disabled constant adaptation at ' + str(itrn))

                # make sure this will always predict >= snrs to the actual spectrum in use
                SAET_tot_base = np.min([SAET_tot_base, SAET_tot_cur], axis=0)
                noise_AET_dense_base = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_base, wc, prune=True)

                n_const_suppressed_new = const_suppress2[itrn].sum()
                if switch_next[itrn+1] and const_converged[itrn+1]:
                    print('overriding constant convergence to check background model')
                    const_converged[itrn+1] = False
                    switchf_next[itrn+1] = False
                elif n_const_suppressed_new - n_const_suppressed < 0:
                    if var_converged[itrn+1]:
                        switch_next[itrn+1] = True
                        var_converged[itrn+1] = False
                    switchf_next[itrn+1] = False
                    const_converged[itrn+1] = False
                    print('addition removed values at ' + str(itrn) + ', repeating check iteration')

                elif itrn != 1 and np.abs(n_const_suppressed_new - n_const_suppressed) < const_converge_change_thresh:
                    if switchf_next[itrn+1]:
                        const_converged[itrn+1] = True
                        switchf_next[itrn+1] = False
                        print('addition converged at ' + str(itrn))
                    else:
                        print('near convergence in constant adaption at '+str(itrn), ' doing check iteration')
                        switchf_next[itrn+1] = False
                        const_converged[itrn+1] = False
                else:
                    if var_converged[itrn+1]:
                        print('addition convergence continuing beyond subtraction, try check iteration')
                        switchf_next[itrn+1] = False
                        const_converged[itrn+1] = False
                    else:
                        switchf_next[itrn+1] = False

                n_const_suppressed = n_const_suppressed_new
            else:
                switchf_next[itrn+1] = False

            if switchf_next[itrn+1]:
                assert not const_converged[itrn+1]
            if switch_next[itrn+1]:
                assert not var_converged[itrn+1]


            parseval_tot[itrn] = np.sum((galactic_bg_const_base+galactic_bg_const+galactic_bg+galactic_bg_suppress).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])
            parseval_bg[itrn] = np.sum((galactic_bg_const_base+galactic_bg_const+galactic_bg).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])
            parseval_const[itrn] = np.sum((galactic_bg_const_base+galactic_bg_const).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])
            parseval_sup[itrn] = np.sum((galactic_bg_suppress).reshape((wc.Nt, wc.Nf, wc.NC))[:, 1:, 0:2]**2/SAET_m[1:, 0:2])

            t2n = perf_counter()
            print('made bg %3d in time %7.3fs fit time %7.3fs' % (itrn, t1n-t0n, t2n-t1n))



            if var_converged[itrn+1] and const_converged[itrn+1]:
                print('result fully converged at '+str(itrn)+', no further iterations needed')
                n_full_converged = itrn
                break

        SAE_fin[:] = SAET_tot_cur[:, :, :2]


        do_hf_out = True
        if do_hf_out:
            gfi.store_processed_gb_file(galaxy_dir, galaxy_file, wc, lc, snr_thresh, ic.snr_min, nt_min, nt_max, ic.smooth_lengtht[0], ic.smooth_lengthf[0], galactic_bg_const, galactic_bg_const_base, galactic_bg_suppress, galactic_bg, period_list1, ic.n_iterations, n_bin_use, SAET_m, SAE_fin, const_only, snrs_tot, n_full_converged, argbinmap, const_suppress, const_suppress2, var_suppress, filename_gb_init, filename_gb_common)

        tf = perf_counter()
        print('loop time = %.3es' % (tf-ti))


import matplotlib.pyplot as plt

do_plot_SAET_m = False
if do_plot_SAET_m:
    spec_shift = np.mean((galactic_bg_res.reshape(wc.Nt, wc.Nf, wc.NC)+noise_realization)[:, :, 0:2]**2, axis=0)
    SAET_m_shift = SAET_m
    print(spec_shift[766]/SAET_m_shift[766, 0])
    plt.loglog(np.arange(1, wc.Nf)*wc.DF, SAET_m_shift[1:, 0])
    plt.loglog(np.arange(1, wc.Nf)*wc.DF, np.mean(SAE_tots[0:, :, 1:, 0], axis=1).T)
    plt.ylim(2.e-44, 8.e-41)
    plt.xlim(1.e-4, 1.e-2)
    plt.xlabel('f (Hz)')
    plt.show()


do_parseval_plot = False
if do_parseval_plot:
    plt.plot(parseval_const[1:itrn+1]/parseval_tot[1:itrn+1])
    plt.plot(parseval_bg[1:itrn+1]/parseval_tot[1:itrn+1])
    plt.plot(parseval_sup[1:itrn+1]/parseval_tot[1:itrn+1])
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
    plt.legend(['initial', '1', '2', '3', '4', '5', '6', 'final'])
    plt.ylim([2.e-44, 4.e-43])
    plt.xlim([3.e-4, 6.e-3])
    plt.xlabel('f (Hz)')
    plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
    plt.show()

res_mask = (SAET_tot_cur[:, :, 0]-SAET_m[:, 0]).mean(axis=0) > 0.1*SAET_m[:, 0]
unit_normal_res, _, _, _ = unit_normal_battery((galactic_bg_res.reshape(wc.Nt, wc.Nf, wc.NC)[nt_min:nt_max, res_mask, 0:2]/np.sqrt(SAET_tot_cur[nt_min:nt_max, res_mask, 0:2]-SAET_m[res_mask, 0:2])).flatten(), A2_cut=10., sig_thresh=10.,do_assert=False)
if unit_normal_res:
    print('After iteration, final background PASSES normality tests')
else:
    print('After iteration, final background FAILS  normality tests')
