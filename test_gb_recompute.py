"""scratch to test processing of galactic background"""

from time import perf_counter

import numpy as np
import h5py

import scipy.stats

from binary_search_subs import BinaryWaveletAmpFreqDT
from instrument_noise import instrument_noise_AET_wdm_m,DiagonalNonstationaryDenseInstrumentNoiseModel

from galactic_fit_helpers import get_SAET_cyclostationary_mean
import global_file_index as gfi

import global_const as gc

from wdm_const import wdm_const as wc
from wdm_const import lisa_const as lc

TobsYEAR = wc.Tobs/gc.SECSYEAR

def unit_normal_battery(signal,mult=1.,sig_thresh=5.,A2_cut=2.28):
    """battery of tests for checking if signal is unit normal white noise"""
    #default anderson darling cutoff of 2.28 is hand selected to
    #give ~1 in 1e5 empirical probablity of false positive for n=64
    #calibration looks about same for n=32 could probably choose better way
    #with current defaults that should make it the most sensitive test
    n_sig = signal.size
    if n_sig == 0:
        return 0.

    sig_adjust = signal/mult
    mean_wave = np.mean(sig_adjust)
    std_wave = np.std(sig_adjust)
    std_std_wave = np.std(sig_adjust)*np.sqrt(2/n_sig)
    #check mean and variance
    assert np.abs(mean_wave)/std_wave<sig_thresh
    assert np.abs(std_wave-1.)/std_std_wave<sig_thresh

    #anderson darling test statistic assuming true mean and variance are unknown
    sig_sort = np.sort((sig_adjust-mean_wave)/std_wave)
    phis = scipy.stats.norm.cdf(sig_sort)
    A2 = -n_sig-1/n_sig*np.sum((2*np.arange(1,n_sig+1)-1)*np.log(phis)+(2*(n_sig-np.arange(1,n_sig+1))+1)*np.log(1-phis))
    A2Star = A2*(1+4/n_sig-25/n_sig**2)
    print(A2Star,A2_cut)
    assert A2Star<A2_cut #should be less than cutoff value

    return A2Star

if __name__=='__main__':
    for itrm in range(0,1):
        const_only = False
        nt_min = 256*6
        nt_max = nt_min+2*512
        print(nt_min,nt_max,wc.Nt,wc.Nf,const_only)

        params_gb,n_dgb,n_igb,n_vgb,n_tot = gfi.get_full_galactic_params(fmin=1.e-4)
        params_gb = params_gb[:100000]

        snr_thresh = 7

        smooth_lengthf = 6
        smooth_lengtht = 0
        n_iterations = 40
        #iteration to switch to cyclostationary noise model (using it too early may be noisy)
        n_cyclo_switch = 3
        n_const_force = 6

        snr_autosuppresses = np.zeros(n_iterations)+snr_thresh
        snr_autosuppresses[0] = 30
        snr_autosuppresses[1] = (1.+np.exp(-6/n_cyclo_switch))*snr_thresh
        snr_autosuppresses[2] = (1.+np.exp(-8/n_cyclo_switch))*snr_thresh
        snr_autosuppresses[3] = (1.+np.exp(-10/n_cyclo_switch))*snr_thresh

        smooth_lengthf_targ = 0.25

        smooth_lengthfs = np.zeros(n_iterations)+smooth_lengthf_targ
        smooth_lengthfs[0] = smooth_lengthf_targ+smooth_lengthf*np.exp(-0)
        smooth_lengthfs[1] = smooth_lengthf_targ+smooth_lengthf*np.exp(-1)
        smooth_lengthfs[2] = smooth_lengthf_targ+smooth_lengthf*np.exp(-2)
        smooth_lengthfs[3] = smooth_lengthf_targ+smooth_lengthf*np.exp(-3)
        smooth_lengthf_cur = smooth_lengthfs[0]

        snr_autosuppress = snr_autosuppresses[0]

        const_converge_change_thresh = 3

        common_noise = True
        filename_gb_common,noise_realization_common = gfi.get_noise_common(snr_thresh,Nf=wc.Nf,Nt=gfi.global_max_Nt,dt=wc.dt)
        if common_noise:
            #get a common noise realization so results at different lengths are comparable
            assert wc.Nt<=noise_realization_common.shape[0]

        filename_gb_init,snr_min_got,galactic_bg_const_in,noise_realization_got,smooth_lengthf_got,smooth_lengtht_got,n_iterations_got,snr_tots_in,SAET_m = gfi.load_init_galactic_file(snr_thresh,Nf=wc.Nf,Nt=wc.Nt,dt=wc.dt)
        SAET_m_alt = np.zeros((wc.Nf,wc.NC))
        for itrc in range(0, 3):
            SAET_m_alt[:,itrc] = (noise_realization_common[:,:,itrc]**2).mean(axis=0)
        #first element isn't validated so don't necessarily expect it to be correct
        assert np.allclose(SAET_m[1:],SAET_m_alt[1:],atol=1.e-80,rtol=4.e-1)

        SAET_m = instrument_noise_AET_wdm_m()
        assert np.allclose(SAET_m[1:],SAET_m_alt[1:],atol=1.e-80,rtol=4.e-1)

        #check input SAET makes sense with noise realization

        galactic_bg_const = np.zeros_like(galactic_bg_const_in)#galactic_bg_const_in.copy()

        if common_noise:
            noise_realization = noise_realization_common[0:wc.Nt,:,:].copy() #TODO possibly needs to be an nt_min offset?
        else:
            noise_realization = noise_realization_got.copy()
        old_noise = False
        if old_noise:
            noise_realization = np.sqrt(wc.Tobs/(8*wc.Nt*wc.Nf))*noise_realization
        else:
            pass

        snr_min = snr_thresh #for first iteration set to thresh because spectrum is pure noise

        if const_only:
            period_list1 = np.array([])
        else:
            period_list1 = np.array([1,2,3,4,5])
        #iteration to switch to fitting spectrum fully

        #TODO eliminate if vgb included
        const_suppress_in = (snr_tots_in<snr_min_got)|(params_gb[:,3]>=(wc.Nf-1)*wc.DF)
        argbinmap = np.argwhere(~const_suppress_in).flatten()
        const_suppress = const_suppress_in[argbinmap]
        params_gb = params_gb[argbinmap]
        n_bin_use = argbinmap.size

        snrs_tot = np.zeros((n_iterations,n_bin_use))
        idx_SAE_save = np.hstack([np.arange(0,min(10,n_iterations)),np.arange(min(10,n_iterations),4),n_iterations-1])
        itr_save = 0

        SAE_tots = np.zeros((idx_SAE_save.size,wc.Nt,wc.Nf,2))
        SAE_fin = np.zeros((wc.Nt,wc.Nf,2))

        snrs = np.zeros((n_iterations,n_bin_use,wc.NC))
        snrs_base = np.zeros((n_iterations,n_bin_use,wc.NC))
        snrs_tot_base = np.zeros((n_iterations,n_bin_use))
        var_suppress = np.zeros((n_iterations,n_bin_use),dtype=np.bool_)

        const_suppress2 = np.zeros((n_iterations,n_bin_use),dtype=np.bool_)
        parseval_const = np.zeros(n_iterations)
        parseval_bg = np.zeros(n_iterations)
        parseval_sup = np.zeros(n_iterations)
        parseval_tot = np.zeros(n_iterations)


        params0 = params_gb[0].copy()
        waveT_ini = BinaryWaveletAmpFreqDT(params0.copy())
        listT_temp,waveT_temp,NUTs_temp = waveT_ini.get_unsorted_coeffs()

        SAET_tot_cur = np.zeros((wc.Nt,wc.Nf,wc.NC))
        SAET_tot_cur[:] = SAET_m

        SAET_tot_base = np.zeros((wc.Nt,wc.Nf,wc.NC))
        SAET_tot_base[:] = SAET_m
        if idx_SAE_save[itr_save]==0:
            SAE_tots[0] = SAET_tot_cur[:,:,:2]
            itr_save += 1
        SAET_tot_base = np.min([SAET_tot_base,SAET_tot_cur],axis=0)

        noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_cur,prune=True)
        noise_AET_dense_base = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_base,prune=True)

        galactic_bg_const_base = galactic_bg_const_in.copy()
        galactic_full_signal = np.zeros((wc.Nt*wc.Nf,wc.NC))
        galactic_bg_suppress = np.zeros((wc.Nt*wc.Nf,wc.NC))
        n_var_suppressed = var_suppress[0].sum()
        n_const_suppressed = const_suppress2[0].sum()
        var_converged = False
        const_converged = False
        switch_next = False
        switchf_next = False
        force_converge = False
        n_full_converged = n_iterations-1


        print('entered loop',itrm)
        ti = perf_counter()
        for itrn in range(1,n_iterations):
            if switchf_next:
                galactic_bg_const = np.zeros((wc.Nt*wc.Nf,wc.NC))#galactic_bg_const_full.copy()
                const_suppress2[itrn] = False
            else:
                const_suppress2[itrn] = const_suppress2[itrn-1]

            if var_converged:
                galactic_bg = galactic_bg.copy()
            else:
                galactic_bg = np.zeros((wc.Nt*wc.Nf,wc.NC))
                if switch_next:
                    galactic_bg_suppress = np.zeros((wc.Nt*wc.Nf,wc.NC))
                    var_suppress[itrn] = False#var_suppress[itrn-1]
                else:
                    var_suppress[itrn] = var_suppress[itrn-1]

            t0n = perf_counter()
            #do the finishing step for itrn=0 to set everything at the end of the loop as it should be

            suppressed = var_suppress[itrn]|const_suppress2[itrn]|const_suppress

            idxbs = np.argwhere(~suppressed).flatten()
            for itrb in idxbs:
                #TODO can block for var suppress if timing of later iterations becomes an issue
                if not suppressed[itrb]:
                    waveT_ini.update_params(params_gb[itrb].copy())
                    listT_temp,waveT_temp,NUTs_temp = waveT_ini.get_unsorted_coeffs()
                    if not const_converged:
                        snrs_base[itrn,itrb] = noise_AET_dense_base.get_sparse_snrs(NUTs_temp,listT_temp,waveT_temp,nt_min,nt_max)
                        snrs_tot_base[itrn,itrb] = np.linalg.norm(snrs_base[itrn,itrb])
                        thresh_base = (snrs_tot_base[itrn,itrb]<snr_min)
                    else:
                        snrs_base[itrn,itrb] = snrs_base[itrn-1,itrb]
                        snrs_tot_base[itrn,itrb] = snrs_tot_base[itrn-1,itrb]
                        thresh_base = False

                    if not var_converged:
                        snrs[itrn,itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp,listT_temp,waveT_temp,nt_min,nt_max)
                        snrs_tot[itrn,itrb] = np.linalg.norm(snrs[itrn,itrb])
                        thresh_var = snrs_tot[itrn,itrb]>=snr_autosuppress
                    else:
                        snrs[itrn,itrb] = snrs[itrn-1,itrb]
                        snrs_tot[itrn,itrb] = snrs_tot[itrn-1,itrb]
                        thresh_var = False
                    if np.isnan(snrs_tot[itrn,itrb]) or np.isnan(snrs_tot_base[itrn,itrb]):
                        var_suppress_loc = False
                        const_suppress_loc = False
                        raise ValueError('nan detected in snr at '+str(itrn)+','+str(itrb))
                    elif thresh_var and thresh_base:
                        #satifisfied conditions to be eliminated in both directions so just keep it
                        var_suppress_loc = False
                        const_suppress_loc = False
                    elif thresh_var:
                        if snrs_tot[itrn,itrb]>snrs_tot_base[itrn,itrb]:
                            #handle case where snr ordering is wrong to prevent oscillation
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

                    var_suppress[itrn,itrb] = var_suppress_loc
                    const_suppress2[itrn,itrb] = const_suppress_loc

                    if not var_suppress_loc and not const_suppress_loc:
                        for itrc in range(0,2):
                            galactic_bg[listT_temp[itrc,:NUTs_temp[itrc]],itrc] += waveT_temp[itrc,:NUTs_temp[itrc]]
                    elif var_suppress_loc and not const_suppress_loc:
                        for itrc in range(0,2):
                            galactic_bg_suppress[listT_temp[itrc,:NUTs_temp[itrc]],itrc] += waveT_temp[itrc,:NUTs_temp[itrc]]
                    elif not var_suppress_loc and const_suppress_loc:
                        if itrn==1:
                            const_suppress2[itrn,itrb] = False
                            const_suppress[itrb] = True
                            for itrc in range(0,2):
                                galactic_bg_const_base[listT_temp[itrc,:NUTs_temp[itrc]],itrc] += waveT_temp[itrc,:NUTs_temp[itrc]]
                        else:
                            for itrc in range(0,2):
                                galactic_bg_const[listT_temp[itrc,:NUTs_temp[itrc]],itrc] += waveT_temp[itrc,:NUTs_temp[itrc]]
                    else:
                        raise ValueError('impossible state')




            #carry forward any other snr values we still know
            if const_converged:
                snrs_tot_base[itrn,suppressed] = snrs_tot_base[itrn-1,suppressed]
                snrs_base[itrn,suppressed] = snrs_base[itrn-1,suppressed]
            if var_converged:
                snrs_tot[itrn,suppressed] = snrs_tot[itrn-1,suppressed]
                snrs[itrn,suppressed] = snrs[itrn-1,suppressed]

            if itrn==1:
                assert np.all(galactic_bg_const==0.)
                galactic_full_signal[:] = galactic_bg_const_base+galactic_bg_const+galactic_bg+galactic_bg_suppress
                snr_min = 0.999*snr_thresh#/np.sqrt(1+(snr_thresh/(2*smooth_lengthf+1))**2) #choose value to ensure almost nothing gets suppressed as constant because of its own power alone
            else:
                #check all contributions to the total signal are tracked accurately
                assert np.allclose(galactic_full_signal,galactic_bg_const_base+galactic_bg_const+galactic_bg+galactic_bg_suppress,atol=1.e-300,rtol=1.e-6)


            t1n = perf_counter()

            if not var_converged:
                galactic_bg_res = galactic_bg+galactic_bg_const+galactic_bg_const_base
                n_var_suppressed_new = var_suppress[itrn].sum()

                if itrn>1 and (force_converge or (np.all(var_suppress[itrn]==var_suppress[itrn-1]) or np.all(var_suppress[itrn]==var_suppress[itrn-2]) or np.all(var_suppress[itrn]==var_suppress[itrn-3]))):
                    assert n_var_suppressed == n_var_suppressed_new or force_converge or np.all(var_suppress[itrn]==var_suppress[itrn-2]) or np.all(var_suppress[itrn]==var_suppress[itrn-3])
                    if switch_next:
                        print('subtraction converged at '+str(itrn))
                        var_converged = True
                        switch_next = False
                        const_converged = True
                    else:
                        if (np.all(var_suppress[itrn]==var_suppress[itrn-2]) or np.all(var_suppress[itrn]==var_suppress[itrn-3])) and not np.all(var_suppress[itrn]==var_suppress[itrn-1]):
                            print('cycling detected at '+str(itrn)+', doing final check iteration aborting')
                            force_converge = True
                        print('subtraction predicted initial converged at '+str(itrn)+' next iteration will be check iteration')
                        switch_next = True
                        var_converged = False
                else:
                    switch_next = False

                    if itrn<n_cyclo_switch:
                        #TODO check this is being used appropriately
                        print('here')
                        SAET_tot_cur, _, _ = get_SAET_cyclostationary_mean(galactic_bg_res,SAET_m,smooth_lengthf_cur,filter_periods=False,period_list=period_list1)
                    else:
                        SAET_tot_cur, _, _ = get_SAET_cyclostationary_mean(galactic_bg_res,SAET_m,smooth_lengthf_cur,filter_periods=not const_only,period_list=period_list1)

                    noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_cur,prune=True)
                    n_var_suppressed = n_var_suppressed_new
            else:
                switch_next = False

            if itr_save<idx_SAE_save.size and itrn == idx_SAE_save[itr_save]:
                SAE_tots[itr_save] = SAET_tot_cur[:,:,:2]
                itr_save += 1

            if not const_converged or switch_next:
                if itrn<n_const_force:
                    SAET_tot_base, _, _ = get_SAET_cyclostationary_mean(galactic_bg_const+galactic_bg_const_base,SAET_m,smooth_lengthf_targ,filter_periods=not const_only,period_list=period_list1)
                else:
                    SAET_tot_base, _, _ = get_SAET_cyclostationary_mean(galactic_bg_const+galactic_bg_const_base,SAET_m,smooth_lengthf_targ,filter_periods=not const_only,period_list=period_list1)
                    const_converged = True
                    #need to disable adaption of constant here because after this point the convergence isn't guaranteed to be monotonic
                    print('disabled constant adaptation at '+str(itrn))

                #make sure this will always predict >=snrs to the actual spectrum in use
                SAET_tot_base = np.min([SAET_tot_base,SAET_tot_cur],axis=0)
                noise_AET_dense_base = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot_base,prune=True)

                n_const_suppressed_new = const_suppress2[itrn].sum()
                if switch_next and const_converged:#and check_count <= 1:
                    print('overriding constant convergence to check background model')
                    const_converged = False
                    switchf_next = False
                elif n_const_suppressed_new - n_const_suppressed<0:
                    if var_converged:
                        switch_next = True
                        var_converged = False
                    switchf_next = False
                    const_converged = False
                    print('addition removed values at '+str(itrn)+', repeating check iteration')

                elif itrn!=1 and np.abs(n_const_suppressed_new - n_const_suppressed) < const_converge_change_thresh:
                    if switchf_next:
                        const_converged = True
                        switchf_next = False
                        print('addition converged at '+str(itrn))
                    else:
                        print('near convergence in constant adaption at '+str(itrn),' doing check iteration')
                        switchf_next = False
                        const_converged = False
                else:
                    if var_converged:
                        print('addition convergence continuing beyond subtraction, try check iteration')
                        switchf_next = False
                        const_converged = False
                    else:
                        switchf_next = False

                n_const_suppressed = n_const_suppressed_new
            else:
                switchf_next = False

            if switchf_next:
                assert not const_converged
            if switch_next:
                assert not var_converged


            parseval_tot[itrn] = np.sum((galactic_bg_const_base+galactic_bg_const+galactic_bg+galactic_bg_suppress).reshape((wc.Nt,wc.Nf,wc.NC))[:,1:,0:2]**2/SAET_m[1:,0:2])
            parseval_bg[itrn] = np.sum((galactic_bg_const_base+galactic_bg_const+galactic_bg).reshape((wc.Nt,wc.Nf,wc.NC))[:,1:,0:2]**2/SAET_m[1:,0:2])
            parseval_const[itrn] = np.sum((galactic_bg_const_base+galactic_bg_const).reshape((wc.Nt,wc.Nf,wc.NC))[:,1:,0:2]**2/SAET_m[1:,0:2])
            parseval_sup[itrn] = np.sum((galactic_bg_suppress).reshape((wc.Nt,wc.Nf,wc.NC))[:,1:,0:2]**2/SAET_m[1:,0:2])

            t2n = perf_counter()
            print('made bg %3d in time %7.3fs fit time %7.3fs'%(itrn,t1n-t0n,t2n-t1n))


            #give absorbing constants a relative advantage on early iterations
            #because for the first iteration we included no galactic background
            snr_autosuppress = snr_autosuppresses[itrn]
            smooth_lengthf_cur = smooth_lengthfs[itrn]

            if var_converged and const_converged:
                print('result fully converged at '+str(itrn)+', no further iterations needed')
                n_full_converged = itrn
                break

        SAE_fin[:] = SAET_tot_cur[:,:,:2]


        do_hf_out = True
        if do_hf_out:
            filename_out = gfi.get_processed_gb_filename(const_only,snr_thresh=snr_thresh,nt_min=nt_min,nt_max=nt_max,Nf=wc.Nf,Nt=wc.Nt,smooth_lengtht=smooth_lengtht,smooth_lengthf=smooth_lengthf,dt=wc.dt)
            hf_out = h5py.File(filename_out,'w')
            hf_out.create_group('SAET')
            hf_out['SAET'].create_dataset('galactic_bg_const',data=galactic_bg_const+galactic_bg_const_base,compression='gzip')
            hf_out['SAET'].create_dataset('galactic_bg_suppress',data=galactic_bg_suppress,compression='gzip')
            hf_out['SAET'].create_dataset('galactic_bg',data=galactic_bg,compression='gzip')
            hf_out['SAET'].create_dataset('smooth_lengthf',data=smooth_lengthf)
            hf_out['SAET'].create_dataset('smooth_lengtht',data=smooth_lengtht)
            hf_out['SAET'].create_dataset('snr_thresh',data=snr_thresh)
            hf_out['SAET'].create_dataset('snr_min',data=snr_min)
            hf_out['SAET'].create_dataset('period_list',data=period_list1)

            hf_out['SAET'].create_dataset('Nt',data=wc.Nt)
            hf_out['SAET'].create_dataset('Nf',data=wc.Nf)
            hf_out['SAET'].create_dataset('dt',data=wc.dt)
            hf_out['SAET'].create_dataset('n_iterations',data=n_iterations)
            hf_out['SAET'].create_dataset('n_bin_use',data=n_bin_use)
            hf_out['SAET'].create_dataset('SAET_m',data=SAET_m)
            hf_out['SAET'].create_dataset('snrs_tot',data=snrs_tot[n_full_converged],compression='gzip')
            hf_out['SAET'].create_dataset('argbinmap',data=argbinmap,compression='gzip')

            hf_out['SAET'].create_dataset('const_suppress',data=const_suppress,compression='gzip')
            hf_out['SAET'].create_dataset('const_suppress2',data=const_suppress2[n_full_converged],compression='gzip')

            hf_out['SAET'].create_dataset('var_suppress',data=var_suppress[n_full_converged],compression='gzip')
            hf_out['SAET'].create_dataset('SAEf',data=SAE_fin,compression='gzip')

            hf_out['SAET'].create_dataset('source_gb_file',data=gfi.full_galactic_params_filename)
            hf_out['SAET'].create_dataset('master_gb_file',data=gfi.master_gb_filename)
            hf_out['SAET'].create_dataset('init_gb_file',data=filename_gb_init)
            hf_out['SAET'].create_dataset('common_gb_noise_file',data=filename_gb_common)

            hf_out.close()

        do_hf_SAET = False
        if do_hf_SAET:
            filename_out = "Galaxy/gb75_SAET_evolve_smoothf="+str(smooth_lengthf)+'smootht='+str(smooth_lengtht)+'snr'+str(snr_thresh)+"_Nf="+str(wc.Nf)+"_Nt="+str(wc.Nt)+"_dt="+str(wc.dt)+"const="+str(const_only)+"nt_min="+str(nt_min)+"nt_max="+str(nt_max)+".hdf5"
            hf_out = h5py.File(filename_out,'w')
            hf_out.create_group('SAET')
            hf_out['SAET'].create_dataset('SAE_tots',data=SAE_tots,compression='gzip')
            hf_out.close()

        do_hf_realization = False
        if do_hf_realization:
            filename_out = "Galaxy/gb75_realization_evolve_smoothf="+str(smooth_lengthf)+'smootht='+str(smooth_lengtht)+'snr'+str(snr_thresh)+"_Nf="+str(wc.Nf)+"_Nt="+str(wc.Nt)+"_dt="+str(wc.dt)+"const="+str(const_only)+"nt_min="+str(nt_min)+"nt_max="+str(nt_max)+".hdf5"
            hf_out = h5py.File(filename_out,'w')
            hf_out.create_group('SAET')
            hf_out['SAET'].create_dataset('data_realization',data=(galactic_bg_res.reshape(wc.Nt,wc.Nf,wc.NC)+noise_realization),compression='gzip')
            hf_out.close()

        tf = perf_counter()
        print('loop time = %.3es'%(tf-ti))


    plot_noise_spectrum_evolve = True
    if plot_noise_spectrum_evolve:
        import matplotlib.pyplot as plt

spec_shift = np.mean((galactic_bg_res.reshape(wc.Nt,wc.Nf,wc.NC)+noise_realization)[:,:,0:2]**2,axis=0)
SAET_m_shift = SAET_m
print(spec_shift[766]/SAET_m_shift[766,0])
plt.loglog(np.arange(1,wc.Nf)*wc.DF,SAET_m_shift[1:,0])
plt.loglog(np.arange(1,wc.Nf)*wc.DF,np.mean(SAE_tots[0:,:,1:,0],axis=1).T)
plt.ylim(2.e-44,8.e-41)
plt.xlim(1.e-4,1.e-2)
plt.xlabel('f (Hz)')
plt.show()

plt.plot(parseval_const[1:itrn+1]/parseval_tot[1:itrn+1])
plt.plot(parseval_bg[1:itrn+1]/parseval_tot[1:itrn+1])
plt.plot(parseval_sup[1:itrn+1]/parseval_tot[1:itrn+1])
plt.show()

res_mask = (SAET_tot_cur[:,:,0]-SAET_m[:,0]).mean(axis=0)>0.1*SAET_m[:,0]
unit_normal_battery((galactic_bg_res.reshape(wc.Nt,wc.Nf,wc.NC)[nt_min:nt_max,res_mask,0:2]/np.sqrt(SAET_tot_cur[nt_min:nt_max,res_mask,0:2]-SAET_m[res_mask,0:2])).flatten(),A2_cut=10.,sig_thresh=10.)

plt.imshow(np.rot90(galactic_bg_res.reshape(wc.Nt,wc.Nf,wc.NC)[:,res_mask,0]**2/(SAET_tot_cur[:,res_mask,0]-SAET_m[res_mask,0])),aspect='auto')
plt.show()

fig = plt.figure(figsize=(5.4,3.5))
ax = fig.subplots(1)
fig.subplots_adjust(wspace=0.,hspace=0.,left=0.13,top=0.99,right=0.99,bottom=0.12)
ax.loglog(np.arange(1,wc.Nf)*wc.DF,(galactic_full_signal.reshape((wc.Nt,wc.Nf,wc.NC))[:,1:,0:2]**2).mean(axis=0).mean(axis=1)+SAET_m[1:,0],'k',alpha=0.3,zorder=-90)
ax.loglog(np.arange(1,wc.Nf)*wc.DF,np.mean(SAE_tots[[1,2,3,4],:,1:,0],axis=1).T,'--',alpha=0.7)
ax.loglog(np.arange(1,wc.Nf)*wc.DF,np.mean(SAET_tot_cur[:,1:,0:2],axis=0).mean(axis=1).T)
ax.loglog(np.arange(1,wc.Nf)*wc.DF,SAET_m_shift[1:,0],'k--',zorder=-100)
ax.tick_params(axis='both',direction='in',which='both',top=True,right=True)
plt.legend(['initial','1','2','3','4','5','6','final'])
plt.ylim([2.e-44,4.e-43])
plt.xlim([3.e-4,6.e-3])
plt.xlabel('f (Hz)')
plt.ylabel(r"$\langle S^{AE}_{m} \rangle$")
plt.show()
