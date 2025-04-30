"""second step in processing the waveform from a galaxy of binaries"""
from time import perf_counter

import numpy as np
import h5py

from wavelet_detector_waveforms import BinaryWaveletAmpFreqDT
from ra_waveform_time import BinaryTimeWaveformAmpFreqD
from wdm_const import wdm_const as wc
from wdm_const import lisa_const as lc
import global_const as gc
from instrument_noise import DiagonalStationaryDenseInstrumentNoiseModel, instrument_noise_AET_wdm_m, DiagonalNonstationaryDenseInstrumentNoiseModel
import global_file_index as gfi

if __name__=='__main__':
    params_gb, n_dgb, n_igb, n_vgb, n_tot = gfi.get_full_galactic_params()

    params0 = params_gb[0].copy()

    snr_thresh = 7
    snr_min = 7

    galactic_bg_const_in, noise_realization_common, snrs_tot_in = gfi.load_master_galactic_file(snr_thresh, wc)
    const_suppress_in = snrs_tot_in<snr_min

    waveT_ini = BinaryWaveletAmpFreqDT(params0.copy(), wc, lc)
    listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)
    noise_AET_dense_pure = DiagonalStationaryDenseInstrumentNoiseModel(SAET_m, wc, prune=False)

    noise_realization = noise_realization_common[:wc.Nt, :, :].copy()

    galactic_bg = np.zeros((wc.Nt*wc.Nf, wc.NC))
    n_bin_use = n_tot

    n_iterations = 2
    SAET_tot = np.zeros((n_iterations+1, wc.Nt, wc.Nf, wc.NC))
    SAET_tot[0] = noise_AET_dense_pure.SAET.copy()

    snr_autosuppress = 500
    snrs = np.zeros((n_iterations, n_bin_use, wc.NC))
    snrs_tot = np.zeros((n_iterations, n_bin_use))
    snrs_tot[:] = snrs_tot_in

    const_suppress = const_suppress_in.copy()
    var_suppress = np.zeros((n_iterations, n_bin_use), dtype=np.bool_)

    galactic_bg_const = galactic_bg_const_in.reshape(wc.Nt, wc.Nf, wc.NC)[:wc.Nt].reshape(wc.Nt*wc.Nf, wc.NC)

    ti = perf_counter()
    for itrn in range(0, n_iterations):
        galactic_bg = np.zeros((wc.Nt*wc.Nf, wc.NC))
        noise_AET_dense = DiagonalNonstationaryDenseInstrumentNoiseModel(SAET_tot[itrn], wc, prune=False)
        t0n = perf_counter()

        for itrb in range(0, n_bin_use):
            #TODO can block for var suppress if timing of later iterations becomes an issue
            if itrb%10000==0 and itrn==0:
                tin = perf_counter()
                print("itrb="+str(itrb)+" at t="+str(tin-t0n)+" s in loop")
            if not const_suppress[itrb]:
                waveT_ini.update_params(params_gb[itrb].copy())
                listT_temp, waveT_temp, NUTs_temp = waveT_ini.get_unsorted_coeffs()
                snrs[itrn, itrb] = noise_AET_dense.get_sparse_snrs(NUTs_temp, listT_temp, waveT_temp)
                snrs_tot[itrn, itrb] = np.linalg.norm(snrs[itrn, itrb])
                if itrn == 0 and snrs_tot[0, itrb]<snr_min:
                    const_suppress[itrb] = True
                    for itrc in range(0, wc.NC):
                        galactic_bg_const[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
                elif snrs_tot[itrn, itrb]<snr_autosuppress:
                    for itrc in range(0, wc.NC):
                        galactic_bg[listT_temp[itrc, :NUTs_temp[itrc]], itrc] += waveT_temp[itrc, :NUTs_temp[itrc]]
                else:
                    var_suppress[itrn, itrb] = True
        t1n = perf_counter()
        print('made bg '+str(itrn)+' in time '+str(t1n-t0n)+'s')

        galactic_bg_res = galactic_bg+galactic_bg_const

        galactic_bg_full = galactic_bg_res.reshape(wc.Nt, wc.Nf, wc.NC).copy()#+noise_realization

        signal_full = galactic_bg_full+noise_realization

        SAET_galactic_bg_smoothf_white = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_galactic_bg_smoothft_white = np.zeros((wc.Nt, wc.Nf, wc.NC))
        SAET_galactic_bg_smooth = np.zeros((wc.Nt, wc.Nf, wc.NC))

        for itrc in range(0, wc.NC):
            smooth_lengthf = 8
            SAET_galactic_bg_white = signal_full[:, :, itrc]**2/SAET_m[:, itrc]
            for itrf in range(0, wc.Nf):
                rreach = smooth_lengthf//2-max(itrf-wc.Nf+smooth_lengthf//2+1, 0)
                lreach = smooth_lengthf//2-max(smooth_lengthf//2-itrf, 0)
                SAET_galactic_bg_smoothf_white[:, itrf, itrc] = np.mean(SAET_galactic_bg_white[:, itrf-lreach:itrf+rreach+1], axis=1)
            smooth_lengtht = 84*2
            for itrt in range(0, wc.Nt):
                rreach = smooth_lengtht//2-max(itrt-wc.Nt+smooth_lengtht//2+1, 0)
                lreach = smooth_lengtht//2-max(smooth_lengtht//2-itrt, 0)
                SAET_galactic_bg_smoothft_white[itrt, :, itrc] = np.mean(SAET_galactic_bg_smoothf_white[itrt-lreach:itrt+rreach+1, :, itrc], axis=0)
            SAET_galactic_bg_smooth[:, :, itrc] = SAET_galactic_bg_smoothft_white[:, :, itrc]*SAET_m[:, itrc]

        snr_autosuppress = snr_thresh
        SAET_tot[itrn+1] = SAET_galactic_bg_smooth

    do_hf_write = True
    if do_hf_write:
        filename_out = gfi.get_init_filename(snr_thresh, wc)
        hf_out = h5py.File(filename_out, 'w')
        hf_out.create_group('SAET')
        hf_out['SAET'].create_dataset('galactic_bg_const', data=galactic_bg_const, compression='gzip')
        hf_out['SAET'].create_dataset('noise_realization', data=noise_realization, compression='gzip')
        hf_out['SAET'].create_dataset('smooth_lengthf', data=smooth_lengthf)
        hf_out['SAET'].create_dataset('smooth_lengtht', data=smooth_lengtht)
        hf_out['SAET'].create_dataset('snr_thresh', data=snr_thresh)
        hf_out['SAET'].create_dataset('snr_min', data=snr_min)
        hf_out['SAET'].create_dataset('Nt', data=wc.Nt)
        hf_out['SAET'].create_dataset('Nf', data=wc.Nf)
        hf_out['SAET'].create_dataset('dt', data=wc.dt)
        hf_out['SAET'].create_dataset('n_iterations', data=n_iterations)
        hf_out['SAET'].create_dataset('n_bin_use', data=n_bin_use)
        hf_out['SAET'].create_dataset('SAET_m', data=SAET_m)
        hf_out['SAET'].create_dataset('snrs_tot', data=snrs_tot[0], compression='gzip')
        hf_out['SAET'].create_dataset('source_gb_file', data=gfi.full_galactic_params_filename)
        hf_out['SAET'].create_dataset('master_gb_file', data=gfi.get_master_filename(snr_thresh, wc))

        hf_out.close()

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
