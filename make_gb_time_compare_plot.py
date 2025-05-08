"""scratch to test processing of galactic background"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


from instrument_noise import instrument_noise_AET_wdm_m

from galactic_fit_helpers import get_SAET_cyclostationary_mean

import config_helper

import global_file_index as gfi

mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.5
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.width'] = 1.5


if __name__=='__main__':
    config, wc, lc = config_helper.get_config_objects('default_parameters.ini')
    galaxy_dir = config['files']['galaxy_dir']

    snr_thresh = 7.

    const_only = False

    if const_only:
        period_list = np.array([])
    else:
        period_list = np.array([1,2,3,4,5])

    SAET_m = instrument_noise_AET_wdm_m(lc, wc)

    Nts =     np.array([4096, 4096, 4096])
    nt_mins = np.array([256*4,256*6,256*7])
    nt_maxs = np.array([2048,1024,512])+nt_mins

    nk = nt_maxs.size
    assert nt_mins.size==nt_maxs.size
    assert nt_maxs.size==Nts.size

    SAETs_var = []

    for itrk in range(0,nk):
        print(Nts[itrk])
        _, galactic_bg_total = gfi.load_processed_gb_file(galaxy_dir, snr_thresh, wc, lc, nt_mins[itrk], nt_maxs[itrk], const_only)
        SAET_model_var, _, _, _, _ = get_SAET_cyclostationary_mean(galactic_bg_total, SAET_m,wc, 0.,filter_periods=True,period_list=period_list, Nt_loc=wc.Nt)
        SAETs_var.append(SAET_model_var)

    do_SAE_plot = True
    if do_SAE_plot:
        for itrk in range(0,nk):
            SAET_loc = SAETs_var[itrk]
            #TODO investigate last bin behavior
            plt.loglog(np.arange(1,wc.Nf-1)*wc.DF,np.mean(SAET_loc[:,1:wc.Nf-1,0],axis=0).T)

        plt.xlabel('f (Hz)')
        plt.ylabel(r"$S^{AE}_{nm}$ (1/Hz)")
        #plt.xlim(1.e-4,1.e-2)
        plt.ylim(5.e-45,1.e-40)
        plt.show()
