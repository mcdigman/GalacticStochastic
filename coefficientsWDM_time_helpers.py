"""C 2023 Matthew C. Digman
test the wdm time functions"""
from time import time

import h5py
import numpy as np
import scipy as sp
from WDMWaveletTransforms.transform_freq_funcs import phitilde_vec

import global_const as gc
from coefficientsWDM_time_funcs import get_ev_t_full, wavelet


def get_evTs(wc, check_cache=True, hf_out=False):
    """helper to get the ev matrices"""
    t0 = time()

    print("Filter length (seconds) %e" % wc.Tw)
    print("dt="+str(wc.dt)+"s Tobs="+str(wc.Tobs/gc.SECSYEAR))

    print("full filter bandwidth %e  samples %d" % ((wc.A+wc.B)/np.pi, (wc.A+wc.B)/np.pi*wc.Tw))
    cache_good = False

    if check_cache:
        coeffTs_in = 'coeffs/WDMcoeffs_Nsf='+str(wc.Nsf)+'_mult='+str(wc.mult)+'_Nfd='+str(wc.Nfd)+'_Nf='+str(wc.Nf)\
            + '_Nt='+str(wc.Nt)+'_dt='+str(wc.dt)+'_dfdot='+str(wc.dfdot)+'_Nfd_neg'+str(wc.Nfd_negative)+'_fast.h5'

        fds = wc.dfd*np.arange(-wc.Nfd_negative, wc.Nfd-wc.Nfd_negative)

        # number of samples for each frequency derivative layer (grow with increasing BW)
        Nfsam = ((wc.BW+np.abs(fds)*wc.Tw)/wc.df).astype(np.int64)
        Nfsam[Nfsam % 2 != 0] += 1  # makes sure it is an even number

        max_shape = np.max(Nfsam)
        evcs = np.zeros((wc.Nfd, max_shape))
        evss = np.zeros((wc.Nfd, max_shape))

        try:
            hf_in = h5py.File(coeffTs_in, 'r')
            for i in range(0, wc.Nfd):
                evcs[i, :Nfsam[i]] = np.asarray(hf_in['evcs'][str(i)])
                evss[i, :Nfsam[i]] = np.asarray(hf_in['evss'][str(i)])
            hf_in.close()
            cache_good = True
        except OSError:
            print(coeffTs_in)
            print('cache checked and missed')

    if not cache_good:
        phi = np.zeros(wc.K)
        DX = np.zeros(wc.K, dtype=np.complex128)
        DX[0] = wc.insDOM

        DX[1:np.int64(wc.K/2)+1] = np.sqrt(wc.dt)*phitilde_vec(wc.dom*wc.dt*np.arange(1, np.int64(wc.K/2)+1), wc.Nf, wc.nx)
        DX[np.int64(wc.K/2)+1:] = np.sqrt(wc.dt)*phitilde_vec(-wc.dom*wc.dt*np.arange(np.int64(wc.K/2)-1, 0, -1), wc.Nf, wc.nx)

        DX = sp.fft.fft(DX, wc.K, overwrite_x=True)

        for i in range(0, np.int64(wc.K/2)):
            phi[i] = np.real(DX[np.int64(wc.K/2)+i])
            phi[np.int64(wc.K/2)+i] = np.real(DX[i])

        nrm = np.linalg.norm(phi)
        print("norm="+str(nrm))

        # it turns out that all the wavelet layers are the same modulo a
        # shift in the reference frequency. Just have to do a single layer
        # we pick one far from the boundaries to avoid edge effects

        k = wc.Nf/16

        wave = wavelet(k, wc.K, nrm, wc.dom, wc.DOM, wc.Nf, wc.dt, wc.nx)

        fd = wc.DF/wc.Tw*wc.dfdot*np.arange(-wc.Nfd_negative, wc.Nfd-wc.Nfd_negative)  # set f-dot increments

        print("%e %.14e %.14e %e %e" % (wc.DT, wc.DF, wc.DOM/(2*np.pi), fd[1], fd[wc.Nfd-1]))

        Nfsam = ((wc.BW+np.abs(fd)*wc.Tw)/wc.df).astype(np.int64)
        odd_mask = np.mod(Nfsam, 2) != 0
        Nfsam[odd_mask] += 1

        # The odd wavelets coefficienst can be obtained from the even.
        # odd cosine = -even sine, odd sine = even cosine

        # each wavelet covers a frequency band of width DW
        # except for the first and last wavelets
        # there is some overlap. The wavelet pixels are of width
        # DOM/PI, except for the first and last which have width
        # half that

        t1 = time()
        print("loop start time ", t1-t0, "s")
        evcs, evss = get_ev_t_full(wave, wc)
        tf = time()
        print("got full evcs in %f s" % (tf-t1))
        t1 = time()

    if hf_out:
        hf = h5py.File('coeffs/WDMcoeffs_Nsf='+str(wc.Nsf)+'_mult='+str(wc.mult)+'_Nfd='+str(wc.Nfd)+'_Nf='+str(wc.Nf) +
                       '_Nt='+str(wc.Nt)+'_dt='+str(wc.dt)+'_dfdot='+str(wc.dfdot)+'_Nfd_neg'+str(wc.Nfd_negative)+'_fast.h5', 'w')
        hf.create_group('inds')
        hf.create_group('evcs')
        hf.create_group('evss')
        for jj in range(0, wc.Nfd):
            hf['inds'].create_dataset(str(jj), data=np.arange(0, Nfsam[jj]))
            hf['evcs'].create_dataset(str(jj), data=evcs[jj, :Nfsam[jj]])
            hf['evss'].create_dataset(str(jj), data=evss[jj, :Nfsam[jj]])
        hf.close()
        t3 = time()
        print("output time", t3-tf, "s")

    return Nfsam, evcs, evss
