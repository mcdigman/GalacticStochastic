"""helper functions for Chirp_WDM"""
from numba import njit
import numpy as np

@njit(fastmath=True)
def wavemaket_multi_inplace(waveTs,Tlists,Phases,fas,fdas,Amps,NC,nt_range,wc, evcTs, evsTs, NfsamT, force_nulls=False):
    """compute the actual wavelets using taylor time method"""

    #indicates this pixel not used
    NUs = np.zeros(NC,dtype=np.int64)
    for itrc in range(0,NC):
        mm = 0
        for j in range(0,nt_range):
            j_ind = j

            y0 = fdas[itrc,j]/wc.dfd
            ny = np.int64(np.floor(y0))
            n_ind = ny+wc.Nfd_negative

            if 0<=n_ind<wc.Nfd-2: #and 0.<=fdas[itrc,j]<fdmx:
                #print(y0,n_ind,j)
                c = Amps[itrc,j]*np.cos(Phases[itrc,j])
                s = Amps[itrc,j]*np.sin(Phases[itrc,j])


                dy = y0-ny
                fa = fas[itrc,j]
                za = fa/wc.df

                Nfsam1_loc = NfsamT[n_ind]
                Nfsam2_loc = NfsamT[n_ind+1]
                HBW  = (min(Nfsam1_loc,Nfsam2_loc)-1)*wc.df/2

                #lowest frequency layer
                kmin = max(0,np.int64(np.ceil((fa-HBW)/wc.DF)))

                #highest frequency layer
                kmax = min(wc.Nf-1,np.int64(np.floor((fa+HBW)/wc.DF)))
                for k in range(kmin,kmax+1):
                    Tlists[itrc,mm] = j_ind*wc.Nf+k

                    zmid = (wc.DF/wc.df)*k

                    kk = np.floor(za-zmid-0.5)
                    zsam = zmid+kk+0.5
                    kk = np.int64(kk)
                    dx = za-zsam  # used for linear interpolation



                    #interpolate over frequency
                    jj1 = kk+Nfsam1_loc//2
                    jj2 = kk+Nfsam2_loc//2

                    assert evcTs[n_ind,jj1]!=0.
                    assert evcTs[n_ind,jj1+1]!=0.
                    assert evcTs[n_ind+1,jj2]!=0.
                    assert evcTs[n_ind+1,jj2+1]!=0.

                    y  = (1.-dx)*evcTs[n_ind,jj1]+dx*evcTs[n_ind,jj1+1]
                    yy = (1.-dx)*evcTs[n_ind+1,jj2]+dx*evcTs[n_ind+1,jj2+1]

                    z  = (1.-dx)*evsTs[n_ind,jj1]+dx*evsTs[n_ind,jj1+1]
                    zz = (1.-dx)*evsTs[n_ind+1,jj2]+dx*evsTs[n_ind+1,jj2+1]

                    #interpolate over fdot
                    y = (1.-dy)*y+dy*yy
                    z = (1.-dy)*z+dy*zz

                    if (j_ind+k)%2:
                        waveTs[itrc,mm] = -(c*z+s*y)
                    else:
                        waveTs[itrc,mm] =  (c*y-s*z)

                    mm += 1
                    #end loop over frequency layers
            elif force_nulls:
                #we know what the indices would be for values not precomputed in the table
                #so force values outside the range of the table to 0 instead of dropping, in order to get likelihoods right
                #which is particularly important around total nulls which can be quite constraining on the parameters but have large spikes in frequency derivative
                #note that if this happens only very rarely, we could also just actually calculate the non-precomputed coefficient
                fa = fas[itrc,j]

                Nfsam1_loc = np.int64((wc.BW+wc.dfd*wc.Tw*ny)/wc.df)
                if Nfsam1_loc%2==1:
                    Nfsam1_loc += 1

                HBW  = (Nfsam1_loc-1)*wc.df/2
                #lowest frequency layer
                kmin = max(0,np.int64(np.ceil((fa-HBW)/wc.DF)))

                #highest frequency layer
                kmax = min(wc.Nf-1,np.int64(np.floor((fa+HBW)/wc.DF)))

                for k in range(kmin,kmax+1):
                    Tlists[itrc,mm] = j_ind*wc.Nf+k
                    waveTs[itrc,mm] = 0.
                    mm += 1

        NUs[itrc] = mm

    return NUs
