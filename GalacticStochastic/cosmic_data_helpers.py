"""A series of small helper functions to load cosmic data from HDF5 files."""

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import Galactocentric, ICRS, GeocentricTrueEcliptic

def get_amplitude(dat):
    """Compute the gravitational wave amplitude for each source in the dataset.
    
    Parameters
    ----------
    dat : pandas.DataFrame
        DataFrame containing source parameters including mass_1, mass_2, f_gw, and dist_sun
    
    Returns
    -------
    amplitude : numpy.ndarray
        Gravitational wave amplitude for each source with appropriate units
    """
    mc = (dat.mass_1.values*dat.mass_2.values)**(3/5) / (dat.mass_1.values + dat.mass_2.values)**(1/5) * u.Msun
    term1 = 64/5 * (const.G * mc)**(10/3)
    term2 = (np.pi*dat.f_gw.values*u.s**(-1))**(4/3)
    denom1 = const.c**8*(dat.dist_sun.values*u.kpc)**2
    amplitude = np.sqrt(term1.to(u.m**10/u.s**(20/3)) * term2 / denom1.to(u.m**10/u.s**8)).value
    return amplitude

def get_Gx_positions(dat):
    """Convert Galactocentric Cartesian coordinates to ecliptic longitude and latitude.
    
    Parameters
    ----------
    dat : pandas.DataFrame
        DataFrame containing xGx, yGx, zGx columns for Galactocentric Cartesian coordinates
        
    Returns
    -------
    lon : numpy.ndarray
        Ecliptic longitude in radians
    lat : numpy.ndarray
        Ecliptic latitude in radians
    """
    galcen = Galactocentric(x=dat.xGx.values*u.kpc, y=dat.yGx.values*u.kpc, z=dat.zGx.values*u.kpc)
    icrs = galcen.transform_to(ICRS())
    ecl = icrs.transform_to(GeocentricTrueEcliptic())
    return ecl.lon.to(u.rad).value, ecl.lat.to(u.rad).value

def get_chirp(dat):
    """Compute the chirp (df/dt) for each source in the dataset
    
    Parameters
    ----------
    dat : pandas.DataFrame
        DataFrame containing source parameters including mass_1, mass_2, and f_gw
    
    Returns
    -------
    chirp : numpy.ndarray
        Chirp (df/dt) for each source with appropriate units
    """
    mc = (dat.mass_1.values*dat.mass_2.values)**(3/5) / (dat.mass_1.values + dat.mass_2.values)**(1/5) * u.Msun
    fgw = dat.f_gw.values*u.s**(-1)
    term1 = (const.G * mc)**(5/3) / (const.c)**5
    term2 = (np.pi * fgw)**(11/3)
    chirp = 96/(5*np.pi) * term1 * term2
    return chirp.to(u.s**(-2)).value

def get_inc_phase_pol(n_dat):
    """Generate random inclination, phase, and polarization angles for each source.
    
    Parameters
    ----------
    n_dat : int
        Number of sources in the dataset
        
    
    Returns
    -------
    inc : numpy.ndarray
        Inclination angles in radians
    phase : numpy.ndarray
        Phase angles in radians
    pol : numpy.ndarray
        Polarization angles in radians
    """
    inc = np.arccos(np.random.uniform(0, 1, n_dat))
    phase = np.random.uniform(0, np.pi, n_dat)
    pol = np.random.uniform(0, np.pi, n_dat)
    return inc, phase, pol

def create_dat_in(dat):
    """Create input data array for waveform generation from source parameters.
    
    Parameters
    ----------
    dat : pandas.DataFrame
        DataFrame containing source parameters including mass_1, mass_2, f_gw, dist_sun, xGx, yGx, zGx 
    
    Returns
    -------
    dat_in : numpy.ndarray
        Array of shape (N, 8) where N is the number of sources, containing the following columns:
        [amplitude, ecliptic longitude, ecliptic latitude, f_gw, chirp, inclination, phase, polarization]
    """
    h = get_amplitude(dat)
    lon, lat = get_Gx_positions(dat)
    chirp = get_chirp(dat)
    inc, phase, pol = get_inc_phase_pol(len(dat))
    dat_in = np.vstack([h, lat, lon, dat.f_gw.values, chirp, inc, phase, pol]).T
    print(np.min(lat), np.max(lat))
    return dat_in