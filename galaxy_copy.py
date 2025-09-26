import h5py

import GalacticStochastic.global_file_index as gfi

galaxy_file = 'LDC2_sangria_training_v2.h5'
ldc_dir = 'LDC/'
config = {'files': {'galaxy_file': galaxy_file, 'galaxy_dir': ldc_dir}}
params_gb, ns_got = gfi.get_full_galactic_params(config)

n_tot = int(ns_got.sum())
n_dgb = ns_got[0]
n_igb = ns_got[1]
n_vgb = ns_got[2]

n_dgb_sel = min(1000000, n_dgb)
n_igb_sel = min(100000, n_igb)
n_vgb_sel = min(1000, n_vgb)

params_dgb = params_gb[:n_dgb_sel]
params_igb = params_gb[n_dgb:n_dgb + n_igb][:n_igb_sel]
params_vgb = params_gb[n_dgb + n_igb:n_dgb + n_igb + n_vgb][:n_vgb_sel]

filename_out = 'Galaxies/Galaxy4/galaxy_binaries.hdf5'
hf_out = h5py.File(filename_out, 'w')
hf_sky = hf_out.create_group('sky')
hf_dgb = hf_sky.create_group('dgb')
hf_igb = hf_sky.create_group('igb')
hf_vgb = hf_sky.create_group('vgb')
hf_dgb_cat = hf_dgb.create_group('cat')
hf_igb_cat = hf_igb.create_group('cat')
hf_vgb_cat = hf_vgb.create_group('cat')

n_par_gb = 8
labels_gb = [
    'Amplitude',
    'EclipticLatitude',
    'EclipticLongitude',
    'Frequency',
    'FrequencyDerivative',
    'Inclination',
    'InitialPhase',
    'Polarization',
]

params_dgb[:, 0] *= 10
params_igb[:, 0] *= 10
params_vgb[:, 0] *= 10

for itrl in range(n_par_gb):
    _ = hf_dgb_cat.create_dataset(labels_gb[itrl], data=params_dgb[:, itrl], compression='gzip')
    _ = hf_igb_cat.create_dataset(labels_gb[itrl], data=params_igb[:, itrl], compression='gzip')
    _ = hf_vgb_cat.create_dataset(labels_gb[itrl], data=params_vgb[:, itrl], compression='gzip')

hf_out.close()
