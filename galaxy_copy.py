import h5py

import GalacticStochastic.global_file_index as gfi

galaxy_file = 'LDC2_sangria_training_v2.h5'
ldc_dir = 'LDC/'
params_gb, n_dgb, n_igb, n_vgb, n_tot = gfi.get_full_galactic_params(galaxy_file, ldc_dir)

n_dgb_sel = min(1000000, n_dgb)
n_igb_sel = min(100000, n_igb)
n_vgb_sel = min(1000, n_vgb)

params_dgb = params_gb[:n_dgb_sel]
params_igb = params_gb[n_dgb:n_dgb + n_igb][:n_igb_sel]
params_vgb = params_gb[n_dgb + n_igb:n_dgb + n_igb + n_vgb][:n_vgb_sel]

filename_out = 'Galaxies/Galaxy4/galaxy_binaries.hdf5'
hf_out = h5py.File(filename_out, 'w')
hf_out.create_group('sky')
hf_out['sky'].create_group('dgb')
hf_out['sky'].create_group('igb')
hf_out['sky'].create_group('vgb')
hf_out['sky']['dgb'].create_group('cat')
hf_out['sky']['vgb'].create_group('cat')
hf_out['sky']['igb'].create_group('cat')

n_par_gb = 8
labels_gb = ['Amplitude', 'EclipticLatitude', 'EclipticLongitude', 'Frequency', 'FrequencyDerivative', 'Inclination', 'InitialPhase', 'Polarization']

params_dgb[:, 0] *= 10
params_vgb[:, 0] *= 10
params_igb[:, 0] *= 10

for itrl in range(n_par_gb):
    hf_out['sky']['dgb']['cat'].create_dataset(labels_gb[itrl], data=params_dgb[:, itrl], compression='gzip')
    hf_out['sky']['vgb']['cat'].create_dataset(labels_gb[itrl], data=params_vgb[:, itrl], compression='gzip')
    hf_out['sky']['igb']['cat'].create_dataset(labels_gb[itrl], data=params_igb[:, itrl], compression='gzip')

hf_out.close()
