python temporal_generator.py --preset linear_gaussian --save data_generated/lg_2layer_sigma1.npz
python temporal_generator.py --preset linear_laplace --save data_generated/ll_2layer_sigma1.npz
python temporal_generator.py --preset linear_uniform --save data_generated/lu_2layer_sigma1.npz

python temporal_generator.py --preset lg_nonlinear_mixing --save data_generated/lg_nm_2layer_sigma1.npz
python temporal_generator.py --preset ll_nonlinear_mixing --save data_generated/ll_nm_2layer_sigma1.npz
python temporal_generator.py --preset lu_nonlinear_mixing --save data_generated/lu_nm_2layer_sigma1.npz

python temporal_generator.py --preset g_nonlinear --save data_generated/g_nl_2layer_sigma1.npz
python temporal_generator.py --preset l_nonlinear --save data_generated/l_nl_2layer_sigma1.npz
python temporal_generator.py --preset u_nonlinear --save data_generated/u_nl_2layer_sigma1.npz

# python temporal_generator.py --preset three_layer_two_var --multi --num_traj 16 --T_each 1000
# python temporal_generator.py --preset nonlinear_transition --multi --num_traj 8 --T_each 2000 --save ./data/nonlinear_transition.npz