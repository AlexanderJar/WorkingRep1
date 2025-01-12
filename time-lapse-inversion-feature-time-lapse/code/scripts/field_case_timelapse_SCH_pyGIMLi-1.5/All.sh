mkdir -p data
cd data 
cp ../../../../data_timelapse_SCH/edited/* .
cd ..
python3 TL2_make_mesh_and_filter.py
python3 TL3_conventional_inversion.py
python3 TL4_apply_4PM.py
python3 TL5_joint_inversion.py
#python3 TL5_joint_inversion_Lcurve.py
python3 TL6_plot_inv_results_TL_vs_noTL.py
python3 Plot_1D_borehole_profiles.py

