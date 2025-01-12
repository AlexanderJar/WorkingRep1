clean:
	rm -rf *.bmat *.npz *.png *.npy *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls

calc:
	python3 ./TL1_create_model.py
	python3 ./TL2_create_syn_data.py

inv:
	python3 ./TL3_conventional_inversion.py

case1:
	python3 ./TL4_apply_4PM.py
	python3 ./TL5_joint_inversion.py
	python3 ./TL6_plot_inv_results.py
	python3 ./TL6_plot_inv_results_with_sum.py

all: clean calc inv case1
