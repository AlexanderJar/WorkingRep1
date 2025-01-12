clean:
	rm -rf *.bmat *.npz *.png *.npy *.vector *.dat *.vec invalid.data *.vtk *.bms constraint.matrix && ls

calc:
	python3 ./TL1_create_model.py
	python3 ./TL2_create_syn_data.py

inv:
	python3 ./TL3_conventional_inversion.py 2

case2:
	python3 ./TL4_apply_4PM.py 2
	python3 ./TL5_joint_inversion.py 2


all: calc inv case2
