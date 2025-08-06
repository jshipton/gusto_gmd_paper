# Runs test cases used for paper's results

terminator_toy:
	mpiexec -n 4 python3 test_case_scripts/terminator_toy.py \
		--ncells_per_edge 16 --dt 450.0 --dirname terminator_toy

galewsky_jet:
	mpiexec -n 4 python3 test_case_scripts/galewsky_jet.py \
		--ncells_per_edge 64 --dt 150.0 --dumpfreq 3456 --icosahedron

	mpiexec -n 4 python3 test_case_scripts/galewsky_jet.py \
		--ncells_per_edge 100 --dt 150.0 --dumpfreq 3456

baroclinic_wave:
	mpiexec -n 72 python3 test_case_scripts/baroclinic_wave.py \
		--ncells_per_edge 32

straka:
	mpiexec -n 4 python3 test_case_scripts/straka_degree_0.py \
		--nlayers 128 --dt 0.5 --dumpfreq 450 --dirname straka_degree_0

	mpiexec -n 4 python3 test_case_scripts/straka_degree_1.py \
		--nlayers 64 --dt 0.5 --dumpfreq 450 --dirname straka_degree_1

moist_3d_bubble:
	mpiexec -n 9 python3 test_case_scripts/moist_bubble_3d.py
