# Runs test cases used for paper's results

terminator_toy:
	mpiexec -n 4 python3 test_case_scripts/terminator_toy.py \
		--ncells_per_edge 16 --dt 450.0 --dirname terminator_toy

galewsky_jet:
	mpiexec -n 4 python3 test_case_scripts/galewsky_jet.py \
		--ncells_per_edge 64 --dt 150.0 --dumpfreq 3456 --icosahedron

	mpiexec -n 4 python3 test_case_scripts/galewsky_jet.py \
		--ncells_per_edge 100 --dt 150.0 --dumpfreq 3456

williamson_5:
	mpiexec -n 4 python3 test_case_scripts/williamson_5.py \
		--ncells_per_edge 64 --dt 300.0 --dumpfreq 1440 --semi_implicit

	mpiexec -n 4 python3 test_case_scripts/williamson_5.py \
		--ncells_per_edge 64 --dt 50.0 --dumpfreq 8640

straka:
	mpiexec -n 4 python3 test_case_scripts/straka_degree_1.py \
		--nlayers 64 --dt 1.0 --dumpfreq 450 --dirname straka_degree_1

	mpiexec -n 4 python3 test_case_scripts/straka_degree_0.py \
		--nlayers 128 --dt 1.0 --dumpfreq 450 --dirname straka_degree_0

moist_3d_bubble:
	mpiexec -n 9 python3 test_case_scripts/moist_3d_bubble.py