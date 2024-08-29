# Runs test cases used for paper's results

straka:
	mpiexec -n 4 python3 test_case_scripts/straka_degree_1.py \
		--nlayers 64 --dt 1.0 --dumpfreq 450 --dirname straka_degree_1

	mpiexec -n 4 python3 test_case_scripts/straka_degree_0.py \
		--nlayers 128 --dt 1.0 --dumpfreq 450 --dirname straka_degree_0

moist_3d_bubble:
	mpiexec -n 9 python3 test_case_scripts/moist_3d_bubble.py