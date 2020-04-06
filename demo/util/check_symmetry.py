# Python script to verify that Gakco output kernels are symmetric
# Run after performing updates as a sanity check or simply validate GaKCo's correctness

import sys
import os
import numpy as np

'''
For example, kernel must look like:
1:1.000000e+00 2:0.000000e+00
1:0.000000e+00 2:1.000000e+00
'''
def getNumpyMatrix(kernel_file):
	kernel = []
	with open(kernel_file) as f:
		line = f.readline()
		while line:
			row = []
			entries = line.split()
			for entry in entries:
				entry = float(entry.split(":")[1])
				row.append(entry)
			kernel.append(row)
			line = f.readline()
	return np.matrix(kernel)

# matrix argument must be a numpy matrix
def checkIfSymmetric(matrix, tolerance=1e-8):
	return np.allclose(matrix, matrix.T, atol=tolerance)

if __name__=="__main__":
	if (len(sys.argv) != 2):
		print("Usage: \n\tpython check_symmetry kernel.txt\n")
		print("\tKernel should be in format outputted by GaKCo. For example:")
		print("\t1:1.000000e+00 2:0.000000e+00\n\t1:0.000000e+00 2:1.000000e+00")
		sys.exit()
	kernel_file = sys.argv[1]
	print("Checking symmetry of {}...".format(kernel_file))
	kernel = getNumpyMatrix(kernel_file)
	symmetric = checkIfSymmetric(kernel)
	if (symmetric):
		print("{} is symmetric".format(kernel_file))
	else:
		print("{} is not symmetric".format(kernel_file))