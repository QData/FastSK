cimport cigakco
from cpython cimport array
import array
import struct

# distutils: language = c++

def Init_iGakco():
	return None

def igakco(g, m, trainfile, testfile, dictionary, labels, kernel_type=1, halt=None, kernelfile=None, modelfile=None, C=1, probability=False, threads=4, loadkernel=False, loadmodel=False):
	cdef int numArgs = 15
	cdef char* argv[35]
	trainfile = trainfile.encode()
	testfile = testfile.encode()
	dictionary = dictionary.encode()
	labels = labels.encode()


	argv[0] = "./iGakco"
	argv[1] = trainfile
	argv[2] = testfile
	argv[3] = dictionary
	argv[4] = labels

	gflag = str(g).encode()
	mflag = str(m).encode()
	t = str(threads).encode()
	r = str(kernel_type).encode()
	c=str(C).encode()#bytearray(struct.pack("f", C))


	argv[5] = "-g"
	argv[6] = gflag
	argv[7] = "-m"
	argv[8] = mflag
	argv[9] = "-r"
	argv[10] = r
	argv[11] = "-C"
	argv[12] = c
	argv[13] = "-t"
	argv[14] = t

	if halt:
		argv[numArgs] = "-h"
		numArgs+=1
		h = bytes(halt)
		argv[numArgs] = h
		numArgs+=1
	if kernelfile:
		kernelfile = kernelfile.encode()
		argv[numArgs] = "-k"
		numArgs+=1
		argv[numArgs] = kernelfile
		numArgs+=1
	if modelfile:
		modelfile = modelfile.encode()
		argv[numArgs] = "-o"
		numArgs+=1
		argv[numArgs] = modelfile
		numArgs+=1
	if probability:
		argv[numArgs] = "-p"
		numArgs+=1
		argv[numArgs] = "1"
		numArgs +=1



	return cigakco.igakco_main_wrapper(numArgs, argv)
	
