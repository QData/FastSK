# Makefile for iGakcO

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -g -O3 -Wno-write-strings -D_GNU_SOURCE # openmp and pthread, g for debugging
INCLUDE=-I/usr/include/libsvm-code
# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = Gakco.o GakcoSVM.o libsvm-code/svm.o shared.o readInput.o

gakco: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o iGakco
	@echo Produced iGakco executable 

python: pywrap/cigakco.pxd pywrap/igakco.pyx $(OFILES)
	cython pywrap/igakco.pyx 
	python pywrap/setup.py build_ext -i 

python3: pywrap/cigakco.pxd pywrap/igakco.pyx $(OFILES)
	cython pywrap/igakco.pyx
	python3 pywrap/setup.py build_ext -i

clean:
	$(RM) *.o *~

# Dependency rules for *.o files
svm.o: ./libsvm-code/svm.cpp
Gakco.o: shared.cpp Gakco.cpp readInput.cpp
shared.o: shared.cpp
readInput.o: readInput.cpp
GakcoSVM.o: readInput.cpp GakcoSVM.cpp shared.cpp

