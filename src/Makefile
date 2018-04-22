# Makefile for GaKCO

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -O0 -g -Wno-write-strings -D_GNU_SOURCE # openmp and pthread, g for debugging
INCLUDE=-I/usr/include/libsvm-code
# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = GaKCo.o GakcoSVM.o libsvm-code/svm.o

gakco: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o GaKCo
	@echo Produced GaKCo executable 

clean:
	$(RM) *.o *~

# Dependency rules for *.o files
svm.o: ./libsvm-code/svm.cpp ./libsvm-code/svm.h
GaKCo.o: GaKCo.cpp shared.h shared.cpp readInput.cpp

GakcoSVM.o: GakcoSVM.cpp shared.cpp
