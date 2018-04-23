# Makefile for iGakco

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -O0 -g -Wno-write-strings -D_GNU_SOURCE # openmp and pthread, g for debugging
INCLUDE=-I/usr/include/libsvm-code
# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = Gakco.o GakcoSVM.o libsvm-code/svm.o

gakco: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o Gakco
	@echo Produced Gakco executable 

clean:
	$(RM) *.o *~

# Dependency rules for *.o files
svm.o: ./libsvm-code/svm.cpp ./libsvm-code/svm.h
Gakco.o: Gakco.cpp shared.h shared.cpp readInput.cpp

GakcoSVM.o: GakcoSVM.cpp shared.cpp
