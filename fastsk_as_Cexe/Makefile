# Makefile for GaKCO

# Compiler options
CXX = g++

CXXFLAGS = -lpthread -pthread -std=c++11 # openmp and pthread

# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = GaKCo.o

.PHONY: all
all: gakco

.PHONY: gakco
gakco: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o GaKCo
	@echo Produced GaKCo executable

install: GaKCo
	mkdir -p ../bin
	cp $< ../bin/GaKCo
	rm GaKCo
	@echo GaKCo executable installed in the GaKCo-SVM/bin directory

clean:
	$(RM) *.o *~

# Dependency rules for *.o files
GaKCo.o: GaKCo.cpp shared.h shared.cpp readInput.cpp