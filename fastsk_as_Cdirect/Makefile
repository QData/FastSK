# Makefile for fastsk from C++ files source

# Compiler options
CXX = g++

CXXFLAGS = -lpthread -pthread -std=c++11 # openmp and pthread

# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = fastsk.o

.PHONY: all
all: fastsk

.PHONY: fastsk
fastsk: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o fastsk
	@echo Produced fastsk executable

install: fastsk
	mkdir -p ../bin
	cp $< ../bin/fastsk
	rm fastsk
	@echo fastsk executable installed in the fastsk-SVM/bin directory

clean:
	$(RM) *.o *~

# Dependency rules for *.o files
fastsk.o: fastsk.cpp shared.h shared.cpp readInput.cpp