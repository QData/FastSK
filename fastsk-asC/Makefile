# Makefile for fastsk-asC

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -O3 -Wno-write-strings -D_GNU_SOURCE # pthread, g for debugging
INCLUDE=-I/usr/include/libsvm-code

# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = fastsk.o kernel.o svm.o shared.o dataset.o

fastsk: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o fastsk
	@echo Produced fastsk executable 

clean:
	$(RM) *.o *~

.PHONY: all
all: fastsk


install: 
	mkdir -p ../bin
	cp $< ../bin/fastsk
	rm fastsk
	@echo fastsk executable installed in the fastsk-asC/bin directory


# Dependency rules for *.o files
svm.o: svm.cpp
fastsk.o: shared.cpp fastsk.cpp dataset.cpp
shared.o: shared.cpp
dataset.o: dataset.cpp
kernel.o: dataset.cpp kernel.cpp shared.cpp
