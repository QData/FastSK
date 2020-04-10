# Makefile for fastsk-asC

# Compiler options
CXX = g++ # use g++ compiler
CXXFLAGS = -lpthread -pthread -std=c++11 -O3 -Wno-write-strings -D_GNU_SOURCE # pthread, g for debugging
#INCLUDE=-I/usr/include/libsvm-code

# To get an o file, we use the cpp file
.SUFFIXES: .o .cpp
OFILES = main.o fastsk.o fastsk_kernel.o fastsk_svm.o shared.o utils.o libsvm-code/eval.o libsvm-code/svm.o

fastsk: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o main
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
svm.o: ./libsvm-code/svm.cpp ./libsvm-code/eval.cpp
eval.o: ./libsvm-code/eval.cpp
fastsk.o: fastsk.cpp fastsk_kernel.cpp utils.cpp shared.cpp fastsk_svm.cpp
shared.o: shared.cpp
utils.o: utils.cpp
fastsk_kernel.o: fastsk_kernel.cpp shared.cpp 
fastsk_svm.o: libsvm-code/svm.cpp libsvm-code/eval.cpp
main.o: main.cpp fastsk.cpp
