CXX = g++
CXXFLAGS = -lpthread -pthread -std=c++11 -O3 -Wall -Wpedantic -Wno-write-strings -D_GNU_SOURCE

.SUFFIXES: .o .cpp
OFILES = main.o fastsk.o fastsk_kernel.o shared.o libsvm-code/eval.o libsvm-code/svm.o libsvm-code/svm-predict.o

main: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o fastsk
	@echo Produced fastsk executable

clean:
	$(RM) *.o *~ fastsk

.PHONY: all
all: main

main: main.cpp fastsk.cpp
fastsk.o: fastsk.cpp shared.cpp fastsk_kernel.cpp libsvm-code/svm.cpp libsvm-code/eval.cpp
shared.o: shared.cpp
fastsk_kernel.o: fastsk_kernel.cpp shared.cpp 
libsvm-code/svm.o: libsvm-code/svm.cpp
libsvm-code/eval.o: libsvm-code/eval.cpp libsvm-code/svm.cpp libsvm-code/svm-predict.c 