CXX = g++
CXXFLAGS = -lpthread -pthread -std=c++11 -O3 -Wno-write-strings -D_GNU_SOURCE

.SUFFIXES: .o .cpp
OFILES = main.o fastsk.o fastsk_kernel.o shared.o utils.o libsvm-code/eval.o libsvm-code/svm.o libsvm-code/svm-predict.o

main: $(OFILES)
	$(CXX) $(CXXFLAGS) $(OFILES) -o fastsk
	@echo Produced fastsk executable

clean:
	$(RM) *.o *~ fastsk

.PHONY: all
all: main

install: 
	mkdir -p ../bin
	cp $< ../bin/fastsk
	rm fastsk
	@echo fastsk executable installed in the fastsk-asC/bin directory

main: main.cpp fastsk.cpp
fastsk.o: fastsk.cpp shared.cpp utils.cpp fastsk_kernel.cpp libsvm-code/svm.cpp libsvm-code/eval.cpp
shared.o: shared.cpp
utils.o: utils.cpp
fastsk_kernel.o: fastsk_kernel.cpp shared.cpp 
libsvm-code/svm.o: libsvm-code/svm.cpp
libsvm-code/eval.o: libsvm-code/eval.cpp libsvm-code/svm.cpp libsvm-code/svm-predict.c 