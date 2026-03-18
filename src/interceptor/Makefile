CC=g++
CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib64

liborion_capture.so: minimal_intercept.cpp
	$(CC) -O3 -fPIC -shared minimal_intercept.cpp -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -ldl -o liborion_capture.so

all:
	make liborion_capture.so

clean:
	rm -rf *.o liborion_capture.so
