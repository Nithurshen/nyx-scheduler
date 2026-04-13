CC=g++
CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib64

SRCS = minimal_intercept.cpp \
       orion_original/intercept_temp.cpp \
       orion_original/intercept_cudnn.cpp \
       orion_original/intercept_cublas.cpp \
       orion_original/utils_interc.cpp

liborion_capture.so: $(SRCS)
	$(CC) -O3 -fPIC -shared $(SRCS) -I$(CUDAINCLUDE) -L$(CUDALIB) -lcudart -lcublas -lcudnn -ldl -o liborion_capture.so

all: liborion_capture.so

clean:
	rm -rf *.o liborion_capture.so