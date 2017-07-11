# Makefile

TF_INC = `python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -L/usr/local/cuda-8.0/lib64/
GPUCFLAGS = -c
LFLAGS    = -pthread -shared -fPIC
GPULFLAGS = -x cu -Xcompiler -fPIC
GPUDEF    = -DGOOGLE_CUDA=1
CGPUFLAGS = -lcudart #-lcuda # shouldn't it be -lcudart ?? (not found)

SRC       = zero_out.cc
GPUSRC    = zero_out.cu.cc
PROD      = zero_out.so
GPUPROD   = zero_out_cu.o

default: gpu

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD) -D_GLIBCXX_USE_CXX11_ABI=0

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) 
	
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -f $(PROD) $(GPUPROD)
