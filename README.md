# Zero Out GPU and CPU

nvcc -std=c++11 -c -o zero_out.cu.o zero_out.cu.cc -I $TF_INC -L /usr/local/cuda-8.0/lib64/ -
D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 ./zero_out.cu.o
