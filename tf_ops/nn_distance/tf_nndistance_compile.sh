TF_INC='/home/lhx/.virtualenvs/tensorflow/local/lib/python2.7/site-packages/tensorflow/include/'
TF_LIB='/home/lhx/.virtualenvs/tensorflow/local/lib/python2.7/site-packages/tensorflow/'

/usr/local/cuda-9.0/bin/nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-9.0/include -I $TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
