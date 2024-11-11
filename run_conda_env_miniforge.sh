#!/bin/bash -e

# conda install -c pytorch faiss-cpu=1.7.4 mkl=2023 blas=1.0=mkl
conda install -c pytorch mkl=2023 blas=1.0=mkl
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/root/miniforge3/lib/:/usr/local/lib64/"
ln -s /root/miniforge3/lib/libmkl_intel_lp64.so.2 /root/miniforge3/lib/libmkl_intel_lp64.so.1
ln -s /root/miniforge3/lib/libmkl_gnu_thread.so.2 /root/miniforge3/lib/libmkl_gnu_thread.so.1
ln -s /root/miniforge3/lib/libmkl_core.so.2 /root/miniforge3/lib/libmkl_core.so.1

ln -s /root/miniforge3/lib/libmkl_intel_lp64.so.2 /usr/local/lib/libmkl_intel_lp64.so.2
ln -s /root/miniforge3/lib/libmkl_gnu_thread.so.2 /usr/local/lib/libmkl_gnu_thread.so.2
ln -s /root/miniforge3/lib/libmkl_core.so.2 /usr/local/lib/libmkl_core.so.2


# python3 bench_polysemous_sift1m.py
# python3 bench_polysemous_1bn.py Deep1B OPQ20_80,IMI2x14,PQ20 autotune
