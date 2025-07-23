#!/usr/bin/env bash
set -e

cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_AMX=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF  \
    -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 -DBLA_VENDOR=Intel10_64lp_dyn \
    "-DMKL_LIBRARIES=-Wl,--start-group;${MKL_PATH}/libmkl_intel_lp64.so;${MKL_PATH}/libmkl_gnu_thread.so;${MKL_PATH}/libmkl_core.so;-Wl,--end-group" \
    -DPython_EXECUTABLE=/root/miniforge3/${ENV_NAME}/bin/python3 \
    -DPython_INCLUDE_DIRS=/root/miniforge3/${ENV_NAME}/include/python3.11/ \
    -DPython_LIBRARIES=/root/miniforge3/${ENV_NAME}/lib/python3.11/ \
    -DPython_NumPy_INCLUDE_DIRS=/root/miniforge3/${ENV_NAME}/lib/python3.11/site-packages/numpy \
    .

make -C build -j faiss_avx512
make -C build -j swigfaiss
(cd build/faiss/python && python3 setup.py install)
make -C build install -j
