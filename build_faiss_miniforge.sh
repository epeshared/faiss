export MKL_PATH=/root/miniforge3/lib
export LD_LIBRARY_PATH=/root/miniforge3/lib/:$LD_LIBRARY_PATH
cmake -B build -DBUILD_TESTING=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_SHARED_LIBS=ON  \
    -DCMAKE_BUILD_TYPE=Release -DFAISS_OPT_LEVEL=avx512 -DBLA_VENDOR=Intel10_64lp_dyn \
    "-DMKL_LIBRARIES=-Wl,--start-group;${MKL_PATH}/libmkl_intel_lp64.so;${MKL_PATH}/libmkl_gnu_thread.so;${MKL_PATH}/libmkl_core.so;-Wl,--end-group" \
    -DPython_EXECUTABLE=/root/miniforge3/bin/python3 \
    -DPython_INCLUDE_DIRS=/root/miniforge3/include/python3.10/ \
    -DPython_LIBRARIES=/root/miniforge3/lib/python3.10/ \
    -DPython_NumPy_INCLUDE_DIRS=/root/miniforge3/lib/python3.10/site-packages/numpy

make -C build -j faiss_avx512
make -C build -j swigfaiss
(cd build/faiss/python && python3 setup.py install)
make -C build install
