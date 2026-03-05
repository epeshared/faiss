# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import platform
import shutil

from setuptools import setup

# make the faiss python package dir
shutil.rmtree("faiss", ignore_errors=True)
os.mkdir("faiss")
shutil.copytree("contrib", "faiss/contrib")
shutil.copyfile("__init__.py", "faiss/__init__.py")
shutil.copyfile("loader.py", "faiss/loader.py")
shutil.copyfile("class_wrappers.py", "faiss/class_wrappers.py")
shutil.copyfile("gpu_wrappers.py", "faiss/gpu_wrappers.py")
shutil.copyfile("extra_wrappers.py", "faiss/extra_wrappers.py")
shutil.copyfile("array_conversions.py", "faiss/array_conversions.py")

if os.path.exists("__init__.pyi"):
    shutil.copyfile("__init__.pyi", "faiss/__init__.pyi")
if os.path.exists("py.typed"):
    shutil.copyfile("py.typed", "faiss/py.typed")

if platform.system() != "AIX":
    ext = ".pyd" if platform.system() == "Windows" else ".so"
else:
    ext = ".a"
prefix = "Release/" * (platform.system() == "Windows")

def check_exists(libname):
    if os.path.exists(libname):
        return libname
    if os.path.exists(os.path.join("faiss", libname)):
        return os.path.join("faiss", libname)
    return None

swigfaiss_generic_lib_name = f"{prefix}_swigfaiss{ext}"
swigfaiss_avx2_lib_name = f"{prefix}_swigfaiss_avx2{ext}"
swigfaiss_avx512_lib_name = f"{prefix}_swigfaiss_avx512{ext}"
swigfaiss_avx512_spr_lib_name = f"{prefix}_swigfaiss_avx512_spr{ext}"
callbacks_lib_name = f"{prefix}libfaiss_python_callbacks{ext}"
swigfaiss_sve_lib_name = f"{prefix}_swigfaiss_sve{ext}"
faiss_example_external_module_lib_name = f"_faiss_example_external_module{ext}"

swigfaiss_generic_lib = check_exists(swigfaiss_generic_lib_name)
swigfaiss_avx2_lib = check_exists(swigfaiss_avx2_lib_name)
swigfaiss_avx512_lib = check_exists(swigfaiss_avx512_lib_name)
swigfaiss_avx512_spr_lib = check_exists(swigfaiss_avx512_spr_lib_name)
callbacks_lib = check_exists(callbacks_lib_name)
swigfaiss_sve_lib = check_exists(swigfaiss_sve_lib_name)
faiss_example_external_module_lib = check_exists(faiss_example_external_module_lib_name)

found_swigfaiss_generic = swigfaiss_generic_lib is not None
found_swigfaiss_avx2 = swigfaiss_avx2_lib is not None
found_swigfaiss_avx512 = swigfaiss_avx512_lib is not None
found_swigfaiss_avx512_spr = swigfaiss_avx512_spr_lib is not None
found_callbacks = callbacks_lib is not None
found_swigfaiss_sve = swigfaiss_sve_lib is not None
found_faiss_example_external_module_lib = faiss_example_external_module_lib is not None

if platform.system() != "AIX":
    assert (
        found_swigfaiss_generic
        or found_swigfaiss_avx2
        or found_swigfaiss_avx512
        or found_swigfaiss_avx512_spr
        or found_swigfaiss_sve
        or found_faiss_example_external_module_lib
    ), (
        f"Could not find {swigfaiss_generic_lib_name} or "
        f"{swigfaiss_avx2_lib_name} or {swigfaiss_avx512_lib_name} or {swigfaiss_avx512_spr_lib_name} or {swigfaiss_sve_lib_name} or {faiss_example_external_module_lib_name}. "
        f"Faiss may not be compiled yet."
    )

if found_swigfaiss_generic:
    print(f"Copying {swigfaiss_generic_lib}")
    shutil.copyfile("swigfaiss.py", "faiss/swigfaiss.py")
    shutil.copyfile(swigfaiss_generic_lib, f"faiss/_swigfaiss{ext}")

if found_swigfaiss_avx2:
    print(f"Copying {swigfaiss_avx2_lib}")
    shutil.copyfile("swigfaiss_avx2.py", "faiss/swigfaiss_avx2.py")
    shutil.copyfile(swigfaiss_avx2_lib, f"faiss/_swigfaiss_avx2{ext}")

if found_swigfaiss_avx512:
    print(f"Copying {swigfaiss_avx512_lib}")
    shutil.copyfile("swigfaiss_avx512.py", "faiss/swigfaiss_avx512.py")
    shutil.copyfile(swigfaiss_avx512_lib, f"faiss/_swigfaiss_avx512{ext}")

if found_swigfaiss_avx512_spr:
    print(f"Copying {swigfaiss_avx512_spr_lib}")
    shutil.copyfile("swigfaiss_avx512_spr.py", "faiss/swigfaiss_avx512_spr.py")
    shutil.copyfile(swigfaiss_avx512_spr_lib, f"faiss/_swigfaiss_avx512_spr{ext}")

if found_callbacks:
    print(f"Copying {callbacks_lib}")
    shutil.copyfile(callbacks_lib, f"faiss/{callbacks_lib}")

if found_swigfaiss_sve:
    print(f"Copying {swigfaiss_sve_lib}")
    shutil.copyfile("swigfaiss_sve.py", "faiss/swigfaiss_sve.py")
    shutil.copyfile(swigfaiss_sve_lib, f"faiss/_swigfaiss_sve{ext}")

if found_faiss_example_external_module_lib:
    print(f"Copying {faiss_example_external_module_lib}")
    shutil.copyfile(
        "faiss_example_external_module.py", "faiss/faiss_example_external_module.py"
    )
    shutil.copyfile(
        faiss_example_external_module_lib,
        f"faiss/_faiss_example_external_module{ext}",
    )

long_description = """
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""
setup(
    name="faiss",
    version="1.14.0",
    description="A library for efficient similarity search and clustering of dense vectors",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/facebookresearch/faiss",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="faiss@meta.com",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy", "packaging"],
    packages=["faiss", "faiss.contrib", "faiss.contrib.torch"],
    package_data={
        "faiss": ["*.so", "*.pyd", "*.a", "*.pyi", "py.typed"],
    },
    zip_safe=False,
)
