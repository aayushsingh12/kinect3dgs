import os
import subprocess

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

prefix = os.environ["KINECT3DGS_PREFIX"]
usb_inc = subprocess.run(["pkg-config", "--cflags-only-I", "libusb-1.0"],
                         capture_output=True, text=True, check=True).stdout.split()
include_dirs = [f"{prefix}/include/libfreenect", np.get_include()] + [i[2:] for i in usb_inc]

ext = Extension(
    "freenect", ["freenect.pyx"],
    include_dirs=include_dirs,
    libraries=["freenect", "freenect_sync"],
    library_dirs=[f"{prefix}/lib"],
    runtime_library_dirs=[f"{prefix}/lib"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)
setup(ext_modules=cythonize([ext], language_level=3))
