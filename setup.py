import os
import re
import sys
import platform
import subprocess
import glob

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion

from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('README.md') as f:
    long_description = f.read()


def get_sources():
    return sorted([
        "src/fastsk/_fastsk/bindings.cpp",
        "src/fastsk/_fastsk/fastsk.cpp",
        "src/fastsk/_fastsk/fastsk_kernel.cpp",
        "src/fastsk/_fastsk/shared.cpp",
        "src/fastsk/_fastsk/libsvm-code/svm.cpp",
        "src/fastsk/_fastsk/libsvm-code/eval.cpp",
    ])

ext_modules = [
    Pybind11Extension(
        name="fastsk._fastsk",
        sources=get_sources()
    ),
]

setup(
    name="fastsk-test",
    version="1.0.0",
    author="Derrick Blakely",
    author_email="blakelyderrick@gmail.com",
    description="FastSK PyPi Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

