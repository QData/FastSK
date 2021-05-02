import os
import re

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion
from pybind11.setup_helpers import Pybind11Extension, build_ext

#from docs import conf as docs_conf

with open('README.md') as f:
    long_description = f.read()


extras = {}
# Packages required for installing docs.
extras["docs"] = [
    "recommonmark",
    "nbsphinx",
    "sphinx-autobuild",
    "sphinx-rtd-theme"
]
# Packages required for formatting code & running tests.
extras["test"] = [
    "black==20.8b1",
    "docformatter",
    "isort==5.6.4",
    "flake8",
    "pytest",
    "pytest-xdist",
]

# For developers, install development tools along with all optional dependencies.
extras["dev"] = (
    extras["docs"] + extras["test"] 
)

def get_cpp_sources():
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
        sources=get_cpp_sources()
    ),
]

setup(
    name="fastsk",
    version="0.0.1",
    author="QData Lab at the University of Virginia",
    author_email="yanjun@virginia.edu",
    description="A library for generating gkm-svm faster",
    include_package_data=False,
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QData/FastSk",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").readlines(),
    zip_safe=False,
)
