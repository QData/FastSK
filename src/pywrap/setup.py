#This is supporting the python wrapping of the iGakco module

from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

os.environ["CC"] = "g++" 
os.environ["CXX"] = "g++"

# the c++ extension module
extension_mod = Extension("igakco", 
						sources=["shared.cpp","igakco.c", "Gakco.cpp" ,"GakcoSVM.cpp", "readInput.cpp","libsvm-code/svm.cpp"],
						language= "c++",
						include_dirs=["/usr/include/libsvm-code","/usr/include/","../src"],
						extra_compile_args=['-std=c++11','-D_GNU_SOURCE'],
						 )

setup(name = "iGakco",
		version= "1.0.0",
		ext_modules=cythonize([extension_mod], quiet=True)
	)