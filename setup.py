#cython: language_level=3, boundscheck=False, wraparound=False
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import os
os.environ["CC"] = "icc"
os.environ["LDSHARED"] = "icc -shared"

setup(
  name = "phi",
  cmdclass = {"build_ext": build_ext},
  ext_modules =
  [
    Extension("phi",
              ["phi.pyx"],
              #extra_compile_args = ["-O3", "-fopenmp","-xMIC-AVX512"],
              extra_compile_args = ["-O3", "-fopenmp"],
              extra_link_args=['-fopenmp']
              )
  ]
)

# for compile run:
# python3 setup.py build_ext --inplace