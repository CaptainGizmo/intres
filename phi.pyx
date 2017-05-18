# cython: cdivision=True
import cython
import numpy as np
cimport numpy as np
#from scipy.constants import pi
from libc.math cimport M_PI
from libc.math cimport exp
#from cython.parrallel import prange, parallel

ctypedef np.complex64_t cpl_t

#def extern from "<complex.h>" namespace "std":
#	double complex exp(double complex z)
#	float complex exp(float complex z)  # overload

#cdef extern from "math.h":
#	double M_PI

cdef extern from "complex.h":
	double complex cexp(double complex)
	float complex cexp(float complex)

@cython.boundscheck(False)
cdef inline cpl_t phi(float [:] kpt, int[:,:] igall, int nplane, cpl_t [:] coeff, float [:] r):
	cdef:
		int iplane
		cpl_t out = 0
		float [3] k

	for iplane in range(nplane):
		k[0] = kpt[0] + float(igall[iplane,0])
		k[1] = kpt[1] + float(igall[iplane,1])
		k[2] = kpt[2] + float(igall[iplane,2])
		out += coeff[iplane] * cexp( 2j * M_PI * (k[0]*r[0] + k[1]*r[1] + k[2]*r[2]))

	return out


@cython.boundscheck(False)
def phi_skn(float [:] kpt,int [:,:] igall, int nplane, cpl_t [:] coeff, double Vcell, long [:] rs, cpl_t [:,:,:] phi_out):

	cdef:
		cpl_t out = 0
		int x, y, z
		float [3] r

	for x in range(rs[0]):
		for y in range(rs[1]):
			for z in range(rs[2]):
				# convert indeces to reduced coordinates
				r[0] = x/float(rs[0])
				r[1] = y/float(rs[1])
				r[2] = z/float(rs[2])
				phi_out[x][y][z] = phi(kpt, igall, nplane, coeff, r) / (Vcell**0.5)

	return
