import cython
import numpy as np
cimport numpy as np
#from scipy.linalg import *
from scipy.constants import pi
#from math import sqrt


ctypedef np.complex64_t cpl_t
cpl = np.complex64

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
def phi(np.ndarray[float, ndim = 1] kpt, np.ndarray[int, ndim = 2] igall, int nplane, np.ndarray[cpl_t, ndim = 1] coeff, double Vcell, long [:] rs, np.ndarray[cpl_t, ndim = 3] out):
	"Return wavefunction value rs[0]*rs[1]*rs[2] points in real space"
	cdef int x, y, z, iplane
	cdef complex csum
	#cdef double complex [:,:,:] out
	cdef np.ndarray[float,ndim=1] k = np.zeros((3),dtype = np.float32)
	cdef np.ndarray[float,ndim=1] r = np.zeros((3),dtype = np.float32)

	#out = np.empty([rs[0],rs[1],rs[2]],dtype='complex64')
	
	for x in range(int(rs[0])):
		for y in range(int(rs[1])):
			for z in range(int(rs[2])):
				csum = complex(0.,0.)
				for iplane in range(nplane):
					k[0] = kpt[0] + igall[iplane][0]
					k[1] = kpt[1] + igall[iplane][1]
					k[2] = kpt[2] + igall[iplane][2]
					
					r[0] = x
					r[1] = y
					r[2] = z
					
					csum += coeff[iplane] * np.exp( 2.* pi * 1j * k.dot(r) )
				out[x][y][z] = csum / np.sqrt(Vcell)

	return
