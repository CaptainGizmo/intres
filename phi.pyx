#!/usr/bin/env python3
# cython: cdivision=True
cimport cython
from mpi4py import MPI
#from mpi4py cimport MPI
#from mpi4py cimport mpi_c

#cimport openmp
#from cython.parallel cimport *
from libc.math cimport M_PI
#from libc.math cimport exp
#import ctypes
import numpy as np
cimport numpy as np
#ctypedef np.complex64_t cpl_t

cdef extern from "complex.h" nogil:
	double complex cexp(double complex)
	float complex cexp(float complex)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex phi(double* kpt, long * igall, long nplane, double complex * coeff, double * r) nogil:
	cdef:
		long iplane, dim
		double complex out = 0.0
		double k_r = 0.0
	
	for iplane in range(nplane):
		# k.dot.r
		k_r = 0.0
		for dim in range(3): k_r += (kpt[dim] + igall[3*iplane+dim]) * r[dim]
		out += coeff[iplane] * cexp( 2j * M_PI * k_r)

	return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef phi_skn(np.ndarray[double, ndim = 1] np_kpt, \
				np.ndarray[long, ndim = 2]  np_igall, \
				long nplane, \
				np.ndarray[double complex, ndim = 1]  np_coeff, \
				double Vcell, \
				np.ndarray[long, ndim = 1]  np_rs, \
				double complex [:,:,:] phi_out):

	#reshape continious-memory array
	cdef np.ndarray[long, ndim=2, mode = 'c']    np_buff  = np.ascontiguousarray(np_igall, dtype = long)
	cdef np.ndarray[double complex, ndim=1, mode = 'c'] np_buff2 = np.ascontiguousarray(np_coeff, dtype = np.complex128)

	cdef:
		long x, y, z
		double [3] r
		long idx
		long * igall = <long*> np_buff.data
		double * kpt = <double*> np_kpt.data
		double complex * coeff = <double complex*> np_buff2.data
		
		long XMAX = np_rs[0]
		long YMAX = np_rs[1]
		long ZMAX = np_rs[2]
		
		np.ndarray[double complex,ndim=3] phi_rank = np.zeros([XMAX,YMAX,ZMAX], dtype = np.complex128)
	
	comm = MPI.COMM_WORLD

	# Distribute workload so that each MPI process analyzes point number i, where
	#  i % comm.size == comm.rank

	for idx in range(comm.rank, XMAX*YMAX*ZMAX, comm.size):
			# convert common index to dimention indexes
			z = idx / (XMAX * YMAX)
			y = (idx - z * XMAX * YMAX) / XMAX
			x = (idx - z * XMAX * YMAX) % XMAX

			# convert indeces to reduced coordinates
			r[0] = x/(XMAX*1.0)
			r[1] = y/(YMAX*1.0)
			r[2] = z/(ZMAX*1.0)
			
			#write only our values, all other's are still 0
			phi_rank[x,y,z] = phi(kpt,igall,nplane,coeff,r) / (Vcell**0.5)

	#comm.Reduce(rank_out, phi_out, op=MPI.SUM, root = 0)
	#comm.Allreduce([rank_out, MPI.C_DOUBLE_COMPLEX], [phi_out, MPI.C_DOUBLE_COMPLEX], op=MPI.SUM)
	comm.Allreduce(phi_rank, phi_out, op=MPI.SUM)

	return
