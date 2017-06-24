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
cdef complex phi(double* kpt, long * igall, long nplane, complex * coeff, double * r) nogil:
	cdef:
		long iplane, dim
		complex out = 0.0
		double k_r = 0.0
	
	for iplane in range(nplane):
		# k.dot.r
		for dim in range(3): k_r += (kpt[dim] + igall[3*iplane+dim]) * r[dim]
		out += coeff[iplane] * cexp( 2j * M_PI * k_r)

	return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef phi_skn(np.ndarray[double, ndim = 1] np_kpt, \
				np.ndarray[long, ndim = 2]  np_igall, \
				long nplane, \
				np.ndarray[complex, ndim = 1]  np_coeff, \
				double Vcell, \
				np.ndarray[long, ndim = 1]  np_rs, \
				complex [:,:,:] phi_out):

	#reshape continious-memory array
	cdef np.ndarray[long, ndim=2, mode = 'c']    np_buff  = np.ascontiguousarray(np_igall, dtype = long)
	cdef np.ndarray[complex, ndim=1, mode = 'c'] np_buff2 = np.ascontiguousarray(np_coeff, dtype = complex)

	cdef:
		long x, y, z
		double [3] r
		long idx
		long * igall = <long*> np_buff.data
		double * kpt = <double*> np_kpt.data
		complex * coeff = <complex*> np_buff2.data
		
		long XMAX = np_rs[0]
		long YMAX = np_rs[1]
		long ZMAX = np_rs[2]
		
		np.ndarray[complex,ndim=3] rank_out = np.zeros([XMAX,YMAX,ZMAX],dtype = complex)
	
	comm = MPI.COMM_WORLD

	# Distribute workload so that each MPI process analyzes image number i, where
	#  i % comm.size == comm.rank
	#
	# For example if comm.size == 4:
	#   rank 0: 0, 4, 8, ...
	#   rank 1: 1, 5, 9, ...
	#   rank 2: 2, 6, 10, ...
	#   rank 3: 3, 7, 11, ...
	
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
			rank_out[x,y,z] = phi(kpt,igall,nplane,coeff,r) / (Vcell**0.5)
	
	# Now reduce the matrix into *phi_out* by summing all *out* arrays
	# them all together. The result is only avalable at rank 0.
	# If you want the result to be availabe on all processes, use
	# Allreduce(...)
	
	#spec = np.zeros_like(rank_out)
	#comm.Reduce(rank_out, phi_out, op=MPI.SUM, root = 0)
	comm.Allreduce([rank_out, MPI.C_DOUBLE_COMPLEX], [phi_out, MPI.C_DOUBLE_COMPLEX], op=MPI.SUM)
	#comm.Allreduce([rank_out, MPI.COMPLEX], [phi_out, MPI.COMPLEX], op=MPI.SUM)

	return
