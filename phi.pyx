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
cdef float complex phi(double* kpt, long * igall, long nplane, float complex * coeff, double * r) nogil:
	cdef:
		long iplane, dim
		float complex out = 0.0
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
				np.ndarray[float complex, ndim = 1]  np_coeff, \
				double Vcell, \
				np.ndarray[long, ndim = 1]  np_rs, \
				float complex [:,:,:] phi_out):

	#reshape continious-memory array
	cdef np.ndarray[long, ndim=2, mode = 'c']    np_buff  = np.ascontiguousarray(np_igall, dtype = long)
	cdef np.ndarray[float complex, ndim=1, mode = 'c'] np_buff2 = np.ascontiguousarray(np_coeff, dtype = np.complex64)

	cdef:
		long x, y, z
		double [3] r
		long idx
		long * igall = <long*> np_buff.data
		double * kpt = <double*> np_kpt.data
		float complex * coeff = <float complex*> np_buff2.data
		
		long XMAX = np_rs[0]
		long YMAX = np_rs[1]
		long ZMAX = np_rs[2]
		
		np.ndarray[float complex,ndim=3] phi_rank = np.zeros([XMAX,YMAX,ZMAX], dtype = np.complex64)
	
	comm = MPI.COMM_WORLD

	# Distribute workload so that each MPI process analyzes point number i, where i % comm.size == comm.rank
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

	comm.Allreduce(phi_rank, phi_out, op=MPI.SUM)

	return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float complex T(np.ndarray[float complex, ndim = 3] np_phi_f, \
	np.ndarray[double, ndim = 3] np_dV, \
	np.ndarray[float complex, ndim = 3] np_phi_i):

	#reshape continious-memory array
	cdef np.ndarray[double, ndim=1, mode = 'c']    np_buff_dV  = np.ascontiguousarray(np_dV.reshape(-1), dtype = np.float64)
	cdef np.ndarray[float complex, ndim=1, mode = 'c'] np_buff_phi_f = np.ascontiguousarray(np_phi_f.reshape(-1), dtype = np.complex64)
	cdef np.ndarray[float complex, ndim=1, mode = 'c'] np_buff_phi_i = np.ascontiguousarray(np_phi_i.reshape(-1), dtype = np.complex64)

	cdef:
		#long x, y, z
		long idx
		float complex * phi_f = <float complex*> np_buff_phi_f.data
		float complex * phi_i = <float complex*> np_buff_phi_i.data
		double * dV = <double*> np_buff_dV.data
		
		long XMAX = np.shape(np_dV)[0]
		long YMAX = np.shape(np_dV)[1]
		long ZMAX = np.shape(np_dV)[2]
		
		float complex T_rank = 0.0
		float complex T_out  = 0.0

	#print(XMAX,YMAX,ZMAX)
	comm = MPI.COMM_WORLD

	# Distribute workload so that each MPI process analyzes point number i, where i % comm.size == comm.rank
	#for idx in range(comm.rank, XMAX*YMAX*ZMAX, comm.size):
	#	# convert common index to dimention indexes
	#	z = idx / (XMAX * YMAX)
	#	y = (idx - z * XMAX * YMAX) / XMAX
	#	x = (idx - z * XMAX * YMAX) % XMAX
	#	T_rank += ( np_phi_f[x,y,z].real - np_phi_f[x,y,z].imag ) * np_dV[x,y,z] * np_phi_i[x,y,z]
	#	#print(comm.rank,T_rank)
	
	for idx in range(comm.rank, XMAX*YMAX*ZMAX, comm.size):
		T_rank += np.conj(phi_f[idx]) * dV[idx] * phi_i[idx]
		#T_rank += ( phi_f[idx].real - phi_f[idx].imag ) * dV[idx] * phi_i[idx]

	T_out = comm.allreduce(T_rank, op=MPI.SUM)

	return T_out
