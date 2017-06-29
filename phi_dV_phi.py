#!/usr/bin/env python3
import os,sys,time
import subprocess
import struct
from ase import Atoms
from ase.calculators.vasp import *
import numpy as np
from numpy import linalg as LA
from numpy.lib import pad
from scipy.linalg import *
from scipy.constants import *
from scipy import sparse as sps
from sympy import DiracDelta
from math import sin,cos,asin,acos,sqrt

#import mpi4py
#mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import phi
from VaspKpointsInterpolated import *
from ibz2fbz import *

class Lifetime(object):

	class wf():
		def __init__(self,nspin,nkpt,npmax,nband):
			# assign memory
			self.occ   = np.zeros([nspin,nkpt,nband], dtype = 'float64')
			self.cener = np.zeros([nspin,nkpt,nband], dtype = 'complex128')
			self.igall = np.zeros([nspin,nkpt,npmax,3],dtype='int_')
			self.coeff = np.zeros([nspin,nkpt,nband,npmax],dtype='complex128')
			self.kpt   = np.zeros([nspin,nkpt,3],dtype='float64')
			self.nplane = np.zeros([nspin,nkpt],dtype='int_')
			self.Vcell = 0
			self.nband = nband

	def __init__(self, debug=False, restart= False):
		self.comm = MPI.COMM_WORLD
		self.debug = debug
		self.restart = restart

		#reading wavefunction
		if self.comm.rank == 0:
			if self.debug : print('Reading wave-function coefficients from WAVECAR.', flush = True)
		self.wavecoef()
		"""
		else:
			#self.wavecoef()
			pass
		"""

		if self.comm.rank == 0:
			if self.debug : print('Reading charge perturbation from CHGCAR difference.', flush = True)
			self.CHGCAR = VaspChargeDensity("CHGCAR_diff")
			self.charge = self.CHGCAR.chg[0]
		else:
			self.charge = None
		#scatter CHGCAR
		self.charge = self.comm.bcast(self.charge, root = 0)

		# number of points in CHGCAR gives points for integral in RS
		self.r = np.array(self.charge.shape, dtype='int_')

		# real space calculation reduction, calculate only every scale-th point
		self.scale = 1

		# calculate divergence of scattering potential
		#self.divcharge = self.div()

		if self.comm.rank == 0:
			if self.debug : print('Reading simulation configuration and group velocities from vasprun.xml', flush = True)
			calc = VaspKpointsInterpolated("vasprun.xml")
		else:
			calc = None
		calc = self.comm.bcast(calc, root = 0)

		self.cell = calc.basis

		# parameters for K-points in IBZ (formally interpolated, but with scaling factor 1)
		self.inkpt = calc.inkpts
		self.ikpts = calc.ikpts
		self.iene = calc.ienergies
		self.ivel = calc.ivelocities
		self.iocc = calc.ipopulations
		self.kgrid = calc.kptgrid_divisions

		self.nbands = calc.inbands
		self.sigma = calc.sigma
		self.fermi = calc.efermi_interpolated
		self.nelect = calc.nelect

		# parameters for K-points in full BZ (formally interpolated, but with scaling factor 1)
		self.nkpt = calc.nkpts
		self.kpts = calc.kpts
		self.ene = calc.energies
		self.vel = calc.velocities
		#print(self.vel.shape)

		if self.comm.rank == 0: 
			if self.debug:
				print("Cell vectors:")
				print(self.cell)
				print("Number of k-points in full IBZ interploated:",self.inkpt)
				print("Number of k-points in full BZ interpolated:",self.nkpt)


		if self.comm.rank == 0:
			# mapping for k-points from IBZ to FBZ
			i2f = ibz2fbz("OUTCAR")
		else:
			i2f = None

		i2f = self.comm.bcast(i2f, root = 0)

		self.nibz2fbz = i2f.nitpi2f
		self.ibz2fbz = i2f.itpi2f

		if self.comm.rank == 0:
			if self.debug:
				print(self.nibz2fbz.shape[0]," noniterpolated IBZ - FBZ k-point mappings", flush = True)
				print(self.ibz2fbz.shape[0]," iterpolated IBZ - FBZ k-point mappings", flush = True)

		# for cubic cell
		self.dr = np.array([0,0,0],dtype='float64')
		for i in range(3): self.dr[i] = (self.cell[i][i]) / self.r[i] # in Ang

		if self.debug:
			if self.comm.rank == 0:
				print('dr',self.dr, flush = True)

		# step of K-mesh in reciprocal space
		self.dk = np.array([0,0,0],dtype='float')
		for i in range(3): self.dk[i] = 2.0 * pi  / self.kgrid[i]  # unitless
		if self.debug:
			if self.comm.rank == 0: print('dk',self.dk, flush = True)

		if self.comm.rank == 0:
			# cached value of scattering probability matrix (squared elements)
			self.T2 = sps.dok_matrix((self.inkpt*self.nbands,self.inkpt*self.nbands), dtype=np.float64)
			if (self.restart):
				data = sps.load_npz("./data_sparse.npz")
				self.T2 = data.todok()
		else:
			self.T2 = None
		self.T2 = self.comm.bcast(self.T2, root = 0)

	def wavecoef(self):
		#   constant 'c' below is 2m/hbar**2 in units of 1/eV Ang^2 (value is
		#   adjusted in final decimal places to agree with VASP value; program
		#   checks for discrepancy of any results between this and VASP values)
		c = 0.262465831
		# c/0.26246582250210965422d0/.

		
		recl = 24 # Default record length in bytes
		filename = 'WAVECAR'
		f = open(filename,'rb')
		buffer = f.read(recl)
		# Read the file format: record length, spins and precision
		fmt = '3d'
		(recl_,spin_,prec_) = struct.unpack(fmt,buffer)
		recl = int(recl_)
		nspin = int(spin_)
		prec = int(prec_)
		#print(recl,spin,prec)
		f.close()

		if prec == 45210 :
			sys.exit('*** error - WAVECAR_double requires complex*16')

		a1 = np.empty([3],dtype='d')
		a2 = np.empty([3],dtype='d')
		a3 = np.empty([3],dtype='d')

		# Reopen file with new record length
		f = open(filename,'rb')
		buffer = f.read(recl) # skip the first entrance
		buffer = f.read(recl) # read number of K-points, bands, energy cut-off and cell dimensions
		fmt='12d'
		(nkpt_,nband_,ecut_,a1[0],a1[1],a1[2],a2[0],a2[1],a2[2],a3[0],a3[1],a3[2]) = struct.unpack(fmt,buffer[:96])

		nkpt = int(nkpt_)
		nband = int(nband_)
		ecut = float(ecut_)

		#if self.debug: 
		if self.comm.rank==0:
			print('Nuber of K-points',nkpt)
			print('Number of energy bands',nband)
			print('Energy cut-off',ecut)
			print('Lattice vectors:')
			print(a1)
			print(a2)
			print(a3)

		#   compute reciprocal propertieS
		Vcell = a1.dot(np.cross(a2,a3))
		
		b1 = np.cross(a2,a3)
		b2 = np.cross(a3,a1)
		b3 = np.cross(a1,a2)

		b1 *= 2.*pi/Vcell
		b2 *= 2.*pi/Vcell
		b3 *= 2.*pi/Vcell

		b1mag = norm(b1)
		b2mag = norm(b2)
		b3mag = norm(b3)

		#if self.debug: 
		if self.comm.rank==0:
			print('Volume unit cell =',Vcell)
			print('Reciprocal lattice vectors:')
			print(b1)
			print(b2)
			print(b3)
			print('Reciprocal lattice vector magnitudes:')
			print(b1mag, b2mag, b3mag)

		phi12 = acos(b1.dot(b2)/(b1mag*b2mag))
		vtmp = np.cross(b1,b2)
		vmag = norm(vtmp)
		sinphi123=b3.dot(vtmp)/(vmag*b3mag)

		nb1maxA=int(sqrt(ecut*c)/(b1mag*abs(sin(phi12))))+1
		nb2maxA=int(sqrt(ecut*c)/(b2mag*abs(sin(phi12))))+1
		nb3maxA=int(sqrt(ecut*c)/(b3mag*abs(sinphi123)))+1
		npmaxA=int(4.*pi*nb1maxA*nb2maxA*nb3maxA/3.)

		phi13 = acos(b1.dot(b3)/(b1mag*b3mag))
		vtmp = np.cross(b1,b3)
		vmag = norm(vtmp)
		sinphi123=b2.dot(vtmp)/(vmag*b2mag)
		phi123=abs(asin(sinphi123))
		nb1maxB=int(sqrt(ecut*c)/(b1mag*abs(sin(phi13))))+1
		nb2maxB=int(sqrt(ecut*c)/(b2mag*abs(sinphi123)))+1
		nb3maxB=int(sqrt(ecut*c)/(b3mag*abs(sin(phi13))))+1
		npmaxB=int(4.*pi*nb1maxB*nb2maxB*nb3maxB/3.)

		phi23 = acos(b2.dot(b3)/(b2mag*b3mag))
		vtmp = np.cross(b2,b3)
		vmag = norm(vtmp)
		sinphi123=b1.dot(vtmp)/(vmag*b1mag)
		phi123=abs(asin(sinphi123))
		nb1maxC=int(sqrt(ecut*c)/(b1mag*abs(sinphi123)))+1
		nb2maxC=int(sqrt(ecut*c)/(b2mag*abs(sin(phi23))))+1
		nb3maxC=int(sqrt(ecut*c)/(b3mag*abs(sin(phi23))))+1
		npmaxC=int(4.*pi*nb1maxC*nb2maxC*nb3maxC/3.)

		nb1max=max(nb1maxA,nb1maxB,nb1maxC)
		nb2max=max(nb2maxA,nb2maxB,nb2maxC)
		nb3max=max(nb3maxA,nb3maxB,nb3maxC)
		npmax=min(npmaxA,npmaxB,npmaxC)

		#if self.debug: 
		if self.comm.rank==0 :
			print('max. no. G values; 1,2,3 =',nb1max,nb2max,nb3max)
			print('estimated max. no. plane waves =',npmax, flush = True)

		# assign memory
		wf_rank = Lifetime.wf(nspin,nkpt,npmax,nband)
		wf_rank.Vcell = Vcell

		# Begin loops over spin, k-points and bands
		for spin in range(nspin):
			#if self.debug : 
			if self.comm.rank==0 :
				print()
				print('********')
				print('reading spin ',spin, flush = True)

			for ik in range(self.comm.rank, nkpt, self.comm.size):
				#for ik in range(nkpt):
				# search and read
				recpos = (2 + ik*(nband+1) + spin*nkpt*(nband+1)) * recl
				f.seek( recpos )
				buffer = f.read(recl)
				dummy = np.empty([int(recl/8)],dtype='d')
				fmt=str(int(recl/8))+'d'
				(dummy) = struct.unpack(fmt,buffer)
				wf_rank.nplane[spin][ik]=int(dummy[0])
				wf_rank.kpt[spin][ik] = np.array(dummy[1:4])
				for i in range(nband):
					wf_rank.cener[spin][ik][i] = dummy[5+2*i] + 1j * dummy[5+2*i+1]
					wf_rank.occ[spin][ik][i] = dummy[5+2*nband+i]

				if self.debug:
					# and self.comm.rank==0):
					print('k point #',ik,'  input no. of plane waves =', wf_rank.nplane[spin][ik], 'k value =',wf_rank.kpt[spin][ik], flush=True)

				# Calculate available plane waves
				ncnt = 0
				for ig3 in range(0, 2*nb3max+1):
					ig3p = ig3
					if ig3 > nb3max: ig3p = ig3 - 2*nb3max - 1

					for ig2 in range(0, 2*nb2max+1):
						ig2p = ig2
						if ig2 > nb2max: ig2p = ig2 - 2*nb2max - 1
						for ig1 in range(0, 2*nb1max+1):
							ig1p = ig1
							if ig1 > nb1max: ig1p = ig1 - 2*nb1max - 1

							sumkg = np.empty([3],dtype='d')
							for j in range(3):
								sumkg[j] = (wf_rank.kpt[spin][ik][0] + ig1p) * b1[j] \
										 + (wf_rank.kpt[spin][ik][1] + ig2p) * b2[j] \
										 + (wf_rank.kpt[spin][ik][2] + ig3p) * b3[j]

							gtot = norm(sumkg)
							etot = gtot*gtot/c
							
							if etot < ecut :
								wf_rank.igall[spin][ik][ncnt][0] = ig1p
								wf_rank.igall[spin][ik][ncnt][1] = ig2p
								wf_rank.igall[spin][ik][ncnt][2] = ig3p
								ncnt += 1

				if ncnt > npmax:
					sys.exit('*** error - plane wave count exceeds estimate')
				if ncnt != wf_rank.nplane[spin][ik] :
					sys.exit('*** error - computed no. '+str(ncnt)+' != input no.'+str(wf_rank.nplane[spin][ik]))

				# read planewave coefficients for each energy band at given K-point
				for iband in range(wf_rank.nband):
					buffer = f.read(recl)
					fmt=str(int(8*wf_rank.nplane[spin][ik]/4))+'f'
					(dummy) = struct.unpack(fmt,buffer[:int(8*wf_rank.nplane[spin][ik])])
					for iplane in range(wf_rank.nplane[spin][ik]):
						wf_rank.coeff[spin][ik][iband][iplane] = dummy[2*iplane] + 1j * dummy[2*iplane+1]

			#if self.comm.rank == 0:
			self.wf = Lifetime.wf(nspin,nkpt,npmax,nband)
			self.wf.Vcell = Vcell
			
			#else:
			#	self.wf = None
			"""
			self.comm.Reduce(wf_rank.occ,    self.wf.occ,    op=MPI.SUM, root = 0)
			self.comm.Reduce(wf_rank.cener,  self.wf.cener,  op=MPI.SUM, root = 0)
			self.comm.Reduce(wf_rank.igall,  self.wf.igall,  op=MPI.SUM, root = 0)
			self.comm.Reduce(wf_rank.coeff,  self.wf.coeff,  op=MPI.SUM, root = 0)
			self.comm.Reduce(wf_rank.kpt,    self.wf.kpt,    op=MPI.SUM, root = 0)
			self.comm.Reduce(wf_rank.nplane, self.wf.nplane, op=MPI.SUM, root = 0)
			"""

			self.comm.Allreduce(wf_rank.occ,    self.wf.occ,    op=MPI.SUM)
			self.comm.Allreduce(wf_rank.cener,  self.wf.cener,  op=MPI.SUM)
			self.comm.Allreduce(wf_rank.igall,  self.wf.igall,  op=MPI.SUM)
			self.comm.Allreduce(wf_rank.coeff,  self.wf.coeff,  op=MPI.SUM)
			self.comm.Allreduce(wf_rank.kpt,    self.wf.kpt,    op=MPI.SUM)
			self.comm.Allreduce(wf_rank.nplane, self.wf.nplane, op=MPI.SUM)

		f.close()

		return

	def T(self, ki, ni, kf, nf):
		"Calculate scattering matrix element, can be compex. ki - initial K-point, ni - initial energy band"

		T = 0.0
		Trank = 0.0
		Ti = 0.0
		Tf = 0.0
		s = self.scale #scale of the charge array

		rs = np.array([int(self.r[0]/s),int(self.r[1]/s),int(self.r[2]/s)],dtype='int_') #number of points in space

		phi_i = np.zeros((rs[0],rs[1],rs[2]),dtype='complex128')
		phi_f = np.zeros((rs[0],rs[1],rs[2]),dtype='complex128')

		spin = 0 #non spin-polarized

		# >0 i.e. on all nodes
		if self.comm.rank == 0:
			kpt = self.wf.kpt[spin][ki]
			igall = self.wf.igall[spin][ki]
			nplane = self.wf.nplane[spin][ki]
			coeff = self.wf.coeff[spin][ki][ni]
			Vcell = self.wf.Vcell

		else:
			kpt = None
			igall = None
			nplane = None
			coeff = None
			Vcell = None

		kpt = self.comm.bcast(kpt, root = 0)
		igall = self.comm.bcast(igall, root = 0)
		nplane = self.comm.bcast(nplane, root = 0)
		coeff = self.comm.bcast(coeff, root = 0)
		Vcell = self.comm.bcast(Vcell, root = 0)

		phi.phi_skn(kpt, igall, nplane, coeff, Vcell, rs, phi_i)

		# >0 i.e. on all nodes
		if self.comm.rank == 0:
			kpt = self.wf.kpt[spin][kf]
			igall = self.wf.igall[spin][kf]
			nplane = self.wf.nplane[spin][kf]
			coeff = self.wf.coeff[spin][kf][nf]
			Vcell = self.wf.Vcell

		else:
			kpt = None
			igall = None
			nplane = None
			coeff = None
			Vcell = None

		kpt = self.comm.bcast(kpt, root = 0)
		igall = self.comm.bcast(igall, root = 0)
		nplane = self.comm.bcast(nplane, root = 0)
		coeff = self.comm.bcast(coeff, root = 0)
		Vcell = self.comm.bcast(Vcell, root = 0)

		phi.phi_skn(kpt, igall, nplane, coeff, Vcell, rs, phi_f)

		# x y z - indeces of points in the charge array
		for idx in range(self.comm.rank, int(rs[0]*rs[1]*rs[2]), self.comm.size):
			# convert common index to dimention indexes
			z = int(  idx / (rs[1]*rs[2])           )
			y = int( (idx - z *rs[1]*rs[2]) / rs[0] )
			x = int( (idx - z *rs[1]*rs[2]) % rs[0] )
			Trank += np.conj( phi_f[x][y][z] ) * self.charge[x*s][y*s][z*s] * phi_i[x][y][z]

		T = self.comm.allreduce(Trank,op=MPI.SUM)

		T  *= self.dr[0] * self.dr[1] * self.dr[2] * pow(s,3) 
		#Ti *= self.dr[0] * self.dr[1] * self.dr[2] * pow(s,3)
		#Tf *= self.dr[0] * self.dr[1] * self.dr[2] * pow(s,3)
		
		if self.comm.rank==0:
			#print('\t<{},{}|V|{},{}> = {:f} <i|i> = {:f} <f|f> = {:f}'.format(kf,nf,ki,ni,abs(T),abs(Ti),abs(Tf)))
			print('\t<{},{}|V|{},{}> = {:e} '.format(kf,nf,ki,ni,abs(T)))
		return T


	def DDelta(self, x):
		"Dirac delta function in a form of Haussian"
		# normalization for (2pi)^3 volume
		# sigma =  pow(2.0 * pi, 2.5)
		#sigma = self.nkpt / pow(2.0 * pi,0.5)
		return 1.0 #/(pow(2.0 * pi,0.5) * sigma) * np.exp(-x*x/(2*sigma*sigma))


	def dFde(self,k,n):
		# derivative of Fermi distribution
		sigma = self.sigma
		x = self.ene[k][n] - self.fermi
		return -1.0/(pow(2*pi,0.5) * sigma) * np.exp(-x*x/(2*sigma*sigma))


	def R(self,kf,nf):
		"Get inverse lifetime for state n, k"
		# defect dencity per cubic Angstr
		nd = 1.0 # per atom, actuall will be 1e-20 or so

		hbar = 4.135667662e-15 # eV*s

		# initial value for scattering rate for kf nf
		R = 0.0
		dk = self.dk # unitless 0..2pi

		# loop over all initial energy levels
		for ni in range(self.nbands):

			# scatering coefficient for n-th k-point
			R_n = 0.0

			# loop over initial k-points in IBZ
			for iki in range(self.inkpt):


				
				ikf = kf # we already supply only indeces from IBZ

				# 3 find if we in proximity of Ef
				#if (self.occ[iki][ni] == 0.0 or self.occ[iki][ni]==2.0): continue
				if (self.iocc[iki][ni] == 0.0 or self.iocc[iki][ni] == 1.0) : continue

				# 7 get eigenstates
				ei = self.ene[iki][ni]
				ef = self.ene[ikf][nf] 
				
				if abs(ei - ef) > self.sigma*0.1: continue

				# 6 check if we have cached value for calculated T element
				init = iki*self.nbands + ni
				final = ikf*self.nbands + nf
				if not self.T2[init,final]:
					t0 = time.time()
					self.T2[init,final] = self.T2[final,init] = pow(abs( self.T(iki,ni,ikf,nf) ),2.0)
					t1 = time.time()
					if self.comm.rank==0:
						print('\tT(', iki, ni, '=>', ikf, nf, ') =', sqrt(self.T2[init,final])*1000.0,'meV, R =', 2.0*pi/hbar*self.T2[init,final]*self.DDelta(ef - ei), 'eV/s' ,int((t1-t0)*100.0)/100.0,'s', flush=True)
				else:
					if self.comm.rank==0:
						print('\tT(', iki, ni, '=>', ikf, nf, ') =',sqrt(self.T2[init,final])*1000.0,'meV REUSE', flush=True)
				T2 = self.T2[init,final]

				# sum over all reflections of reduced K-point
				kpts = np.where(self.ibz2fbz == iki)[0]
				if self.comm.rank==0: print("\tAdding",kpts.shape[0],"kpt reflections")
				for ki in kpts:

					# 4 get FBZ - interpolated FBZ number correspondence for group velocity
					a = self.vel[ki][ni]
					b = self.vel[kf][nf]
					an = LA.norm(a)
					bn = LA.norm(b)

					# 5 check for zero group velocity
					if (not an or not bn): continue
					costheta = np.dot(a,b)/(an * bn)
					if (costheta == 1.0): continue

					# 8 sum integral over bands
					if self.comm.rank == 0:
						R_n += T2 * self.DDelta(ef - ei) * (1.0 - costheta) # (eV)^2

			# sum over K-points
			R += R_n
		
		#print("Preaparing to exit R",self.comm.rank,nf,kf)
		if self.comm.rank == 0:
			R = nd  * (2.0 * pi/hbar * R ) * (dk[0] * dk[1] * dk[2] / pow(2.0*pi,3.0)) # m-3 / eVs * (eV)^2 = eV/m3s
			print( 'R(k=', kf, 'n=', nf, ') = ', R, 'eV/s', flush = True)
		else:
			pass
		
		#print("Now try to exit R",self.comm.rank,nf,kf, flush = True)

		return R

	def mobility(self):
		"Carrier mobility calculation, sum over all bands of integrals over all K-points"
		# step of K-mesh in reciprocal space, kx=ky=kz number of k points in each direction
		dk = self.dk # unitless, 0..2pi

		# carrier concentration per cubic cm, +3 electrons of defected Al atom
		# per cubic cm
		ncarr = (self.nelect + 3) #/ LA.det(self.cell * 1e-10)
		# ncarr = 18.1e22 # el per cubic cm
		#ncarr = 3 # per atom

		# electron charge, since we work in eV units
		#e = -1.6e-19 # coulomb
		e = 1.0 # in electrons
		mob = 0.0

		for nf in range(self.nbands):
			# loop over all k-points
			for ikf in range(self.inkpt):

				
				# 2 get FBZ - IBZ number correspondance
				#ikf = self.ibz2fbz[kf]

				# 3 find if we in proximity of Ef
				if (self.iocc[ikf][nf] == 0.0 or self.iocc[ikf][nf]==1.0): continue

				vel = self.vel[ikf][nf]
				# check for [0,0,0] velocity
				if(not LA.norm(vel)): continue

				R_nk = self.R(ikf,nf)

				if not R_nk: continue
				else: tau = 1.0 / R_nk

				# sum over all reflections of reduced K-point
				for kf in np.where(self.ibz2fbz == ikf)[0]:
					vel = self.vel[kf][nf]
					# velocity units A/fs = 1e-10 m / 1e-15 s = 1e5 m/s
					# projection of group velocity on field direction
					proj1 = np.dot(vel,[1,0,0]) * 1e5
					# projection of group velocity on current direction
					proj2 = np.dot(vel,[1,0,0]) * 1e5

					if self.comm.rank == 0:
						mob += tau * self.dFde(kf,nf) * np.dot(proj1,proj2) #  s/eV * (m/s)^2 = m2/eVs

		
		if self.comm.rank == 0:
			return (-2.0 * e / ncarr) * mob * (dk[0]*dk[1]*dk[2] / pow(2.0 * pi, 3.0)) # e  * (m2/eVs) =  m2/Vs

		return 0

def main(nf = 0, kf = 0):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	t0 = time.time()
	debug = True
	restart = False
	lt = Lifetime(debug,restart)
	t1 = time.time()

	if len(sys.argv) > 1: lt.scale =  int(sys.argv[1])

	if lt.comm.rank == 0:
		if lt.debug: print("Data readed in",int(t1-t0),'s.')

	mob = lt.mobility()

	if lt.comm.rank == 0:
		print("Mobility:",mob*1e-4,' cm2/Vs')
		t2 = time.time()
		print("Real space reduction ",lt.scale,'- fold')
		if lt.debug: print("Data calculation",int(t2-t1),'s, total time',int(t2-t0)," Ncores:",lt.comm.size, flush = True)

	return 0

if __name__ == "__main__": main()
