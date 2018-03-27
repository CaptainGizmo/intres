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
#from sympy import DiracDelta
from math import sin,cos,asin,acos,sqrt

#import mpi4py
#mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import phi

from getsize import get_size
from VaspKpointsInterpolated import *
from ibz2fbz import *

class Lifetime(object):

	class wf():
		def __init__(self,nspin,nkpt,npmax,nband):
			# assign memory
			self.ids = np.zeros([nspin,nkpt], dtype = 'int_')
			self.occ   = np.zeros([nspin,nkpt,nband], dtype = 'float64')
			self.cener = np.zeros([nspin,nkpt,nband], dtype = 'complex128')
			self.igall = np.zeros([nspin,nkpt,npmax,3],dtype='int_')
			self.coeff = np.zeros([nspin,nkpt,nband,npmax],dtype='complex64')
			self.kpt   = np.zeros([nspin,nkpt,3],dtype='float64')
			self.nplane = np.zeros([nspin,nkpt],dtype='int_')
			self.Vcell = 0
			self.nband = nband

	def __init__(self, debug = True, restart = True, scale = 1):
		self.comm = MPI.COMM_WORLD
		self.debug = debug
		self.restart = restart
		# real space calculation reduction, calculate only every scale-th point
		self.scale = scale
		self.T2file = "data_sparse."+str(self.scale)+".npz"

		#default distance between erergy of points in parts of fermi smearing
		self.ds = 1


		# electron charge, since we work in eV units
		#e = -1.6e-19 # coulomb
		self.e = 1.0 # in electrons

		# carriers per atomic site
		self.ncarr =  3. #Aluminum

		##########################################################################################################################
		if self.comm.rank == 0:
			if self.debug : print('* Reading charge perturbation from CHGCAR difference.', flush = True)
			self.CHGCAR = VaspChargeDensity("CHGCAR_diff")
			self.charge = self.CHGCAR.chg[0]
		else:
			self.charge = None
		#scatter CHGCAR
		self.charge = self.comm.bcast(self.charge, root = 0)

		# number of points in CHGCAR gives points for integral in RS
		self.r = np.array(self.charge.shape, dtype='int_')

		# calculate divergence of scattering potential
		#self.divcharge = self.div()
		
		##########################################################################################################################

		if self.comm.rank == 0:
			if self.debug : print('\n* Reading simulation configuration and group velocities from vasprun.xml', flush = True)
			calc = VaspKpointsInterpolated("vasprun.xml")
		else:
			calc = None
		calc = self.comm.bcast(calc, root = 0)

		self.cell = calc.basis
		self.occ = calc.populations
		self.kgrid = calc.kptgrid_divisions
		self.nbands = calc.nbands
		self.sigma = calc.sigma
		self.fermi = calc.efermi_interpolated
		self.nelect = calc.nelect
		self.natoms = calc.natoms

		# defect dencity per atomic site
		self.nd = 1./(self.natoms + 1.) # 1 per cell, +1 since vacancy


		# parameters for K-points in full BZ (formally interpolated, but with scaling factor 1)
		self.nkpt = calc.nkpts
		self.kpts = calc.kpts
		self.ene = calc.energies
		self.vel = calc.velocities




		# for ortho cell
		self.dr = np.array([0,0,0],dtype='float64')
		for i in range(3): self.dr[i] = (self.cell[i][i]) / self.r[i] # in Ang, r - number points of LOCPOT grid

		# step of K-mesh in reciprocal space
		self.dk = np.array([0,0,0],dtype='float')
		for i in range(3): self.dk[i] = LA.norm(calc.rec_basis[i]) / self.kgrid[i]  # unitless
		#for i in range(3): self.dk[i] = 2 * pi / (self.kgrid[i] + 0.0) # unitless!!!!

		# search for bands which are in principle crossing the FS
		self.bandlist=[]
		for band in range(self.nbands):
			emax=np.max(self.ene[:,band])
			emin=np.min(self.ene[:,band])
			if self.fermi >= emin and self.fermi <= emax :
				self.bandlist.append(band)
		self.bandlist = np.array(self.bandlist)

		if self.comm.rank == 0:
			if self.debug:
				print("Cell vectors:")
				print(self.cell)
				print
				print("Atoms: ",self.natoms)
				print
				print('dr',self.dr, flush = True)
				print("\nReciprocal vectors:")
				print(calc.rec_basis)
				#print('dk',self.dk, flush = True)
				print("Found",self.bandlist.shape[0],"bands crossing FS")
				print(self.bandlist)
				print


		##########################################################################################################################
		if self.comm.rank == 0:
			# mapping for k-points from IBZ to FBZ
			print("\n* Reading k-points mapping from OUTCAR", flush = True)
			i2f = ibz2fbz("OUTCAR")
		else:
			i2f = None
		#scatter i2f
		i2f = self.comm.bcast(i2f, root = 0)

		self.nibz2fbz = i2f.nitpi2f
		self.ibz2fbz = i2f.itpi2f

		#find IBZ Kpts:
		self.ikpts=[]
		for kpt in range(self.nkpt):
			# check that Kpt is from IBZ:
			if kpt == self.ibz2fbz[kpt]:
				self.ikpts.append(kpt)
		self.ikpts = np.array(self.ikpts)
		self.inkpt = self.ikpts.shape[0]

		if self.comm.rank == 0:
			if self.debug:
				print("Number of k-points in full IBZ interploated:",self.inkpt)
				print("Number of k-points in full BZ interpolated:",self.nkpt)
				print("\nSorting out k-points with partially occupied band levels:")

		# sorting out kpts we don't need
		# require fix for spin decoupling
		self.kptlist = []
		for kpt in range(self.inkpt):
			flag = 0
			for n in self.bandlist:
				#if (self.occ[kpt][n] == 0.0 or self.occ[kpt][n] == 1.0) : continue
				if (self.occ[kpt][n] <= 0.01 or self.occ[kpt][n] >= 0.99) : continue
				flag = 1
			if flag:
				if self.comm.rank == 0:
					print('Adding k point #',kpt, flush=True)
				self.kptlist.append(kpt)
		self.kptlist=np.array(self.kptlist)
		
		if self.comm.rank == 0:
			if self.debug:
				print("Added",len(self.kptlist),"points from",self.inkpt)



		##########################################################################################################################
		if self.comm.rank == 0:
			# cached value of scattering probability matrix (squared elements)
			self.T2 = sps.dok_matrix((self.inkpt*self.nbands,self.inkpt*self.nbands), dtype=np.float64)
			#if (self.restart):
			if os.path.isfile(self.T2file) :
				if self.comm.rank == 0: print("\n* Reading scattering coefficients from",self.T2file,flush = True)
				data = sps.load_npz(self.T2file)
				self.T2 = data.todok()
				if self.comm.rank == 0: print("scattering coefficients readed.",flush = True)
		else:
			self.T2 = None
		self.T2 = self.comm.bcast(self.T2, root = 0)

		##########################################################################################################################
		#reading wavefunction
		if self.comm.rank == 0:
			if self.debug :
				print('\n* Reading wave-function coefficients from WAVECAR.', flush = True)
				print
		self.wavecoef()
		
		##########################################################################################################################

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
			print('Nuber of K-points',nkpt,'reading',len(self.kptlist),'of them')
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

		nkpt = self.kptlist.shape[0]
		cband = self.bandlist.shape[0] # bands Crossing FS

		################### create the output structure ########################################
		
		if self.comm.rank == 0:
			self.wf = Lifetime.wf(nspin,nkpt,npmax,cband)
			self.wf.Vcell = Vcell

		################## read structures: spin, kpt on each thread ###########################
		################## read coeffs parallelized by band          ###########################

		# Begin loops over spin, k-points and bands
		for spin in range(nspin):
			#if self.debug : 
			if self.comm.rank==0 :
				print()
				print('********')
				print('reading spin ',spin, flush = True)

			# loop over selected Kpts
			for ik in range(nkpt):
				# get real number of Kpt
				ikid = self.kptlist[ik]
				if self.comm.rank == 0:
					self.wf.ids[spin][ik] = ikid
				# search and read information about the band
				recpos = (2 + ikid*(nband+1) + spin*nkpt*(nband+1)) * recl
				f.seek(recpos)
				buffer = f.read(recl)
				dummy = np.empty([int(recl/8)],dtype='d')
				fmt=str(int(recl/8))+'d'
				(dummy) = struct.unpack(fmt,buffer)
				if self.comm.rank == 0:
					self.wf.nplane[spin][ik]=int(dummy[0])
					self.wf.kpt[spin][ik] = np.array(dummy[1:4])

				for i in range(cband):
					if self.comm.rank == 0:
						iband = self.bandlist[i]
						self.wf.cener[spin][ik][i] = dummy[5+2*iband] + 1j * dummy[5+2*iband+1]
						self.wf.occ[spin][ik][i] = dummy[5+2*nband+iband]

				if self.debug and self.comm.rank==0 :
					print('k point #',ikid,'  input no. of plane waves =', self.wf.nplane[spin][ik], 'k value =', self.wf.kpt[spin][ik], flush=True)

				# Calculate available plane waves for selected Kpt
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

							if self.comm.rank == 0:
								sumkg = np.zeros([3],dtype='d')
								for j in range(3):
									sumkg[j] = (self.wf.kpt[spin][ik][0] + ig1p) * b1[j] \
										 + (self.wf.kpt[spin][ik][1] + ig2p) * b2[j] \
										 + (self.wf.kpt[spin][ik][2] + ig3p) * b3[j]

								gtot = norm(sumkg)
								etot = gtot*gtot/c
							
								if etot < ecut :
									self.wf.igall[spin][ik][ncnt][0] = ig1p
									self.wf.igall[spin][ik][ncnt][1] = ig2p
									self.wf.igall[spin][ik][ncnt][2] = ig3p
									ncnt += 1

				if self.comm.rank == 0:
					if ncnt > npmax:
						sys.exit('*** error - plane wave count exceeds estimate')
					if ncnt != self.wf.nplane[spin][ik] :
						sys.exit('*** error - computed no. '+str(ncnt)+' != input no.'+str(self.wf.nplane[spin][ik]))

				######### read planewave coefficients for each energy band at given K-point ####################
				rank_k_coeff = np.zeros([cband,npmax],dtype='complex64')

				if self.comm.rank == 0:
					npl = self.wf.nplane[spin][ik]
				else:
					npl = None
				npl = self.comm.bcast(npl,root=0)

				# read coeffs in parallel
				
				#for iband in range(self.comm.rank, nband, self.comm.size):
				for i in range(self.comm.rank, cband, self.comm.size):
					iband = self.bandlist[i]
					recpos = (2 + ikid*(nband+1) + spin*nkpt*(nband+1) + (iband+1) ) * recl
					f.seek(recpos)
					buffer = f.read(recl)
					fmt=str(int(8*npl/4))+'f'
					(dummy) = struct.unpack(fmt,buffer[:int(8*npl)])
					for iplane in range(npl):
						rank_k_coeff[i][iplane] = dummy[2*iplane] + 1j * dummy[2*iplane+1]
				
				# collect coeffs from slave nodes to the root node
				if self.comm.rank == 0:
					k_coeff = np.zeros([cband,npmax],dtype='complex64')
				else:
					k_coeff = None
				self.comm.Reduce(rank_k_coeff,  k_coeff,  op=MPI.SUM, root = 0)

				# save coeffs for the chosen Kpt
				if self.comm.rank == 0:
					self.wf.coeff[spin][ik] = k_coeff

		f.close()
		if self.comm.rank == 0:  print('Reading wavefunction coefficients is done.', flush = True)

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

		#find array number form band number
		bi = np.where(self.bandlist == ni)[0]
		bf = np.where(self.bandlist == nf)[0]
		
		if self.comm.rank == 0:
			idki = np.where(self.wf.ids[spin] == ki)[0][0]
			kpt = self.wf.kpt[spin][idki]
			igall = self.wf.igall[spin][idki]
			nplane = self.wf.nplane[spin][idki]
			coeff = np.asarray(self.wf.coeff[spin][idki][bi],dtype=np.complex128)
			Vcell = self.wf.Vcell

		else:
			idki = None
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

		if self.comm.rank == 0:
			idkf = np.where(self.wf.ids[spin] == kf)[0][0]
			kpt = self.wf.kpt[spin][idkf]
			igall = self.wf.igall[spin][idkf]
			nplane = self.wf.nplane[spin][idkf]
			coeff = np.asarray(self.wf.coeff[spin][idkf][bf],dtype=np.complex128)
			Vcell = self.wf.Vcell

		else:
			idkf = None
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
		
		if self.comm.rank==0:
			print('\t<{},{}|V|{},{}> = {:e} '.format(kf,nf,ki,ni,abs(T)))
		return T


	def DDelta(self, x):
		"Dirac delta function in a form of Haussian"
		# normalization for (2pi)^3 volume
		# sigma =  pow(2.0 * pi, 2.5)
		#sigma = self.nkpt / pow(2.0 * pi,0.5)
		return 1.0 #/(pow(2.0 * pi,0.5) * sigma) * np.exp(-x*x/(2*sigma*sigma))


	def dFdE(self,k,n):
		# derivative of Fermi distribution
		sigma = self.sigma
		de = self.ene[k][n] - self.fermi
		#return -1.0/(pow(2*pi,0.5) * sigma) * np.exp(-de*de/(2*sigma*sigma))
		return -1.0/sigma * np.exp(de/sigma) / (np.exp(de/sigma) + 1.0)**2.0

	def saveT2(self):
		#save new T
		if self.comm.rank == 0:
			#mv old file
			if os.path.isfile(self.T2file) :
				os.rename(self.T2file, self.T2file+".0")
			
			# convert matrix to save format
			data_save = sps.csc_matrix(self.T2)
			
			#save matrix
			sps.save_npz(self.T2file, data_save)
			#pass



	def R(self,kf,nf):
		"Get inverse lifetime for state n, k"
		ikf = kf # we already supply only indeces from IBZ

		hbar = 4.135667662e-15 # eV*s

		# initial value for scattering rate for kf nf
		R = 0.0
		#dk = self.dk # unitless 0..2pi

		# loop over all initial energy levels
		for ni in self.bandlist:

			# scatering coefficient for n-th k-point
			R_n = 0.0

			# loop over initial k-points in IBZ
			for iki in self.kptlist:

				# 3 find if we in proximity of Ef
				if (self.occ[iki][ni] == 0.0 or self.occ[iki][ni] == 1.0) : continue

				# 7 get eigenstates
				ei = self.ene[iki][ni]
				ef = self.ene[ikf][nf] 
				
				if abs(ei - ef) > self.sigma*self.ds: 
					continue

				# 6 check if we have cached value for calculated T element
				init = iki*self.nbands + ni
				final = ikf*self.nbands + nf
				
				# plug
				#self.T2[init,final] = 0.1
				
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
			#R = self.nd  * (2.0 * pi/hbar * R ) * (dk[0] * dk[1] * dk[2] / pow(2.0*pi,3.0)) # 1 * (1 / eVs) * (eV)^2 * 1 != eV/s
			R = self.nd  * (2.0 * pi/hbar * R ) # 1 * (1 / eVs) * (eV)^2 * 1 != eV/s
			print( 'R(k=', kf, 'n=', nf, ') = ', R, 'eV/s', flush = True)
			# save updatet T matrix here ##################################################################################################
			self.saveT2()
		else:
			pass

		return R

	def mobility(self):
		"Carrier mobility calculation, sum over all bands of integrals over all K-points"
		# step of K-mesh in reciprocal space, kx=ky=kz number of k points in each direction
		#dk = self.dk # unitless, 0..2pi/N

		mob = 0.0

		for nf in self.bandlist:
			# loop over all k-points
			for ikf in self.kptlist:

				# 2 get FBZ - IBZ number correspondance
				#ikf = self.ibz2fbz[kf]

				# 3 find if we in proximity of Ef
				if (self.occ[ikf][nf] == 0.0 or self.occ[ikf][nf]==1.0): continue

				vel = self.vel[ikf][nf]
				# check for [0,0,0] velocity
				if(not LA.norm(vel)): continue

				R_nk = self.R(ikf,nf)
				#print("ene",self.ene[ikf][nf])

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
						mob += tau * self.dFdE(kf,nf) * np.dot(proj1,proj2) #  s/eV * (m/s)^2 = m2/eVs

		
		if self.comm.rank == 0:
			#return (-2.0 * self.e / self.ncarr) * mob * (dk[0]*dk[1]*dk[2] / pow(2.0 * pi, 3.0)) # ( e / 1 ) * (m2/eVs) * 1 =  m2/Vs
			return (-2.0 * self.e / self.ncarr) * mob  # ( e / 1 ) * (m2/eVs) * 1 =  m2/Vs

		return 0

def main(nf = 0, kf = 0):

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	if len(sys.argv) > 1: 
		scale =  int(sys.argv[1])
		if rank == 0 : print("Reading scale factor for local potential 1 /", scale, flush = True)

	t0 = time.time()
	debug = True
	restart = False
	lt = Lifetime(debug,restart,scale)
	t1 = time.time()

	if len(sys.argv) > 2: 
		lt.ds =  int(sys.argv[2])
		if lt.comm.rank == 0 : print("Reading energy distanse", lt.ds, "of Fermi smearing.", flush = True)

	if lt.comm.rank == 0:
		print("Data readed in",int(t1-t0),'s.')
		print("="*100)
		print()
	mob = lt.mobility()

	if lt.comm.rank == 0:
		print("Mobility:",mob/1e4,' cm2/Vs')
		print("Resistivity for 18.1*10^22 el/cm3 is:",3.45e-3 / mob,"e-8 Omh.m") #1e8/(18.1e28 * 1.6e-19 * mob)
		t2 = time.time()
		print("Real space reduction ",lt.scale,'- fold')
		if lt.debug: print("Data calculation",int(t2-t1),'s, total time',int(t2-t0)," Ncores:",lt.comm.size, flush = True)

	return 0

if __name__ == "__main__": main()
