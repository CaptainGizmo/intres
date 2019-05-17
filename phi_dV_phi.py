#!/usr/bin/env python3
import os,sys,time
import subprocess
import configparser
import json
import struct
from ase import Atoms
from ase.calculators.vasp import *
import numpy as np
from numpy import linalg as LA
from numpy.lib import pad
from scipy.linalg import *
from scipy.constants import *
from scipy import sparse as sps
import scipy.fftpack as fft
from math import sin,cos,asin,acos,sqrt

#import mpi4py
#mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import pickle
import phi

from getsize import get_size
from VaspKpointsInterpolated import *
from ibz2fbz import *

hbar = 4.135667662e-15

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

	def __init__(self, config):
		read_time = -time.time()
		self.nspin = -1
		self.nd = config['nd']
		self.vfield = config['vfield']
		self.vcurr = config['vcurr']
		self.ncarr =  config['ncarr']
		self.debug = config['debug']
		self.restart = config['restart']
		self.scale = config['scale']

		self.T2file = "data_sparse."+str(self.scale)+".npz"
		self.WFfile = "WF_unwrap."+str(self.scale)+".npy"

		self.comm = MPI.COMM_WORLD
		self.mob = np.zeros(2,dtype='float64')
		self.MASTER = 0
		self.WFNODE = 0 #self.comm.Get_size()-1

		# electron charge, since we work in eV units
		#e = -1.6e-19 # coulomb
		self.e = 1.0 # in electrons

		# carriers per atomic site

		##########################################################################################################################
		if self.comm.rank == self.MASTER:
			if self.debug : print('* Reading potential perturbation from LOCPOT difference.', flush = True)
			read_time = -time.time()
			self.CHGCAR = VaspChargeDensity("LOCPOT_diff")
			self.dV = self.CHGCAR.chg[0]
			print('Done in ',int(read_time + time.time()),'s.', flush = True)
			# to be multiplied by the Vcell, since VaspChargeDensity normalize charge by Vcell
		else:
			self.dV = None
		#scatter CHGCAR
		self.dV = self.comm.bcast(self.dV, root = self.MASTER)



		# number of points in LOCPOT gives points for integral in RS
		self.r = np.array(self.dV.shape, dtype='int_')
		#number of reduce points in space
		self.rs = np.array([int(self.r[0]/self.scale),int(self.r[1]/self.scale),int(self.r[2]/self.scale)],dtype='int_') 

		# calculate divergence of scattering potential
		#self.divcharge = self.div()
		
		##########################################################################################################################

		if self.comm.rank == self.MASTER:
			read_time = -time.time()
			if self.debug : print('\n* Reading simulation configuration and group velocities from vasprun.xml', flush = True)
			calc = VaspKpointsInterpolated("vasprun.xml")
			print('Done in ',int(read_time + time.time()),'s.', flush = True)
		else:
			calc = None
		calc = self.comm.bcast(calc, root = self.MASTER)

		self.cell = calc.basis
		self.occ = calc.populations
		self.kgrid = calc.kptgrid_divisions
		self.nbands = calc.nbands
		self.sigma = calc.sigma
		self.fermi = calc.efermi_interpolated
		self.nelect = calc.nelect
		self.natoms = calc.natoms #+ self.nd since now we read undistorted lattice

		# parameters for K-points in full BZ (formally interpolated, but with scaling factor 1)
		self.nkpt = calc.nkpts
		self.kpts = calc.kpts
		self.ene = calc.energies
		self.vel = calc.velocities
		self.nspin = calc.ispin

		# for ortho cell
		self.dr = np.array([0,0,0],dtype='float64')
		for i in range(3): self.dr[i] = (self.cell[i][i]) / self.r[i] #* self.scale # in Ang, r - number points of LOCPOT grid

		# step of K-mesh in reciprocal space
		self.dk = np.array([0,0,0],dtype='float')
		for i in range(3): self.dk[i] = LA.norm(calc.rec_basis[i]) / self.kgrid[i]  # unitless
		#for i in range(3): self.dk[i] = 2 * pi / (self.kgrid[i] + 0.0) # unitless!!!!

		# search for bands which are in principle crossing the FS
		self.bandlist=[]
		for spin in range(self.nspin):
			bandlist = []
			for band in range(self.nbands):
				emax=np.max(self.ene[spin,:,band])
				emin=np.min(self.ene[spin,:,band])
				if self.fermi >= emin and self.fermi <= emax :
					bandlist.append(band)
			self.bandlist.append(bandlist)
		self.bandlist = np.array(self.bandlist)

		if self.comm.rank == self.MASTER:
			if self.debug:
				print("Cell vectors:")
				print(self.cell)
				print("Volume:",LA.det(self.cell))
				print
				print("Atoms: ",self.natoms)
				print
				print('dr',self.dr, flush = True)
				print("\nReciprocal vectors:")
				print(calc.rec_basis)
				print('dk',self.dk, flush = True)
				for spin in range(self.nspin):
					print("Spin",spin,"found",len(self.bandlist[spin]),"bands crossing FS")
				#print(self.bandlist)
				print


		##########################################################################################################################
		if self.comm.rank == self.MASTER:
			# mapping for k-points from IBZ to FBZ
			print("\n* Reading k-points mapping from OUTCAR", flush = True)
			i2f = ibz2fbz("OUTCAR")
		else:
			i2f = None
		#scatter i2f
		i2f = self.comm.bcast(i2f, root = self.MASTER)

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

		if self.comm.rank == self.MASTER:
			if self.debug:
				print("Number of k-points in full IBZ interploated:",self.inkpt)
				print("Number of k-points in full BZ interpolated:",self.nkpt)
				print("\nSorting out k-points with partially occupied band levels:")

		# sorting out kpts we don't need
		# require fix for spin decoupling
		self.kptlist = []
		#print("occ",self.occ.shape)
		for spin in range(self.nspin):
			#print("Spin:",spin)
			kptlist = []
			#print("occ spin",spin,self.occ[spin].shape)
			for kpt in range(self.inkpt):
				flag = 0
				for n in self.bandlist[spin]:
					if (self.occ[spin][kpt][n] == 0.0 or self.occ[spin][kpt][n] == 1.0) : continue
					flag = 1
				if flag:
					if self.comm.rank == self.MASTER:
						print('Adding k point #',kpt, flush=True)
					kptlist.append(kpt)
			self.kptlist.append(kptlist)
			if self.comm.rank == self.MASTER:
				if self.debug:
					print(len(self.kptlist[spin]),"points added from",self.inkpt)

		##########################################################################################################################
		if self.comm.rank == self.MASTER:
			# cached value of scattering probability matrix (squared elements)
			self.T2 = sps.dok_matrix((self.inkpt*self.nbands,self.inkpt*self.nbands), dtype=np.float64)
			#if (self.restart):
			if os.path.isfile(self.T2file) :
				if self.comm.rank == self.MASTER: print("\n* Reading scattering coefficients from",self.T2file,flush = True)
				data = sps.load_npz(self.T2file)
				self.T2 = data.todok()
				if self.comm.rank == self.MASTER: print("scattering coefficients readed.",flush = True)
		else:
			self.T2 = None
		self.T2 = self.comm.bcast(self.T2, root = self.MASTER)

		##########################################################################################################################
		#reading wavefunction coefficient
		if self.comm.rank == self.MASTER:
			if self.debug :
				print('\n* Reading wave-function coefficients from WAVECAR.', flush = True)
				print

		read_time = -time.time()
		self.wavecoef()
		if self.comm.rank == self.MASTER:
			if self.debug :
				print('Done in ',int(read_time + time.time()),'s.', flush = True)
				print

		# dV multiplied by the Vcell, since VaspChargeDensity normalize charge by Vcell
		self.dV *= LA.det(self.cell)
		
		##########################################################################################################################
		# creatimg array for real space WaveFunction values
		# here we might want to keep it not on the master node
		self.WF3Dlist = None
		self.WF3D = []
		self.WF3Didx = 0

		if self.comm.rank == self.WFNODE:

			ckpt = []
			cband = []
			for spin in range(self.nspin):
				ckpt.append(len(self.kptlist[spin]))
				cband.append(len(self.bandlist[spin]))
			ckpt = np.amax(np.array(ckpt))
			cband = np.amax(np.array(cband)) # bands Crossing FS

			self.WF3Dlist = np.ones((self.nspin,ckpt,cband),dtype=int)
			self.WF3Dlist *= -1

		if self.comm.rank == self.WFNODE :
			if os.path.isfile(self.WFfile) :
				print("\n* Reading unwrapped Wave Function values.",flush = True)
				read_time = -time.time()
				file = open(self.WFfile,'rb')
				while 1:
					try:
						spin,idk,idn,phi3d = pickle.load(file)
					except EOFError:
						break
					self.WF3D.append(phi3d)
					self.WF3Dlist[spin][idk][idn] = self.WF3Didx
					self.WF3Didx +=1

				file.close()
				print('Done in ',int(read_time + time.time()),'s.', flush = True)
		#self.comm.bcast(self.WF3D, root = self.WFNODE)

		##########################################################################################################################
	
		if self.comm.rank == self.MASTER:
			print("Data readed in",int(read_time + time.time()),'s.')
			print("="*100)
			print()

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
		self.nspin = nspin

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
		if self.comm.rank == self.MASTER:
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
		if self.comm.rank == self.MASTER:
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
		if self.comm.rank == self.MASTER :
			print('max. no. G values; 1,2,3 =',nb1max,nb2max,nb3max)
			print('estimated max. no. plane waves =',npmax, flush = True)

		ckpt = []
		cband = []
		for spin in range(self.nspin):
			ckpt.append(len(self.kptlist[spin]))
			cband.append(len(self.bandlist[spin]))
		ckpt = np.amax(np.array(ckpt))
		cband = np.amax(np.array(cband)) # bands Crossing FS

		################### create the output structure ########################################
		
		if self.comm.rank == self.MASTER:
			#print("Requesting pw memory",nspin,ckpt,npmax,cband)
			#spacewise is better to request separate wf for each spin
			self.wf = Lifetime.wf(nspin,ckpt,npmax,cband)
			self.wf.Vcell = Vcell

		################## read structures: spin, kpt on each thread ###########################
		################## read coeffs parallelized by band          ###########################

		# Begin loops over spin, k-points and bands
		for spin in range(nspin):
			#if self.debug : 
			if self.comm.rank == self.MASTER :
				print()
				print('********')
				print('Reading spin ',spin,'K-points',len(self.kptlist[spin]),'of',nkpt)

			# loop over selected Kpts
			for ik in range(len(self.kptlist[spin])):
				# get real number of Kpt
				ikid = self.kptlist[spin][ik]
				if self.comm.rank == self.MASTER:
					self.wf.ids[spin][ik] = ikid
				# search and read information about the band
				recpos = (2 + ikid*(nband+1) + spin*nkpt*(nband+1)) * recl
				f.seek(recpos)
				buffer = f.read(recl)
				dummy = np.empty([int(recl/8)],dtype='d')
				fmt=str(int(recl/8))+'d'
				(dummy) = struct.unpack(fmt,buffer)
				if self.comm.rank == self.MASTER:
					self.wf.nplane[spin][ik]=int(dummy[0])
					self.wf.kpt[spin][ik] = np.array(dummy[1:4])

				for i in range(len(self.bandlist[spin])):
					if self.comm.rank == self.MASTER:
						iband = self.bandlist[spin][i]
						self.wf.cener[spin][ik][i] = dummy[5+2*iband] + 1j * dummy[5+2*iband+1]
						self.wf.occ[spin][ik][i] = dummy[5+2*nband+iband]
				if self.debug and self.comm.rank == self.MASTER :
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

							if self.comm.rank == self.MASTER:
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

				if self.comm.rank == self.MASTER:
					if ncnt > npmax:
						sys.exit('*** error - plane wave count exceeds estimate')
					if ncnt != self.wf.nplane[spin][ik] :
						sys.exit('*** error - computed no. '+str(ncnt)+' != input no.'+str(self.wf.nplane[spin][ik]))

				######### read planewave coefficients for each energy band at given K-point ####################
				#print("223",self.comm.rank, flush = True)
				rank_k_coeff = np.zeros([cband,npmax],dtype='complex64')
				#print("224",self.comm.rank, flush = True)

				if self.comm.rank == self.MASTER:
					npl = self.wf.nplane[spin][ik]
				else:
					npl = None
				npl = self.comm.bcast(npl, root = self.MASTER)

				# read coeffs in parallel
				
				#for iband in range(self.comm.rank, nband, self.comm.size):
				for i in range(self.comm.rank, len(self.bandlist[spin]), self.comm.size):
					iband = self.bandlist[spin][i]
					recpos = (2 + ikid*(nband+1) + spin*nkpt*(nband+1) + (iband+1) ) * recl
					f.seek(recpos)
					buffer = f.read(recl)
					fmt=str(int(8*npl/4))+'f'
					(dummy) = struct.unpack(fmt,buffer[:int(8*npl)])
					for iplane in range(npl):
						rank_k_coeff[i][iplane] = dummy[2*iplane] + 1j * dummy[2*iplane+1]
				
				# collect coeffs from slave nodes to the root node
				if self.comm.rank == self.MASTER:
					k_coeff = np.zeros([cband,npmax],dtype='complex64')
				else:
					k_coeff = None
				self.comm.Reduce(rank_k_coeff,  k_coeff,  op=MPI.SUM, root = self.MASTER)

				# save coeffs for the chosen Kpt
				if self.comm.rank == self.MASTER:
					self.wf.coeff[spin][ik] = k_coeff

		f.close()
		if self.comm.rank == self.MASTER:  print('Reading wavefunction coefficients is done.', flush = True)

		return


	def interpolate_fft(self,arr, scale=1):
		shape = np.shape(arr)
		pad_width = tuple( ( int(np.ceil(a/2*(scale-1))), int(np.floor(a/2*(scale-1))) ) for a in shape)
		padded = np.pad( fft.fftshift( fft.fftn(arr) ), pad_width, mode='constant')
		return fft.ifftn(  scale**len(arr.shape)*fft.ifftshift(padded) )

	def WFunwrap(self,s,k,n):
		spin = s
		rs = self.rs
		phi3d = np.zeros((rs[0],rs[1],rs[2]),dtype='complex64')

		if self.comm.rank == self.MASTER:
			idk = np.where(self.wf.ids[spin] == k)[0][0]
			idn = np.where(self.bandlist[spin] == n)[0][0]
			#print("wf ids",np.where(self.wf.ids[spin] == k)[0][0])
			#print("band ids",np.where(self.bandlist[spin] == n)[0][0])
		else:
			idk = None
			idn = None
		idk = self.comm.bcast(idk, root = self.MASTER)
		idn = self.comm.bcast(idn, root = self.MASTER)

		iscached = False
		if self.comm.rank == self.WFNODE :
			#print("self.WF3Dlist",self.WF3Dlist.shape, spin, idk, idn)
			if self.WF3Dlist[spin][idk][idn] >= 0 : # [0][0][0] :
				iscached = True

		iscached = self.comm.bcast(iscached, root = self.WFNODE)
		if iscached : return

		if self.comm.rank == self.MASTER:
			kpt = self.wf.kpt[spin][idk]
			igall = self.wf.igall[spin][idk]
			nplane = self.wf.nplane[spin][idk]
			coeff = np.asarray(self.wf.coeff[spin][idk][idn],dtype=np.complex64)
			Vcell = self.wf.Vcell
		else:
			kpt = None
			igall = None
			nplane = None
			coeff = None
			Vcell = None

		kpt = self.comm.bcast(kpt, root = self.MASTER)
		igall = self.comm.bcast(igall, root = self.MASTER)
		nplane = self.comm.bcast(nplane, root = self.MASTER)
		coeff = self.comm.bcast(coeff, root = self.MASTER)
		Vcell = self.comm.bcast(Vcell, root = self.MASTER)

		phi.phi_skn(kpt, igall, nplane, coeff, Vcell, rs,  phi3d)

		#self.WF3D[spin][idk][idn] = phi3d

		if self.comm.rank == self.WFNODE :
			# save phi by appending a new value
			self.WF3Dlist[spin][idk][idn] = self.WF3Didx
			self.WF3Didx += 1
			self.WF3D.append(phi3d)
			# append new value
			file = open(self.WFfile,'ab')
			pickle.dump([spin,idk,idn,phi3d], file)
			file.close()

		return

	def WF(self):
		"Precalculate all WFs"
		if self.comm.rank == self.MASTER :
			print( 'Precalculating Wave Functions', flush = True)
			total_time = - time.time()
		rs = self.rs

		for spin in range(self.nspin):
			for k in self.kptlist[spin]:
				if self.comm.rank == self.MASTER : print( 's =', spin, 'k =', k, '\tn = ', end='', flush = True)
				for n in self.bandlist[spin]:
					#print(self.occ[spin][k][n],flush = True)
					if self.occ[spin][k][n] == 1.0 or self.occ[spin][k][n] == 0.0 : continue
					self.WFunwrap(spin,k,n)
					if self.comm.rank == self.MASTER : print( n, " ", end='', flush = True)
				if self.comm.rank == self.MASTER : print("",flush = True)

		if self.comm.rank == self.MASTER :
			print("Done in",int(total_time + time.time()),"s.")
			print("="*50)
		return

	def T(self, phi_i, phi_f):
		"Calculate scattering matrix element, can be compex. ki - initial K-point, ni - initial energy band"

		T = 0.0
		#s = self.scale #scale of the local potential array
		r = self.r
		#rs = self.rs

		phi_i_itp = self.interpolate_fft(phi_i,self.scale)
		phi_f_itp = self.interpolate_fft(phi_f,self.scale)

		for x in range(r[0]):
			for y in range(r[1]):
				for z in range(r[2]):
					T += np.conj( phi_f_itp[x][y][z] ) * self.dV[x][y][z] * phi_i_itp[x][y][z]
		T  *= self.dr[0] * self.dr[1] * self.dr[2]

		if self.comm.rank == self.MASTER :
			print('\t<{},{}|V|{},{}> = {:e}'.format(kf,nf,ki,ni,abs(T)))
		return T


	def DDelta(self, x):
		"Dirac delta function in a form of Haussian"
		# normalization for (2pi)^3 volume
		# sigma =  pow(2.0 * pi, 2.5)
		#sigma = self.nkpt / pow(2.0 * pi,0.5)
		return 1.0 #/(pow(2.0 * pi,0.5) * sigma) * np.exp(-x*x/(2*sigma*sigma))


	def dFdE(self,s,k,n):
		# derivative of Fermi distribution 
		sigma = self.sigma #kT in energy units
		de = self.ene[s][k][n] - self.fermi #in energy units
		# sigma=kT , 1ev = 1.602e-19 J
		return -1.0/(sigma * 1.602e-19) * np.exp(de/sigma) / (np.exp(de/sigma) + 1.0)**2.0

	def saveT2(self):
		#save new T
		if self.comm.rank == self.MASTER :
			#mv old file
			if os.path.isfile(self.T2file) :
				os.rename(self.T2file, self.T2file+".0")
			
			# convert matrix to save format
			data_save = sps.csc_matrix(self.T2)
			
			#save matrix
			sps.save_npz(self.T2file, data_save)

	def scattering(self):

		for spin in range(self.nspin):
			# SPREAD ARRAY WF3D
			if self.comm.rank == self.MASTER:
				print("Precalculating scattering matrix, spin",spin)
				total_time = -time.time()
				slave_rank = 0
				issend = 1
				R = 0.0
				R_n = 0.0
				T2 = 0.0
				idx = 0
				idx_start = 0
				jobtime = np.zeros(self.comm.size, dtype = np.double)
				nkpt = len(self.kptlist[spin])
				nbands = len(self.bandlist[spin])
				MAX_idx = pow(len(self.points[spin]),2)

				#serialize over all initial and final bands and kpts
				while idx < MAX_idx:
					rewind = False

					id_in = idx // len(self.points[spin])
					id_out = idx - id_in * len(self.points[spin])
				
					if id_in < id_out:
						idx += 1
						continue

					ki, ni = self.points[spin][id_in]
					kf, nf = self.points[spin][id_out]

					# convert from global ids to array ids
					id_ki = np.where(self.wf.ids[spin] == ki)[0][0]
					id_ni = np.where(self.bandlist[spin] == ni)[0][0]
					id_kf = np.where(self.wf.ids[spin] == kf)[0][0]
					id_nf = np.where(self.bandlist[spin] == nf)[0][0]

					"""
					vel = self.vel[kf][nf]
					if(not LA.norm(vel)): # check for [0,0,0] velocity
						idx += 1
						#print('**')
						continue
					"""

					# convert global ids to scattering matrix ids
					init = ki*self.nbands + ni
					final = kf*self.nbands + nf

					#if self.T2[init,final] and issend > 0:
					if self.T2[init,final] :
						T2_cashed = True
					else:
						T2_cashed = False

					if T2_cashed:
						if issend > 0:
							T2 = self.T2[init,final]
							print('\tT(',ki,ni,'=>\t',kf,nf,') =',sqrt(T2),'eV,\tR =', 2.0*pi/hbar*T2,'eV/s REUSE ',end='\n',flush=True)
					else:

						if slave_rank == self.MASTER: slave_rank+=1 #skip the master node

						# distribute batch of jobs
						if issend > 0:
							#establish communication with a free slave
							jobtime[slave_rank] = -time.time()
							req_alert = self.comm.isend(idx, dest=slave_rank, tag=slave_rank)

							#Send data to slave
							WF3D_id_i = self.WF3Dlist[spin][id_ki][id_ni]
							buf_phi_i = np.ascontiguousarray(self.WF3D[WF3D_id_i] .reshape(-1), dtype = np.complex64)
							req_phi_i = self.comm.Isend([buf_phi_i,self.rs[0]*self.rs[1]*self.rs[2],MPI.COMPLEX], dest=slave_rank)
						
							WF3D_id_f = self.WF3Dlist[spin][id_kf][id_nf]
							buf_phi_f = np.ascontiguousarray(self.WF3D[WF3D_id_f].reshape(-1), dtype = np.complex64)
							req_phi_f = self.comm.Isend([buf_phi_f,self.rs[0]*self.rs[1]*self.rs[2],MPI.COMPLEX], dest=slave_rank)

						if issend <= 0:
							#if all distributed, wait and collect a batch of calculated values
							buf_T2 = np.zeros(2, dtype = np.double)
							req_T2 = self.comm.irecv(buf_T2, source=slave_rank)
							while not req_T2.Get_status() :
								time.sleep(0.01)
							T2 = req_T2.wait()
							self.T2[init,final] = self.T2[final,init] = T2
							jobtime[slave_rank] += time.time()
							print('\tT(',ki,ni,'=>\t',kf,nf,') =',sqrt(T2),'eV,\tR =', 2.0*pi/hbar*T2,'eV/s',int(jobtime[slave_rank]*100.0)/100.0,'s ',end='\n',flush=True)

						# find if the border of data chank or the border of tasks reached
						if ( slave_rank >= self.comm.size - 1) or ( idx == MAX_idx - 1 ):
							#if receiving the last data then receive and exit
							if idx == MAX_idx - 1 and issend < 0:
								self.saveT2()
							else:
								issend *= -1
								slave_rank = 0
								if issend < 0: # just ended with sending (switched from >0)
									idx_temp = idx
									idx = idx_start
									idx_start = idx_temp + 1
									continue
								else: # just ended recv
									self.saveT2()
						else:
							slave_rank += 1
					idx += 1

				############## end of global idx loop #######################################################
				# distribute exit command to the slave nodes
				for slave_rank in range(0, self.comm.size):
					if slave_rank == self.MASTER: continue
					req_alert = self.comm.isend(-1, dest=slave_rank, tag=slave_rank)
				print("\nDone in",int(total_time + time.time()),"s.")
				print("="*50)

			else:
				idx=0
				while idx >= 0 :
				
					req_idx = self.comm.irecv(source=self.MASTER,tag=self.comm.rank)
					while not req_idx.Get_status() :
						time.sleep(0.01)
					idx = req_idx.wait()

					if idx >= 0:
						#print("recv idx",idx,flush=True)
						buf_phi_i = np.empty(self.rs[0]*self.rs[1]*self.rs[2] , dtype = np.complex64)
						req_phi_i = self.comm.Irecv([buf_phi_i,self.rs[0]*self.rs[1]*self.rs[2],MPI.COMPLEX], source=self.MASTER)
						while not req_phi_i.Get_status() :
							time.sleep(0.01)
						req_phi_i.Wait()
						phi_i = buf_phi_i.reshape(self.rs[0], self.rs[1], self.rs[2])


						buf_phi_f = np.empty( int(self.rs[0]*self.rs[1]*self.rs[2]) , dtype = np.complex64)
						req_phi_f = self.comm.Irecv([buf_phi_f,self.rs[0]*self.rs[1]*self.rs[2],MPI.COMPLEX], source=self.MASTER)
						while not req_phi_f.Get_status() :
							time.sleep(0.01)
						req_phi_f.Wait()
						phi_f = buf_phi_f.reshape(self.rs[0], self.rs[1], self.rs[2])

						T2=float(pow(abs(self.T(phi_i, phi_f)),2))
						#return to master
						req_T2 = self.comm.isend(T2, dest=self.MASTER)


		return

	def assembly(self,spin):

		dk = self.dk # unitless, 0..2pi/N

		if self.comm.rank == self.MASTER:
			print("Assemblying scattering integral, spin", spin)
			total_time = -time.time()
		for pt_i in self.points[spin]:
			ki, ni = pt_i
			init = ki*self.nbands + ni
			R_i = 0.0
			
			for pt_f in self.points[spin]:
				kf, nf = pt_f
				final = kf*self.nbands + nf

				# sum over all reflections of reduced K-point
				R_if = 0.0
				kpts = np.where(self.ibz2fbz == kf)[0]
				if self.comm.rank == self.MASTER: print("\tAdding",kpts.shape[0],"kpti reflections",flush=True)
				for kf_rfl in kpts:
					# FBZ - interpolated FBZ number correspondence for group velocity
					v_i = self.vel[spin][ki][ni]
					v_f = self.vel[spin][kf_rfl][nf]
					v_i_n = LA.norm(v_i)
					v_f_n = LA.norm(v_f)
					# 5 check for zero group velocity
					if (not v_i_n or not v_f_n): continue
					costheta = np.dot(v_i,v_f)/(v_i_n * v_f_n)
					de = 0 #de = ef - ei
					#if (costheta != 1.0): R_if +=  self.DDelta(de) * (1.0 - costheta) # (eV)^2
					if (costheta != 1.0): R_if += (1.0 - costheta) # (eV)^2
				R_i += R_if * self.T2[init,final]

			R_i *= (self.natoms / self.nd)  * (2.0 * pi/hbar) * (dk[0] * dk[1] * dk[2] / pow(2.0*pi,3.0)) # 1 * (1 / eVs) * (eV)^2 * 1 != eV/s
			#R_i *= self.nd * (2.0 * pi/hbar) * (dk[0] * dk[1] * dk[2] / pow(2.0*pi,3.0)) # 1 * (1 / eVs) * (eV)^2 * 1 != eV/s
			if self.comm.rank == self.MASTER: print( 'R(k=', ki, 'n=', ni, ') = ', R_i, 'eV/s. ',end='', flush = True)

			# check for null division
			if not R_i:
				continue
			else:
				#tau = 1.0 / R_i / self.natoms
				tau = 1.0 / R_i

			# sum over all reflections of reduced K-point
			kpts = np.where(self.ibz2fbz == ki)[0]
			if self.comm.rank == self.MASTER: print("Adding",kpts.shape[0],"kptf reflections\n",flush = True)
			for ki_rfl in kpts:
				v_i = self.vel[spin][ki_rfl][ni] # velocity units A/fs = 1e-10 m / 1e-15 s = 1e5 m/s
				proj1 = np.dot(v_i,self.vfield) * 1e5 # projection of group velocity on field direction
				proj2 = np.dot(v_i,self.vcurr) * 1e5 # projection of group velocity on current direction
				self.mob[spin] += tau * self.dFdE(spin,ki_rfl,ni) * np.dot(proj1,proj2) #  s/eV * (m/s)^2 = m2/eVs

		self.mob[spin] *= (-1.0 * self.e / self.nelect) * (dk[0]*dk[1]*dk[2] / pow(2.0 * pi, 3.0)) # ( e / 1 ) * (m2/eVs) * 1 =  m2/Vs no spin degeneracy
		#self.mob[spin] *= (-2.0 * self.e / self.nelect) * (dk[0]*dk[1]*dk[2] / pow(2.0 * pi, 3.0)) # ( e / 1 ) * (m2/eVs) * 1 =  m2/Vs

		if self.comm.rank == self.MASTER:
			print("Done in",int(total_time + time.time()),"s.")
			#print("="*50)

		return


	def mobility(self):
		"Carrier mobility calculation, sum over all bands of integrals over all K-points"
		# find "global ids" for nonzero scattering
		self.points = []
		for spin in range(self.nspin):
			points = []
			for k in self.kptlist[spin]:
				for n in self.bandlist[spin]:
					if self.occ[spin][k][n] == 1.0 or self.occ[spin][k][n] == 0.0 : continue
					else: points.append([k,n])
			self.points.append(points)

		# pre-calculate WFs
		self.WF()

		# calculate scattering matrix
		self.scattering()

		# assembly integral 
		self.assembly(0)
		if self.nspin == 2: self.assembly(1)
		else: self.mob[1] = self.mob[0]

		if self.comm.rank == self.MASTER: print("="*100)

		return

def main(nf = 0, kf = 0):

	debug = True
	restart = False

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()

	if len(sys.argv) > 1:
		config_file = sys.argv[1]
	else:
		config_file = "config.ini"
	if rank == 0 : print("Reading configuration from", config_file, flush = True)

	conf = configparser.ConfigParser()
	conf.read(config_file)

	# default
	config = {'element': 'Unknown', \
		'nelec': 1, \
		'nd': 1, \
		'ncarr': 1, \
		'scale': 1, \
		'vfield': [1,0,0], \
		'vcurr': [1,0,0], \
		'restart': False, \
		'debug': True }

	#if conf["config"]["element"]
	config['element'] = conf["config"]["element"]
	config['nelec'] = json.loads(conf["config"]["nelec"])
	config['nd'] = json.loads(conf["config"]["nd"])
	config['ncarr'] = json.loads(conf["config"]["ncarr"])
	config['scale'] = json.loads(conf["config"]["scale"])
	config['vfield'] = json.loads(conf["config"]["vfield"])
	config['vcurr'] = json.loads(conf["config"]["vcurr"])

	if rank == 0 :
		print("Mobility calculation for the element {} with {} electrons per atom and {} carriers per m^3".format(config['element'],config['nelec'],config['ncarr']))
		print("Declared number of defects in the supercell is {}, local potential scaling 1 / {}".format(config['nd'],config['scale']))
		print("Electric field applied in {} direction, current carriers mobility along {} direction.".format(config['vfield'],config['vcurr']), flush = True)

	total_time = -time.time()
	lt = Lifetime(config)
	calc_time = -time.time()
	lt.mobility()

	if lt.comm.rank == 0:

		#ncarr = 18e28 #el/m3 Al
		#ncarr = 7.31e27 # el/m3 W
		#ndef = 1e16  # N/m3 at 300K, formation en 0.7 eV
		#ndef = nd
		q = 1.6e-19

		mob = lt.mob.sum()
		print("Mobility of 1 electron over 1 defect : {:E} cm2/Vs".format(mob * 1e-4)) # convert from m2 to cm2
		print("Resistivity for {:E} el/m3 and {:E} defects/m3 is: {:E} Omh.m".format(config['ncarr'],config['nd'], 1.0 / (config['ncarr'] * q * mob)))
		print("Conductivity for {:E} el/m3 and {:E} defects/m3 is: {:E} S/m".format(config['ncarr'],config['nd'],config['ncarr'] * q * mob))
		print("Real space reduction ",lt.scale,'- fold')
		print("Total data calculation",int(calc_time + time.time()),'s, total time',int(total_time + time.time()),"s. Ncores:",lt.comm.size, flush = True)

	return 0

if __name__ == "__main__": main()
