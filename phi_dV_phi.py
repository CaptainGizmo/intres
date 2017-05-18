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
from sympy import DiracDelta
from math import sin,cos,asin,acos,sqrt


class Lifetime(object):


	class wf():
		def __init__(self,nspin,nkpt,npmax,nband):
			# assign memory
			self.occ   = np.empty([nspin,nkpt,nband], dtype = 'float64')
			self.cener = np.empty([nspin,nkpt,nband], dtype = 'complex128')
			self.igall = np.empty([nspin,nkpt,npmax,3],dtype='i')
			self.coeff = np.empty([nspin,nkpt,nband,npmax],dtype='complex64')
			self.kpt   = np.empty([nspin,nkpt,3],dtype='f')
			self.nplane = np.empty([nspin,nkpt],dtype='i')
			self.Vcell = 0
			self.nband = nband


	def __init__(self, test=False):
		self.test = test

		#reading wavefunction
		#self.wf = 
		self.wavecoef()

		if (self.test):
			print('Creating Lifetime, ', end='', flush=True)
		self.calc = Vasp(restart=True)
		if (self.test):
			print(' VASP readed, ', end='', flush=True)
		self.CHGCAR = VaspChargeDensity("CHGCAR_diff")
		if (self.test):
			print('CHGCAR readed, ', end='', flush=True)
		# size of cell in real space
		self.cell = np.array(self.CHGCAR.atoms[0].cell)
		#size of the grid in CHGCAR
		self.charge = self.CHGCAR.chg[0]
		# number of points in CHGCAR gives points for integral in RS
		self.r = np.array(self.charge.shape, dtype=int)
		self.dr = np.array(1.0/self.r)
		# calculate divergence of scattering potential
		self.divcharge = self.div()
		self.ikpt = self.calc.get_ibz_kpoints()
		self.nikpt = self.ikpt.shape[0]
		self.nbands = self.calc.get_number_of_bands()
		# "empty" 2D array of energies for each IKP, fast initialization
		self.ene = np.empty([self.nikpt,self.nbands],dtype=float)
		#occup = calc.read_occupation_numbers(k)

		# reading energies for each IKP
		for k in range(self.nikpt):
			self.ene[k] = np.array(self.calc.read_eigenvalues(k))
		if (self.test):
			print('Energies readed. Done.')



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
		ecut = ecut_

		if (self.test):
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

		if (self.test):
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

		if (self.test):
			print('max. no. G values; 1,2,3 =',nb1max,nb2max,nb3max)
			print('estimated max. no. plane waves =',npmax)

		# assign memory
		self.wf = Lifetime.wf(nspin,nkpt,npmax,nband)
		self.wf.Vcell = Vcell

		# Begin loops over spin, k-points and bands
		for spin in range(nspin):
			if (self.test):
				print()
				print('********')
				print('reading spin ',spin)

			for ik in range(nkpt):
				buffer = f.read(recl)
				dummy = np.empty([int(recl/8)],dtype='d')
				fmt=str(int(recl/8))+'d'
				(dummy) = struct.unpack(fmt,buffer)
				self.wf.nplane[spin][ik]=int(dummy[0])
				self.wf.kpt[spin][ik] = np.array(dummy[1:4])
				for i in range(nband):
					self.wf.cener[spin][ik][i] = dummy[5+2*i] + 1j * dummy[5+2*i+1]
					self.wf.occ[spin][ik][i] = dummy[5+2*nband+i]

				if (self.test):
					print('k point #',ik,'  input no. of plane waves =', self.wf.nplane[spin][ik], 'k value =',self.wf.kpt[spin][ik])

				# Calculate available plane waves
				ncnt = 0
				for ig3 in range(0, 2*nb3max+1):
					ig3p = ig3
					if ig3 > nb3max:
						ig3p = ig3 - 2*nb3max - 1
					for ig2 in range(0, 2*nb2max+1):
						ig2p = ig2
						if ig2 > nb2max:
							ig2p = ig2 - 2*nb2max - 1
						for ig1 in range(0, 2*nb1max+1):
							ig1p = ig1
							if ig1 > nb1max :
								ig1p = ig1 - 2*nb1max - 1

							sumkg = np.empty([3],dtype='d')
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

				if ncnt > npmax:
					sys.exit('*** error - plane wave count exceeds estimate')
				if ncnt != self.wf.nplane[spin][ik] :
					sys.exit('*** error - computed no. '+str(ncnt)+' != input no.'+str(self.wf.nplane[spin][ik]))

				# read planewave coefficients for each energy band at given K-point
				for iband in range(self.wf.nband):
					buffer = f.read(recl)
					fmt=str(int(8*self.wf.nplane[spin][ik]/4))+'f'
					(dummy) = struct.unpack(fmt,buffer[:int(8*self.wf.nplane[spin][ik])])
					for iplane in range(self.wf.nplane[spin][ik]):
						self.wf.coeff[spin][ik][iband][iplane] = dummy[2*iplane] + 1j * dummy[2*iplane+1]
		f.close()
		return


	def phi(self, spin, band, k, x, y, z):
		"Return wavefunction value for n-th energy level, k-point and {x,y,z} point in real space"
		# x y z in partial coordinates: 0..1 (not including 1)
		if (x<0 or x>=1) or (y<0 or y>=1) or (z<0 or z>=1):
			sys.exit("PHI(x,y,z): coordinate(s) not in reduced scale.")

		csum = complex(0.,0.)
		for iplane in range(self.wf.nplane[spin][k]):
			csum   += self.wf.coeff[spin][k][band][iplane] \
					* np.exp( 2.* pi * 1j \
								* (self.wf.kpt[spin][k] + self.wf.igall[spin][k][iplane]).dot(np.array([x,y,z])) \
							)

		csum /= sqrt(self.wf.Vcell)
		return csum


	def div(self):
		"Calculate the divergence of the charge dencity"
		div = np.empty(self.r,dtype=float)
		dr = np.empty(3)
		dr[0] = self.dr[0]*self.cell[0][0]
		dr[1] = self.dr[1]*self.cell[1][1]
		dr[2] = self.dr[2]*self.cell[2][2]
		
		#padding edges of the array
		#self.charge = pad(indata,(1,1,1),'reflect',reflect_type='odd')

		for x in range(self.r[0]):

			if x == 0:
				xm1 = self.r[0] - 1
			else:
				xm1 = x - 1
			if x == (self.r[0] - 1):
				xp1 = 0
			else:
				xp1 = x + 1

			for y in range(self.r[1]):

				if y == 0:
					ym1 = self.r[1] - 1
				else:
					ym1 = y - 1
				if y == (self.r[1] - 1):
					yp1 = 0
				else:
					yp1 = y + 1

				for z in range(self.r[2]):

					if z == 0:
						zm1 = self.r[2] - 1
					else:
						zm1 = z - 1
					if z == (self.r[2] - 1):
						zp1 = 0
					else:
						zp1 = z + 1
					
					#print(xm1,x,xp1,",",ym1,y,yp1,",",zm1,z,zp1)
					div[x][y][z] = (self.charge[xm1][y][z]-2.0*self.charge[x][y][z]+self.charge[xp1][y][z])/pow(dr[0],2) \
								 + (self.charge[x][ym1][z]-2.0*self.charge[x][y][z]+self.charge[x][yp1][z])/pow(dr[1],2) \
								 + (self.charge[x][y][zm1]-2.0*self.charge[x][y][z]+self.charge[x][y][zp1])/pow(dr[2],2)
		return div



	def T(self, ki, ni, kf, nf):
		"Calculate scattering matrix element, can be compex. ki - initial K-point, ni - initial energy band"

		if (self.test):
			print('T')

		T = 0.0
		for x in range(self.r[0]):
			print (x)
			for y in range(self.r[1]):
				print(y, end=' ', flush=True)
				for z in range(self.r[2]):
					# non spin-polarized
					phi = self.phi(0,nf,kf,x/self.r[0],y/self.r[1],z/self.r[2])
					T += np.conj(phi) * self.charge[x][y][z] * phi
			print()

		T *= self.dr[0] * self.dr[1] * self.dr[2]

		if (self.test):
			print('T(',ki,ni,' -> ',kf,nf,') = ', T)

		return T

	def R(self,ki, ni, kf, nf):
		"Calculate scattering rate matrix element"

		ei = self.ene[ki][ni]
		ef = self.ene[kf][nf]
		#RR = ( 2 * pi / hbar ) * pow(self.T(ni,ki,nf,kf),2) * DiracDelta(ef - ei)
		#if (self.test):
		print("R(",ki,ni,ei," -> ",kf,nf,ef,") = ",pow(self.T(ki,ni,kf,nf),2))
		#return R
		return


	def invlifetime(self,kf,nf):
		"Get inverse lifetime for state n, k"
		# defect dencity
		nd = 1
	
		# initial value for inverse t
		invt = 0.0

		# step of K-mesh in reciprocal space
		# dkix = dxiy = dkiz = 2 * pi / kmax;
		
		costheta = 0

		# loop over all energy levels
		for ni in range(nf-1):
			# loop over irreducible k-points
			for ki in range(kf-1):
				ei = self.ene[ki][ni]
				ef = self.ene[kf][nf]
				if  ei == ef :
					#if ki != kf or ni != nf:
					print(ni,ki,nf,kf)
					#self.R(ni,ki,nf,kf)
			"""
			# loop over k-points
			for kix in range(kmax[0]):
				for kiy in range(kmax[1]):
					for kiz in range(kmax[2]):
						# angle between vectors v(ni,ki) and v(nf,kf) have to be implemented
						#a = v(ni,ki)
						#b = v(nf,kf)
						#costheta=np.dot(a,b)/(LA.norm(a)*LA.norm(b))
						#weight of ki
						#wki = 1
						ki = k[kix,kiy,kiz]
						#invt += wki * (dkix*dkiy*dkiz)/pow(2.0*pi,3) * R(ki,ni,kf,nf) * (1.0 - costheta)
						R(ki,ni,kf,nf)
			"""
		#return nd * invt
		return
	
	def mobility(self):
		for nf in range(self.nbands):
			# loop over irreducible k-points
			for kf in range(self.nikpt):
				self.invlifetime(kf,nf)
		return

def main(nf = 0, kf = 0):

	debug = True
	qwe = Lifetime(debug)
	
	#qwe.mobility()
	qwe.T(0,0,1,0)

	return 0



if __name__ == "__main__": main()
