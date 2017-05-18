import numpy as np
from numpy import linalg as LA
import os,sys,time
from math import sin, cos
from scipy.constants import *
from ase import Atoms
from ase.calculators.vasp import *
import subprocess
from sympy import DiracDelta

class Lifetime(object):

	def __init__(self, test=False):
		self.test = test
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

	def div(self):

		div = np.empty(self.r,dtype=float)
		dr = np.empty(3)
		dr[0] = self.dr[0]*self.cell[0][0]
		dr[1] = self.dr[1]*self.cell[1][1]
		dr[2] = self.dr[2]*self.cell[2][2]

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
		"""
		for x in range(self.r[0]):
			for y in range(self.r[1]):
				for z in range(self.r[2]):
					print(x,y,z,div[x][y][z])
		"""
		return div


	def phi(self,n, k, x, y, nz):
		"Return wavefunction value for n-th energy level, k-point and {x,y,z} point in real space"
		# call external program WaveTransPlot(n,k,x,y), read array [z, Re(phi(z)), Im(phi(z)) ]
		# x y z in partial coordinates: 0..1 (not including 1)
		# k+1, n+1 since array stars from 0
		nmap_out = subprocess.run(args = ['WaveTransPlotXYZ','-b',str(n+1),'-k',str(k+1),'-x',str(x),'-y',str(y),'-nz',str(nz)], universal_newlines = True, stdout = subprocess.PIPE)
		out = np.empty([nz], dtype=complex)

		for z in range(nz):
			nmap_lines = nmap_out.stdout.splitlines()
			out[z] = complex(float(nmap_lines[0].split()[1]),float(nmap_lines[0].split()[2]))
		#if (self.test):
		#	print('Phi(',n,k,z,y,'):')
		#	print(out)
		return out

	def T(self,ni, ki, nf, kf):
		"Calculate scattering matrix element, can be compex"

		if (self.test):
			print('T(',ni,ki,nf,kf,') = ', end='', flush=True)

		T = 0.0
		for x in range(self.r[0]):
			for y in range(self.r[1]):
				# get values for the wavefunction for r[2] points in z direction
				phi_z = self.phi(nf,kf,x,y,self.r[2])
				for z in range(self.r[2]):
					T += np.conj(phi_z[z]) * self.charge[x][y][z] * phi_z[z]  * self.dr[0] * self.dr[1] * self.dr[2]

		#if (self.test):
		#	print(T)

		return T

	def R(self,ni, ki, nf, kf):
		"Calculate scattering rate matrix element"

		ei = self.ene[ki][ni]
		ef = self.ene[kf][nf]
		#RR = ( 2 * pi / hbar ) * pow(self.T(ni,ki,nf,kf),2) * DiracDelta(ef - ei)
		#if (self.test):
		print("R(",ni,ki,ei," -> ",nf,kf,ef,") = ",pow(self.T(ni,ki,nf,kf),2))
		#return R
		return


	def invlifetime(self,nf,kf):
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

#			# loop over k-points
#			for kix in range(kmax[0]):
#				for kiy in range(kmax[1]):
#					for kiz in range(kmax[2]):
#						# angle between vectors v(ni,ki) and v(nf,kf) have to be implemented
#						#a = v(ni,ki)
#						#b = v(nf,kf)
#						#costheta=np.dot(a,b)/(LA.norm(a)*LA.norm(b))
#						#weight of ki
#						#wki = 1
#						ki = k[kix,kiy,kiz]
#						#invt += wki * (dkix*dkiy*dkiz)/pow(2.0*pi,3) * R(ni,ki,nf,kf) * (1.0 - costheta)
#						R(ni,ki,nf,kf)
	
#		return nd * invt
		return
	
	def mobility(self):
		for nf in range(self.nbands):
			# loop over irreducible k-points
			for kf in range(self.nikpt):
				self.invlifetime(nf,kf)
		return

def main(nf = 0, kf = 0):
	
	debug = False
	qwe = Lifetime(debug)
	#qwe.mobility()
	#qwe.R(0,0,0,0)

	return 0

if __name__ == "__main__": main()
