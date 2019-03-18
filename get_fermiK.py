#!/usr/bin/env python3
from ibz2fbz import *
from VaspKpointsInterpolated import *
import numpy as np

calc = VaspKpointsInterpolated("vasprun.xml")
cell = calc.basis
occ = calc.populations
kgrid = calc.kptgrid_divisions
nbands = calc.nbands
sigma = calc.sigma
fermi = calc.efermi_interpolated
nelect = calc.nelect
natoms = calc.natoms #+ nd since now we read undistorted lattice
# parameters for K-points in full BZ (formally interpolated, but with scaling factor 1)
nkpt = calc.nkpts
kpts = calc.kpts
ene = calc.energies
vel = calc.velocities


i2f = ibz2fbz("OUTCAR")
nibz2fbz = i2f.nitpi2f
ibz2fbz = i2f.itpi2f
#find IBZ Kpts:
ikpts=[]
for kpt in range(nkpt):
	# check that Kpt is from IBZ:
	if kpt == ibz2fbz[kpt]:
		ikpts.append(kpt)
ikpts = np.array(ikpts)
inkpt = ikpts.shape[0]


bandlist=[]
for band in range(nbands):
	emax=np.max(ene[:,band])
	emin=np.min(ene[:,band])
	if fermi >= emin and fermi <= emax :
		bandlist.append(band)
bandlist = np.array(bandlist)

kptlist = []
for kpt in range(inkpt):
	flag = 0
	for n in bandlist:
		if (occ[kpt][n] == 0.0 or occ[kpt][n] == 1.0) : continue
		flag = 1
	if flag:
		print('Adding k point #',kpt, flush=True)
		kptlist.append(kpt)

file_KPT=open("KPOINTS_","w")
file_KPT.write("Manual mesh\n")
file_KPT.write(str(len(kptlist))+"\n")
file_KPT.write("rec\n")
for kpt in kptlist:
	#for kpt in nibz2fbz:
	#if kpt[0] in kptlist:
	k=nibz2fbz[kpt]
	file_KPT.write("{:f}\t{:f}\t{:f}\t{:f}\t{:d}\n".format(k[1],k[2],k[3],k[4],int(k[0])))
file_KPT.close()