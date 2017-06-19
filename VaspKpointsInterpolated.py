#!/usr/bin/env python3
"""
Parsing and data handling of spectra obtained by VASP's
internal interpolation functionality.
"""

# Module Information.
__version__   = "0.2"
__author__    = "Paul Erhart"
__url__       = ""
__copyright__ = "(c) 2014 Paul Erhart"
__license__   = "GPL"

from mpi4py import MPI

class VaspKpointsInterpolatedError(Exception): pass

# Import numpy. Error if fails
try:
    import numpy as np
except ImportError:
    raise VaspKpointsInterpolatedError( "'numpy' is not installed." )

try:
    import xml.etree.ElementTree as ET
except ImportError:
    raise VaspKpointsInterpolatedError( "'xml.etree' is not installed." )

try:
    import os
except ImportError:
    raise VaspKpointsInterpolatedError( "'os' is not installed." )



class VaspKpointsInterpolated:
    """
    Parsing and data handling of spectra obtained by VASP's
    internal interpolation functionality.
    """

    names = ['energies',
             'velocities',
             'kpts',
             'nbands',
             'nkpts',
             'kptgrid_type',
             'kptgrid_divisions',
             'ispin',
             'nelect',
             'kinter',
             'efermi_interpolated',
             'dos']
    
    def __init__(self, filename='vasprun.xml', tolerance=1e-5):
        self.data = d = {}
        self.comm = MPI.COMM_WORLD
        self.basis = None
        self.rec_basis = None
        
        # data for irreducible interpolated
        self.ienergies = None
        self.ivelocities = None
        self.ikpts = None
        self.inkpts = None
        self.inbands = None
        self.ipopulations = None
        self.dk = None
        
        # data for full interpolated
        self.energies = None
        self.velocities = None
        self.kpts = None
        self.nkpts = None
        self.nbands = None

        self.kptgrid_type = None
        self.kptgrid_divisions = None
        self.ispin = None
        self.nelect = None
        self.kinter = None
        self.dos = None
        self.efermi_interpolated = None
        self.sigma = None
        self.read_data(filename)



    def __repr__(self):
        return 'I am Munx. Who are you?'


    def copy(self):
        """Return a full copy."""
        import copy
        scf = self.__class__()
        scf.energies = self.energies
        scf.velocities = self.velocities
        scf.kpts = self.kpts
        return scf


    def read_data(self, filename):
        """
        Parse interpolated data from a vasprun.xml file.
        """

        if isinstance(filename, str):
            if not os.path.isfile(filename):
                print("ERROR: cannot access file '"+filename+"'")
                exit(1)
        try:
            xmldoc = ET.parse(filename)
        except:
            raise IOError( "failed reading input file" )


        # 1) extract information regarding k-point grid
        if self.comm.rank == 0: print("parsing k-point grid information")
        self.dk = []
        for block in xmldoc.findall("kpoints/generation"):
            if not 'param' in block.attrib:
                continue
            self.kptgrid_type = block.attrib['param']
            for element in block.findall("v"):
                if not 'name' in element.attrib:
                    continue
                if element.attrib['name'] == "divisions":
                    self.kptgrid_divisions = [int(x) for x in element.text.split()]
                if "genvec" in element.attrib['name']:
                    dk = []
                    for v in list(element.text.split()):
                        dk.append(float(v))
                    self.dk.append(dk)
        self.dk = np.array(self.dk)
        if self.comm.rank == 0: 
            print ("k-point grid: ",self.kptgrid_type)
            print ("k-point step: ",self.kptgrid_divisions)
            print


        # 2) extract value of KINTER
        if self.comm.rank == 0: print("extracting value of KINTER")
        for element in xmldoc.findall("incar/i"):
            if not 'name' in element.attrib:
                continue
            if element.attrib['name'] == 'KINTER':
                self.kinter = int(element.text)
                break
        if self.comm.rank == 0: print ("KINTER = ",self.kinter)


        # 3) extract some parameters
        if self.comm.rank == 0: print("extracting some parameters")
        for element in xmldoc.findall("parameters/*/i"):
            if not 'name' in element.attrib:
                continue
            if element.attrib['name'] == 'NELECT':
                self.nelect = float(element.text)
        if self.comm.rank == 0: print("NELECT = ",self.nelect)

        for element in xmldoc.findall("parameters/*/*/i"):
            if not 'name' in element.attrib:
                continue
            if element.attrib['name'] == 'ISPIN':
                self.ispin = int(element.text)
            if element.attrib['name'] == 'SIGMA':
                self.sigma = float(element.text)
        if self.comm.rank == 0: print("ISPIN = ",self.ispin)

        # 3) extract some parameters
        if self.comm.rank == 0: print("extracting cell vectors")
        self.basis = []
        self.rec_basis = []
        block = None
        for block in xmldoc.findall("structure"):
            if block.attrib['name'] == 'initialpos':
                for element in block.findall("crystal/varray"):
                   if element.attrib['name'] == 'basis':
                       vec = []
                       for v in list(element):
                           vec.append([float(x) for x in v.text.split()])
                       self.basis = vec
                   if element.attrib['name'] == 'rec_basis':
                       rec = []
                       for v in list(element):
                           rec.append([float(x) for x in v.text.split()])
                       self.rec_basis = vec
        self.basis = np.array(self.basis)
        self.rec_basis = np.array(self.rec_basis)

        # ======================================================================================

        # 4) locate eigenvalues block with interpolated data
        if self.comm.rank == 0: print("locating interpolated data for IBZ")
        block = None
        for block in xmldoc.findall("calculation/eigenvalues/velocities"):
            if 'comment' in block.attrib:
                if block.attrib['comment'] == 'interpolated_ibz':
                    break
        if block is None:
            print("ERROR: failed to locate interpolated eigenvalues in input file")
            exit(1)


        # 5) extract information regarding k-point grid
        if self.comm.rank == 0: print("parsing k-point grid information for IBZ")
        self.ikpts = []
        for element in block.findall('kpoints_ibz/varray'):
            if not 'name' in element.attrib:
                continue
            if element.attrib['name'] == 'kpointlist':
                for v in element.findall('v'):
                    self.ikpts.append([float(x) for x in v.text.split()])
        self.ikpts = np.array(self.ikpts)
        self.inkpts = len(self.ikpts)


        # 6) extract eigen energies and group velocities
        if self.comm.rank == 0: print("parsing eigen energies and group velocities for IBZ")
        self.ienergies = []
        self.ivelocities = []
        for element in block.findall('eigenvalues/array/*/*/set'):
            if 'kpoint' in element.attrib['comment']:
                en = []
                vel = []
                for v in list(element):
                    en.append(float(v.text.split()[0]))
                    vel.append([float(x) for x in v.text.split()[1:]])
                self.ienergies.append(en)
                self.ivelocities.append(vel)

        # 7) extract populations
        if self.comm.rank == 0: print("parsing populations")

        block = None
        for block in xmldoc.findall("calculation/eigenvalues"):
            pass
        self.ipopulations = []
        for element in block.findall('array/*/*/set'):
            if 'kpoint' in element.attrib['comment']:
                pop = []
                for v in list(element):
                    pop.append(float(v.text.split()[1]))
                self.ipopulations.append(pop)

        self.ienergies = np.array(self.ienergies)
        self.ivelocities = np.array(self.ivelocities)
        self.ipopulations = np.array(self.ipopulations)
        self.inbands = len(self.ienergies[0])

        # ============================================================

        # 4) locate eigenvalues block with interpolated data
        if self.comm.rank == 0: print("locating interpolated data for the full BZ")

        for block in xmldoc.findall("calculation/eigenvalues/electronvelocities"):
            if 'comment' in block.attrib:
                if block.attrib['comment'] == 'interpolated_ibz':
                    break
        if block is None:
            print("ERROR: failed to locate interpolated eigenvalues for full BZ in input file")
            exit(1)


        # 5) extract information regarding k-point grid
        if self.comm.rank == 0: print("parsing k-point grid information")
        self.kpts = []
        for element in block.findall('kpoints/varray'):
            if not 'name' in element.attrib:
                continue
            if element.attrib['name'] == 'kpointlist':
                for v in element.findall('v'):
                    self.kpts.append([float(x) for x in v.text.split()])
        self.kpts = np.array(self.kpts)
        self.nkpts = len(self.kpts)


        # 6) extract eigen energies and group velocities
        if self.comm.rank == 0: print("parsing eigen energies and group velocities")
        self.energies = []
        self.velocities = []
        for element in block.findall('eigenvalues/array/*/*/set'):
            if 'kpoint' in element.attrib['comment']:
                en = []
                vel = []
                for v in list(element):
                    en.append(float(v.text.split()[0]))
                    vel.append([float(x) for x in v.text.split()[1:]])
                self.energies.append(en)
                self.velocities.append(vel)
        self.energies = np.array(self.energies)
        self.velocities = np.array(self.velocities)
        self.nbands = len(self.energies[0])


        # ===========================================================


        # 8) extract Fermi energy
        if self.comm.rank == 0: print("extracting interpolated Fermi energy")
        block = None
        for block in xmldoc.findall("calculation/dos"):
            if 'comment' in block.attrib:
                if block.attrib['comment'] == 'interpolated':
                    break
        if block is None:
            print("ERROR: failed to locate interpolated DOS in input file")
            exit(1)
        for element in block.findall('i'):
            if not 'name' in element.attrib:
                continue
            if element.attrib['name'] == 'efermi':
                self.efermi_interpolated = float(element.text)
                break
        if self.comm.rank == 0: print("Fermi energy from interpolated data: ",self.efermi_interpolated)

        return True

    
    def get_interpolated_grid(self):
        """
        Return divisions of interpolated k-point grid.
        """
        return np.array(self.kptgrid_divisions) * self.kinter


    def print_kpoints(self):
        """
        Print a list of k-point coordinates and weights to stdout.
        """
        for kpt in zip(self.kpts):
            print(kpt)

    def print_ikpoints(self):
        """
        Print a list of k-point coordinates and weights to stdout.
        """
        for kpt,pop in zip(self.ikpts,self.ipopulations):
            print(kpt,pop)


    def get_kpoint(self, kpt_target):
        """
        Return energy and group velocity (vector) at desired k-point.
        """

        for en,vel,kpt in zip(self.energies, self.velocities, self.kpts):
            if np.all(np.abs(np.array(kpt) - np.array(kpt_target)) < self.tolerance):
                return en,vel
        return None,None


    def write_eigenval(self, filename='EIGENVAL'):
        """
        Write interpolated eigen values to file in EIGENVAL format.
        """
        
        with open(filename, 'w') as f:

            # header
            #  WRITE(22,'(4I5)') T_INFO%NIONS,T_INFO%NIONS,DYN%NBLOCK*DYN%KBLOCK,WDES%ISPIN
            #  WRITE(22,'(5E15.7)') &
            #  &         AOMEGA,((LATT_CUR%ANORM(I)*1E-10_q),I=1,3),DYN%POTIM*1E-15_q
            #  WRITE(22,*) DYN%TEMP
            #  WRITE(22,*) ' CAR '
            #  WRITE(22,*) INFO%SZNAM1
            #  WRITE(22,'(3I5)') NINT(INFO%NELECT),KPOINTS%NKPTS,WDES%NB_TOT
            nat = -1
            f.write("%d %d %d %d\n"%(nat, nat, 1, self.ispin))
            f.write("\n")  # not read by vasp2boltz.py --- can be filled in later
            f.write("\n")  # not read by vasp2boltz.py --- can be filled in later
            f.write(" CAR\n")
            f.write(" Funz Madhattiness\n")
            f.write("%d %d %d\n"%(self.nelect, self.nkpts, self.nbands))

            # data block
            for isp in range(self.ispin):
                if self.ispin == 2:
                    f.write(' spin component %d\n'%(isp+1))
                for en,kpt,wgt in zip(self.energies, self.kpts, self.wgts):
                    f.write('\n')
                    for k in kpt:
                        f.write(' %16.8f'%k)
                    f.write('  %16.8f\n'%wgt)
                    for ib,e in enumerate(en):
                        f.write('%4d %12.4f\n'%(ib+1,e))
        return


    def get(self, name):
        """Get attribute."""
        assert name in self.names
        return self.data[name]


    def set(self, name, value):
        """Set attribute."""
        assert name in self.names
        self.data[name] = value


    def delete(self, name):
        """Delete attribute."""
        assert name in self.names
        self.data[name] = None


    def myproperty(name, doc):
        """Helper function to easily create class attribute property."""
        # procedure adapted from ASE
        def getter(self):
            return self.get(name)

        def setter(self, value):
            self.set(name, value)

        def deleter(self):
            self.delete(name)

        return property(getter, setter, deleter, doc)


    energies = myproperty('energies', 'Eigen energies')
    velocities = myproperty('velocities', 'Group velocities')
    kpts = myproperty('kpts', 'k-point coordinates')
    nkpts = myproperty('nkpts', 'number of k-points NKPTS')
    nbands = myproperty('nbands', 'number of bands NBANDS')
    nelect = myproperty('nelect', 'number of electrons NELECT')
    ispin = myproperty('ispin', 'ISPIN variable')
    kinter = myproperty('kinter', 'KINTER variable')
    dos = myproperty('dos', 'density of states')
    kptgrid_type = myproperty('kptgrid_type', 'type of k-point grid')
    kptgrid_divisions = myproperty('kptgrid_divisions', 'divisions of k-point grid')
    efermi_interpolated = myproperty('efermi_interpolated', 'Fermi energy from interpolated k-point grid')

