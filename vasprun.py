"""
This module contains functionality for reading and writing an ASE
Atoms object in VASP POSCAR format.

"""

import os
from ase.utils import basestring

def __get_xml_parameter(par):
    """An auxillary function that enables convenient extraction of
    parameter values from a vasprun.xml file with proper type
    handling.

    """

    def to_bool(b):
        if b == 'T':
            return True
        else:
            return False

    to_type = {'int': int,
               'logical': to_bool,
               'string': str,
               'float': float}

    text = par.text
    if text is None:
        text = ''

    # Float parameters do not have a 'type' attrib
    var_type = to_type[par.attrib.get('type', 'float')]

    if par.tag == 'v':
        return map(var_type, text.split())
    else:
        return var_type(text.strip())


def read_vasprun_xml(filename='vasprun.xml', index=-1):
    """Parse vasprun.xml file.

    Reads unit cell, atom positions, energies, forces, and constraints
    from vasprun.xml file
    """

    import numpy as np
    import xml.etree.ElementTree as ET
    from ase import Atoms
    from ase.constraints import FixAtoms, FixScaled
    from ase.calculators.singlepoint import (SinglePointDFTCalculator,
                                             SinglePointKPoint)
    from ase.units import GPa
    from collections import OrderedDict

    tree = ET.iterparse(filename, events=['start', 'end'])

    atoms_init = None
    calculation = []
    ibz_kpts = None
    parameters = OrderedDict()

    try:
        for event, elem in tree:

            if event == 'end':
                if elem.tag == 'kpoints':
                    for subelem in elem.iter(tag="generation"):
                        kpts_params = OrderedDict()
                        parameters['kpoints_generation'] = kpts_params
                        for par in subelem.iter():
                            if par.tag in ['v', 'i']:
                                parname = par.attrib['name'].lower()
                                kpts_params[parname] = __get_xml_parameter(par)

                    kpts = elem.findall("varray[@name='kpointlist']/v") #ibz kpoints, nonscaled
                    ibz_kpts = np.zeros((len(kpts), 3))

                    for i, kpt in enumerate(kpts):
                        ibz_kpts[i] = [float(val) for val in kpt.text.split()] 

                elif elem.tag == 'parameters':
                    for par in elem.iter():
                        if par.tag in ['v', 'i']:
                            parname = par.attrib['name'].lower()
                            parameters[parname] = __get_xml_parameter(par)

                elif elem.tag == 'atominfo':
                    species = []

                    for entry in elem.find("array[@name='atoms']/set"):
                        species.append(entry[0].text.strip())

                    natoms = len(species)

                elif (elem.tag == 'structure' and
                      elem.attrib.get('name') == 'initialpos'):
                    cell_init = np.zeros((3, 3), dtype=float)

                    for i, v in enumerate(elem.find(
                            "crystal/varray[@name='basis']")):
                        cell_init[i] = np.array([
                            float(val) for val in v.text.split()])

                    scpos_init = np.zeros((natoms, 3), dtype=float)

                    for i, v in enumerate(elem.find(
                            "varray[@name='positions']")):
                        scpos_init[i] = np.array([
                            float(val) for val in v.text.split()])

                    constraints = []
                    fixed_indices = []

                    for i, entry in enumerate(elem.findall(
                            "varray[@name='selective']/v")):
                        flags = (np.array(entry.text.split() ==
                                          np.array(['F', 'F', 'F'])))
                        if flags.all():
                            fixed_indices.append(i)
                        elif flags.any():
                            constraints.append(FixScaled(cell_init, i, flags))

                    if fixed_indices:
                        constraints.append(FixAtoms(fixed_indices))

                    atoms_init = Atoms(species,
                                       cell=cell_init,
                                       scaled_positions=scpos_init,
                                       constraint=constraints,
                                       pbc=True)

            elif event == 'start' and elem.tag == 'calculation':
                calculation.append(elem)

    except ET.ParseError as parse_error:
        if atoms_init is None:
            raise parse_error
        if calculation[-1].find('energy') is None:
            calculation = calculation[:-1]
        if not calculation:
            yield atoms_init

    if calculation:
        if isinstance(index, int):
            steps = [calculation[index]]
        else:
            steps = calculation[index]
    else:
        steps = []

    for step in steps:
        # Workaround for VASP bug, e_0_energy contains the wrong value
        # in calculation/energy, but calculation/scstep/energy does not
        # include classical VDW corrections. So, first calculate
        # e_0_energy - e_fr_energy from calculation/scstep/energy, then
        # apply that correction to e_fr_energy from calculation/energy.
        lastscf = step.findall('scstep/energy')[-1]

        de = (float(lastscf.find('i[@name="e_0_energy"]').text) -
              float(lastscf.find('i[@name="e_fr_energy"]').text))

        free_energy = float(step.find('energy/i[@name="e_fr_energy"]').text)
        energy = free_energy + de

        cell = np.zeros((3, 3), dtype=float)
        for i, vector in enumerate(step.find(
                'structure/crystal/varray[@name="basis"]')):
            cell[i] = np.array([float(val) for val in vector.text.split()])

        scpos = np.zeros((natoms, 3), dtype=float)
        for i, vector in enumerate(step.find(
                'structure/varray[@name="positions"]')):
            scpos[i] = np.array([float(val) for val in vector.text.split()])

        forces = None
        fblocks = step.find('varray[@name="forces"]')
        if fblocks is not None:
            forces = np.zeros((natoms, 3), dtype=float)
            for i, vector in enumerate(fblocks):
                forces[i] = np.array([float(val)
                                      for val in vector.text.split()])

        stress = None
        sblocks = step.find('varray[@name="stress"]')
        if sblocks is not None:
            stress = np.zeros((3, 3), dtype=float)
            for i, vector in enumerate(sblocks):
                stress[i] = np.array([float(val)
                                      for val in vector.text.split()])
            stress *= -0.1 * GPa
            stress = stress.reshape(9)[[0, 4, 8, 5, 2, 1]]

        efermi = step.find('dos/i[@name="efermi"]')
        if efermi is not None:
            efermi = float(efermi.text)

        kpoints = []
        # ------------------------------------------------------------------------------------------------

        #kpts_itp = elem.findall("varray[@name='kpointlist']/v") #ibz kpoints, interpolated
        #ibz_kpts_itp = np.zeros((len(kpts_itp), 3))
        #for i, kpt in enumerate(kpts):
        #    ibz_kpts[i] = [float(val) for val in kpt.text.split()]
        
        print(len(ibz_kpts))
                                                                        
        # wrong number of kp, we not in ibz anymore 
        for ikpt in range(1, len(ibz_kpts) + 1):
            kblocks = step.findall(
                'eigenvalues/electronvelocities/eigenvalues/array/set/set/set[@comment="kpoint %d"]' % ikpt)
                #'eigenvalues/array/set/set/set[@comment="kpoint %d"]' % ikpt)
                #'eigenvalues[@comment="interpolated_ibz"]/array/set/set/set[@comment="kpoint %d"]' % ikpt)
            #print(kblocks)
            if kblocks is not None:
                for i, kpoint in enumerate(kblocks):
                    eigenvals = kpoint.findall('r')
                    eps_n = np.zeros(len(eigenvals))
                    f_n = np.zeros((len(eigenvals),3))
                    for j, val in enumerate(eigenvals):
                        val = val.text.split()
                        eps_n[j] = float(val[0])
                        f_n[j] = [float(val[1]),float(val[2]),float(val[3])]
                        print(val)
                    if len(kblocks) == 1:
                        f_n *= 2
                    kpoints.append(SinglePointKPoint(1, 0, ikpt, eps_n, f_n))
        if len(kpoints) == 0:
            kpoints = None

        if ibz_kpts is not None:
            bz_kpts = np.dot(ibz_kpts, cell)

        atoms = atoms_init.copy()
        atoms.set_cell(cell)
        atoms.set_scaled_positions(scpos)
        atoms.set_calculator(
            SinglePointDFTCalculator(atoms, energy=energy, forces=forces,
                                     stress=stress, free_energy=free_energy,
                                     bz_kpts=bz_kpts, ibz_kpts=ibz_kpts,
                                     eFermi=efermi))
        atoms.calc.name = 'vasp'
        atoms.calc.kpts = kpoints
        atoms.calc.parameters = parameters
        yield atoms

