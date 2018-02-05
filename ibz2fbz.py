#!/usr/bin/env python3

class ibz2fbz:
    "Interpolated and non-interpolated mappings for IBZ to full BZ"

    def __init__(self,filename='OUTCAR'):
        self.filename = filename
        self.read_data()

    def read_data(self):
        """Import OUTCAR type file.
        Reads unitcell, atom positions, energies, and forces from the OUTCAR file
        and attempts to read constraints (if any) from CONTCAR/POSCAR, if present. 
        """
        import os
        import numpy as np

        if isinstance(self.filename, str):
            f = open(self.filename)
        nitpi2f = None
        itpi2f = None
        i2f = []

        for n,line in enumerate(f):
            if 'k-points in 1st BZ' in line:
                #initiate temporary array
                self.nitpi2f = np.array(i2f)
                i2f = []

            if 'KPOINTS_INTER' in line:
                # save the first array
                self.nitpi2f = np.array(i2f)

            if 't-inv' in line:
                # read the mapping indeces
                i2f.append(int(line.split()[4])-1)

        self.itpi2f = np.array(i2f)
