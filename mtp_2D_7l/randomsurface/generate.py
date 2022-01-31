#%%
import importlib

from numpy.core.fromnumeric import transpose
import aseAtoms2cfg
import myread
importlib.reload(aseAtoms2cfg)
importlib.reload(myread)

import copy
from random import random
import numpy as np
# seed random number generator
from aseAtoms2cfg import atoms2cfg
from ase.build import graphene_nanoribbon
from ase import Atoms, Atom
from ase.calculators.lammpslib import LAMMPSlib
import os

import random

def getAmplifiedRand(amplitud=1):
    return amplitud * ( random() - 0.5 )
#
def perturbPosition(position, amplitud):
    position += getAmplifiedRand(amplitud)
    return position
#
def perturbAllPositions(positions, amplitud):
    for r in positions:
        r = perturbPosition(r, amplitud)
    #
    return positions
#

def getCfgs_rattleAll(atoms, mindist, nCfgs, stdev):
    ats = copy.deepcopy(atoms)
    cfgString = ""
    countCfgs = 0
    for i in range(100000): # try 1000 times, until reaching the desired number of configurations `nCfgs`
        ats.rattle(stdev, seed=i) # see rattle() in https://wiki.fysik.dtu.dk/ase/ase/atoms.html

        # Wrap positions to unit cell.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.set_scaled_positions
        # See ase.geometry.wrap_positions() in https://wiki.fysik.dtu.dk/ase/ase/geometry.html
        ats.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

        if getMinDist(ats) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(ats) 
            countCfgs += 1
            if countCfgs == nCfgs:
                break
            #
        #
        ats = copy.deepcopy(atoms)
        #
        # the following lines does not get random values for x, y, z 
        # newPositions = perturbAllPositions(atoms.get_positions(), amplitud)
        # atoms.set_positions(newPositions)
        # cfgString += atoms2cfg(atoms)
        # atoms = copy.deepcopy(atoms_original)
    #
    print("I generated " + str(countCfgs) + " cfgs.")
    f = open("to_relax.cfg", "a")
    f.write(cfgString)
    f.close()
#
def _rattleAtom(atom, stdev=0.001, seed=None, rng=None):
    """Randomly displace position of an atom of type `Atom`.

    This method adds a random displacement to the atomic position,
    taking a possible constraint into account??.  The random numbers are
    drawn from a normal distribution of standard deviation stdev.

    For a parallel calculation, it is important to use the same
    seed on all processors!  """

    if seed is not None and rng is not None:
        raise ValueError('Please do not provide both seed and rng.')

    if rng is None:
        if seed is None:
            seed = 42
        rng = np.random.RandomState(seed)
    pos = atom.position
    atom.position = pos + rng.normal(scale=stdev, size=3)
#

def getCfgsWithRandAtom(atoms, atom, mindist, nCfgs, stdev):
    ats = copy.deepcopy(atoms)
    pos_original   = copy.deepcopy(atom.position)
    cfgString = ""
    countCfgs = 0
    for i in range(5000): # try 1000 times, until reaching the desired number of configurations `nCfgs`
        _rattleAtom(atom, stdev, seed=i)
        ats.append(atom)
        
        # Wrap positions to unit cell.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.set_scaled_positions
        # See ase.geometry.wrap_positions() in https://wiki.fysik.dtu.dk/ase/ase/geometry.html
        ats.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)
        
        if getMinDist(ats) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(ats) 
            countCfgs += 1
            if countCfgs == nCfgs:
                atom.position = copy.deepcopy(pos_original)
                atoms2qe(ats, "workdir/qe.in")
                break
            #
        #
        ats = copy.deepcopy(atoms)
        atom.position = copy.deepcopy(pos_original)
    #
    #     atom.position = perturbPosition(atom.position, amplitud)
    #     atoms.append(atom)
    #     cfgString += atoms2cfg(atoms)
    #     atoms = copy.deepcopy(atoms_original)
    #     atom.position = pos_original
    # #
    print("I generated " + str(countCfgs) + " cfgs.")
    f = open("to_relax.cfg", "a")
    f.write(cfgString)
    f.close()
#

def getCfgsWithRandAppendedAtomlist(atoms, listAtom2append, mindist, nCfgs, stdev):
    ats = copy.deepcopy(atoms)

    list_pos_original = []
    for atom in listAtom2append:
        list_pos_original.append( copy.deepcopy(atom.position) )
    #
    cfgString = ""
    countCfgs = 0
    for i in range(5000): # try 1000 times, until reaching the desired number of configurations `nCfgs`
        
        for atom in listAtom2append:
            _rattleAtom(atom, stdev, seed=i)
            ats.append(atom)
        #
        # Wrap positions to unit cell.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.set_scaled_positions
        # See ase.geometry.wrap_positions() in https://wiki.fysik.dtu.dk/ase/ase/geometry.html
        ats.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)
        
        if getMinDist(ats) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(ats) 
            countCfgs += 1
            if countCfgs == nCfgs:
                for (i, atom) in enumerate(listAtom2append):
                    atom.position = copy.deepcopy(list_pos_original[i])
                #
                atoms2qe(ats, "workdir/qe.in")
                break
            #
        #
        ats = copy.deepcopy(atoms)

        for (i, atom) in enumerate(listAtom2append):
            atom.position = copy.deepcopy(list_pos_original[i])
        #
    # 
    print("I generated " + str(countCfgs) + " cfgs.")
    f = open("to_relax.cfg", "a")
    f.write(cfgString)
    f.close()
#

def moreCfgs():
    atoms = get_3_AGNR72(cc_dist, vacuum)
    heighList = np.arange(-9.0, 9.0, 0.1)
    cfgString = ""

    for h in heighList:
        pO1 = [0,h,0] + (atoms[7].position + atoms[8].position) / 2
        atoms.append(Atom("O", position=pO1))
        atoms.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

        if getMinDist(atoms) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(atoms)
        #
        atoms = get_3_AGNR72(cc_dist, vacuum)
    #
    for h in heighList:
        pO1 = [0,h,0] + (atoms[7].position + atoms[8].position) / 2
        pO2 = [0,h,0] + (atoms[19].position + atoms[20].position) / 2
        atoms.append(Atom("O", position=pO1))
        atoms.append(Atom("O", position=pO2))
        atoms.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

        if getMinDist(atoms) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(atoms)
        #
        atoms = get_3_AGNR72(cc_dist, vacuum)
    #
    for h in heighList:
        pO1 = [0,h,0] + (atoms[7].position + atoms[8].position) / 2
        pO2 = [0,h,0] + (atoms[19].position + atoms[20].position) / 2
        pO3 = [0,h,0] + (atoms[31].position + atoms[32].position) / 2        
        atoms.append(Atom("O", position=pO1))
        atoms.append(Atom("O", position=pO2))
        atoms.append(Atom("O", position=pO3))
        atoms.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

        if getMinDist(atoms) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(atoms)
        #
        atoms = get_3_AGNR72(cc_dist, vacuum)
    #
    f = open("to_relax.cfg", "w")
    f.write(cfgString)
    f.close()
#

def saveCfg(cfgString, filename="to_relax.cfg", option="w"):
    f = open(filename, option)
    f.write(cfgString)
    f.close()
#

def moreCfgs2():
    atoms = get_2_AGNR42(cc_dist, vacuum)
    

    heighList = np.arange(-7.0, 7.0, 0.2)
    cfgString = ""
    
    a, b, c, _, _, _ = atoms.cell.cellpar()
    indices = [23, 6, 7, 14, 15, 19]
    import copy
    positions = [ copy.deepcopy(atoms[i].position) for i in indices ]

    for l in np.arange(-0.5, 1.0, 0.2): # C_Cdistance = 1.4, so 1.4-0.5 > 0
        new_c = c + l
        atoms.set_cell([a, b, new_c])

        j = 0
        for indx in indices:
            atoms[indx].position = positions[j] + [0, 0, l]
            j += 1
        #
        ats = copy.deepcopy(atoms)

        for h in heighList:
            pO1 = [0,h,0] + (ats[5].position + ats[6].position) / 2
            ats.append(Atom("O", position=pO1))
            ats.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

            if getMinDist(ats) > mindist: # disregard those structures with atoms too close
                cfgString += atoms2cfg(ats)
            #
            ats = copy.deepcopy(atoms)
        #
        for h in heighList:
            pO1 = [0,h,0] + (ats[5].position + ats[6].position) / 2
            pO2 = [0,h,0] + (ats[13].position + ats[14].position) / 2
            ats.append(Atom("O", position=pO1))
            ats.append(Atom("O", position=pO2))
            ats.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

            if getMinDist(ats) > mindist: # disregard those structures with atoms too close
                cfgString += atoms2cfg(ats)
            #
            ats = copy.deepcopy(atoms)
        #
    #
    ats = copy.deepcopy(atoms)
    #
    f = open("to_relax.cfg", "w")
    f.write(cfgString)
    f.close()
#

def getMinDist(atoms):
    distMatrix = atoms.get_all_distances()
    distances = [e for e in distMatrix.flatten() if e != 0.0]
    mindist1 = np.amin(distances)
    #
    distMatrix = atoms.get_all_distances(mic=True)
    distances = [e for e in distMatrix.flatten() if e != 0.0]
    mindist2 = np.amin(distances)
    #
    return min(mindist1, mindist2)
#

def getCfgs_alatSweep(atoms, mindist, nCfgs):
    ats = copy.deepcopy(atoms)
    a, b, c, _, _, _ = ats.cell.cellpar()
    positions = ats.get_positions()
    cfgString = ""
    countCfgs = 0
    for l in np.arange (-0.5, 1.5, 0.1): # C_Cdistance = 1.4, so 1.4-0.5 > 0
        # increase the unit cell along Z-axis
        ats.set_cell([a, b, c + l])
        
        # center z-positions in the unit cell:
        ats.set_positions(positions)
        ats.translate([0, 0, l / 2])

        if getMinDist(ats) > mindist: # disregard those structures with atoms too close
            cfgString += atoms2cfg(ats)
            countCfgs += 1
            if countCfgs == nCfgs:
                break
            #
        #
    #
    print("I generated " + str(countCfgs) + " cfgs.")
    f = open("to_relax.cfg", "a")
    f.write(cfgString)
    f.close()
#

def atoms2qe(atoms, QEin):
    from ase.io.espresso import write_espresso_in
    f = open(QEin, "w")
    write_espresso_in(fd=f, atoms=atoms, input_data=None, pseudopotentials=None, kspacing=None, kpts=None, koffset=(0, 0, 0), crystal_coordinates=False)
    f.close()
#

def getAGNRwire(cc_dist, vacuum): # wire
    return graphene_nanoribbon(2/2, 1, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def get_1_AGNR32(cc_dist, vacuum):
    return graphene_nanoribbon(3/2, 1, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def get_1_AGNR42(cc_dist, vacuum):
    return graphene_nanoribbon(4/2, 1, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def get_1_AGNR52(cc_dist, vacuum):
    return graphene_nanoribbon(5/2, 1, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def get_1_AGNR62(cc_dist, vacuum):
    return graphene_nanoribbon(6/2, 1, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def get_2_AGNR42(cc_dist, vacuum):
    return graphene_nanoribbon(4/2, 2, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def get_3_AGNR72(cc_dist, vacuum):
    return graphene_nanoribbon(7/2, 3, type='armchair', saturated=True, C_H=1.1, C_C=cc_dist, vacuum=vacuum,  magnetic=True, initial_mag=1.12)
#

def getInitMagnets(atoms): # It avoids HYDROGENS!!
    nAtoms = len(atoms)
    initial_magnetic_moments = np.ones(nAtoms)
    for i in range(nAtoms):
        if i % 2 == 1:
            if atoms[i].symbol != 'H':
                initial_magnetic_moments[i] = -1
                # atoms[i].symbol = "B" # <<-- used to visualize on Xcrysden that near neighbors are actually initialized with different spin, and names: C and C1
    # 
    return initial_magnetic_moments
#

def getMindistOxygenToCarbons(atoms):
    nAtoms = len(atoms)

    oxygenIndx = nAtoms - 1
    if atoms[oxygenIndx].symbol == "O":
        #
        # Return distances of atom No.i with a list of atoms. https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_distances
        # Oxygen indx is nAtoms-1, the last one (python begins indx in zero 0, instead of 1)
        distances = atoms.get_distances( nAtoms-1, np.arange(nAtoms) )
        mindistO2C = min( [d for d in distances if d != 0.0] )
        return mindistO2C
    else:
        return None
    #
#

# source: https://github.com/nglviewer/nglview/issues/900
def rotate_view(view, x=0, y=0, z=0, degrees=True):
    radians = 1
    if degrees: radians = math.pi / 180
    view.control.spin([1, 0, 0], x*radians)
    view.control.spin([0, 1, 0], y*radians)
    view.control.spin([0, 0, 1], z*radians)
#

def visualize(atoms, background="green", enumerar=False):
    # screenshot it and go to https://onlinepngtools.com/create-transparent-png 
    # to make green background transparent

    import nglview as nv
    # nv.show_ase(atoms)
    v = nv.show_ase(atoms)
    v.background = background
    rotate_view(v, 90, 45, 15)

    if enumerar:
        for (i, a) in enumerate(atoms):
            a, b, c = a.position
            v.shape.add('text', [a, b, c], [0.1, 0.1, 0.1], 2.5, str(i))
        #
    #
    return v
#

def dict2atoms(dictCfg):
    species = dictCfg['species']
    cell = dictCfg['cell']
    positions = dictCfg['positions']
    atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=(0,0,1))
    return atoms
#

def getLessEnergeticStrFromRelaxed(filename, dictOfSpecies):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    from cfg2ase import read_cfgs

    dictCfgs = read_cfgs(filename, dictOfSpecies)

    energies = [ dictCfgs[i]['outputs']['energy'] for i in range(len(dictCfgs))]
    imin = np.argmin(energies)
    gs = dict2atoms( dictCfgs[imin] )
    sortIndx = np.argsort(energies)
    return gs, energies[imin], dictCfgs, sortIndx
#

def belongs(x, x1, x2):
    return ( (x1 < x) and (x < x2) )
#

def getIndexAbove(indxRef, atoms):
    indices2translate = []
    indices2insertOxygen = []
    zRef = atoms[indxRef].position[2] # z-component in nanoribbon
    for (i, atom) in enumerate(atoms):
        z = atom.position[2]
        if z > zRef:
            indices2translate.append(i)
        #
        # find other carbon atoms at the same height:
        if atom.symbol == "C":
            if belongs(z, zRef - 0.1, zRef + 0.1):
                indices2insertOxygen.append(i)
            #
        #
    #
    return indices2translate, indices2insertOxygen
#

# def insertOxygensAbove(indxRef, atoms, h=0.1, l=0.9):
    
#     a, b, c, _, _, _ = atoms.cell.cellpar()

#     indices2translate, indices2insertOxygen = getIndexAbove(indxRef, atoms)

#     positions = [ copy.deepcopy(atoms[i].position) for i in indices2translate ]
#     # l = 0.9
#     new_c = c + l
#     atoms.set_cell([a, b, new_c])
#     j = 0
#     for indx in indices2translate:
#         atoms[indx].position = positions[j] + [0, 0, l]
#         j += 1
#     #
#     # h = 0.1

#     for i in indices2insertOxygen:
#         if atoms.get_distance(i, i + 1) < 1.5:
#             p = [0,h,0] + (atoms[i].position + atoms[i+1].position) / 2
#             atoms.append(Atom("O", position=p))
#         else:
#             x, y, z = atoms[i].position
#             p = [0,h,0] + np.array([ x, y, z + 1.4/2 ])
#             atoms.append(Atom("O", position=p))
#     #
#     atoms.wrap(pbc=[True, True, True])
#     return atoms
#

def insertOxygensAbove(indxRef, atoms, h=0.1, l=0.9):
    
    a, b, c, _, _, _ = atoms.cell.cellpar()

    indices2translate, indices2insertOxygen = getIndexAbove(indxRef, atoms)
    
    import copy
    positions = [ copy.deepcopy(atoms[i].position) for i in indices2translate ]
    # l = 0.9
    new_c = c + l
    atoms.set_cell([a, b, new_c])
    j = 0
    for indx in indices2translate:
        atoms[indx].position = positions[j] + [0, 0, l]
        j += 1
    #
    # h = 0.1
    for i in indices2insertOxygen:
        if atoms.get_distance(i, i + 1) < 1.5 + l:
            p = [0,h,0] + (atoms[i].position + atoms[i+1].position) / 2
            atoms.append(Atom("O", position=p))
        else:
            x, y, z = atoms[i].position
            p = [0,h,0] + np.array([ x, y, z + 1.4/2 + l/2 ])
            atoms.append(Atom("O", position=p))
    #
    atoms.wrap(pbc=[True, True, True])
    return atoms
#


def insertOxygensAbove2(indxRef, atoms, h=0.1, l=0.9):
    a, b, c, _, _, _ = atoms.cell.cellpar()

    indices2translate, indices2insertOxygen = getIndexAbove(indxRef, atoms)
    
    import copy
    positions = [ copy.deepcopy(atoms[i].position) for i in indices2translate ]
    # l = 0.9
    new_c = c + l
    atoms.set_cell([a, b, new_c])
    j = 0
    for indx in indices2translate:
        atoms[indx].position = positions[j] + [0, 0, l]
        j += 1
    #
    # h = 0.1
    dZ = (atoms[3+1].position[2] - atoms[3].position[2]) / 2
    
    for i in indices2insertOxygen:
        if atoms.get_distance(i, i + 1) < 1.5 + l:
            p = [0,h,0] + (atoms[i].position + atoms[i+1].position) / 2
            atoms.append(Atom("O", position=p))
        else:
            x, y, z = atoms[i].position
            # p = [0,h,0] + np.array([ x, y, z + cc_dist/2 + l/2 ])
            p = [0,h,0] + np.array([ x, y, z + dZ ])
            atoms.append(Atom("O", position=p))
    #
    atoms.wrap(pbc=[True, True, True])
    return atoms
#
#%%

def insertAtomAbove(specie, indxRef, atoms, h=0.1, l=0.9, ):
    a, b, c, _, _, _ = atoms.cell.cellpar()

    indices2translate, indices2insertOxygen = getIndexAbove(indxRef, atoms)
    
    import copy
    positions = [ copy.deepcopy(atoms[i].position) for i in indices2translate ]
    # l = 0.9
    new_c = c + l
    atoms.set_cell([a, b, new_c])
    j = 0
    for indx in indices2translate:
        atoms[indx].position = positions[j] + [0, 0, l]
        j += 1
    #
    # h = 0.1
    for i in indices2insertOxygen:
        if atoms.get_distance(i, i + 1) < 1.5 + l:
            p = [0,h,0] + (atoms[i].position + atoms[i+1].position) / 2
            atoms.append(Atom(specie, position=p))
        else:
            x, y, z = atoms[i].position
            p = [0,h,0] + np.array([ x, y, z + 1.4/2 + l/2 ])
            atoms.append(Atom(specie, position=p))
    #
    atoms.wrap(pbc=[True, True, True])
    return atoms


def get_calculator():
    # from ase.calculators.lammpslib import LAMMPSlib
    import numpy as np
    cmds = ["pair_style mlip mlip.ini",  "pair_coeff * * "]
    # mylammps = LAMMPSlib(lmpcmds=cmds, log_file='lammpslog',keep_alive=True)
    mylammps = LAMMPSlib(lmpcmds=cmds, atom_types={ 'C':1, 'O':2, 'B':3, 'N':4 } , log_file='lammpslog',keep_alive=True)
    return mylammps
#

def relax(atoms):
    command = "rm -f relaxed.cfg_0 selection.log_0" # `-f` to avoid stopping if file not found
    os.system(command)

    # files to be present: mlip.ini, pot.mtp, state.mvs, relax.ini
    saveCfg(atoms2cfg(atoms)) # convert atoms to cfg, and write it into to_relax.cfg
    command = "mlp relax relax.ini --cfg-filename=to_relax.cfg --min-dist=0.5 --save-relaxed=relaxed.cfg"
    os.system(command)

    # load the relaxed.cfg_0:
    dictOfSpecies = {"0":"C", "1":"H", "2":"O"}
    filename = "relaxed.cfg_0"
    gs, energy, dictCfgs, sortindx = getLessEnergeticStrFromRelaxed(filename, dictOfSpecies)
    print("energy of MTP-relaxed structure: ", energy)
    return gs
#
import math
def rotar(x, y, angleRad):
    x0 = x
    y0 = y
    x = (x0 * math.cos(angleRad)) - ( y0 * math.sin(angleRad) )
    y = (x0 * math.sin(angleRad)) + ( y0 * math.cos(angleRad) )
    return x, y

def findAtomsRangeOxygens(atoms, listInflectionPoints):
    inflPos = [ atoms[j].position for j in listInflectionPoints ]
    listGroups = []
    m = len(inflPos) - 1
    for i in range(m):
        group = []
        z0 = inflPos[i][2]
        y0 = inflPos[i][1]
        #
        zf = inflPos[i + 1][2]
        #
        for (k, atom) in enumerate(atoms):
            x, y, z = atom.position
            if ( (z0 < z) and (z <= zf) ):
                group.append(k)
            elif (i + 1 == m) and (z > zf): ## I added this
                group.append(k) ## I added this
            # ## I added this
            #
        #
        listGroups.append(group)
    #
    return listGroups
#

def deg2Rad(deg):
    return (deg / 180.0) * math.pi


def bend(atoms, listInflectionPoints, bigAngleDeg):
    a, b, c, _, _, _ = atoms.cell.cellpar()
    listGroups = findAtomsRangeOxygens(atoms, listInflectionPoints)
    listPosRef0 = [ copy.deepcopy(atoms[i].position) for i in listInflectionPoints ]
    pos_gp1_0 = copy.deepcopy(atoms[listInflectionPoints[0 + 1]].position)
    rad = deg2Rad( 90 - (bigAngleDeg)/2 )
    for (g, group) in enumerate(listGroups):
        # if g <= 2:
        # print("g=", g)
        angleRad = -((-1)**(g)) * rad
        posRef  = atoms[listInflectionPoints[g]].position
        
        # print(listInflectionPoints[g])

        # print(posRef)
        for i in group:
            # if i == 15:
            #     print(atoms[i].position)

            x, y, z = atoms[i].position - posRef
            # print(i, x, y, z)
            z, y = rotar(z, y, angleRad)
            # print(i, x, y, z)
            atoms[i].position = [x, y, z] + posRef
            # print(i)
            # if i == 15:
            #     print(atoms[i].position)
        #
    
        if (g + 1) <= len(listInflectionPoints) - 2:
            # pos_gp1_0 = listPosRef0[g + 1]
            pos_gp1   = atoms[listInflectionPoints[g + 1]].position
            # print("....")
            # print(g+1, listInflectionPoints[g + 1])
            # print(pos_gp1_0)
            # print(pos_gp1)
            # print("---")
            deltaR  = pos_gp1 - pos_gp1_0
            for gg in range(g + 1, len(listGroups)):                
                for j in listGroups[gg]:
                    # print(gg, j, deltaR[1], deltaR[2])
                    atoms[j].position += [0.0, deltaR[1], deltaR[2]]
        #
        if (g + 2) <= len(listInflectionPoints) - 1:
            pos_gp1_0 = copy.deepcopy(atoms[listInflectionPoints[g + 2]].position)
    #
    # print("old cell: ", a, b, c)
    new_c = c * math.cos(rad)
    # print("new cell: ", a, b, new_c)
    atoms.set_cell([a, b, new_c])
    return atoms
#

def getBendedAtoms(nLines=3, sheet=False, deg=100):
    cc_dist = 1.42
    vacuum = 9.0
    atoms = graphene_nanoribbon(nLines/2, 4, type='armchair', saturated=False, C_C=cc_dist, vacuum=vacuum, sheet=sheet)
    nAtoms = len(atoms)
    insertOxygensAbove(3, atoms, h=0.0, l=cc_dist)
    insertOxygensAbove(11, atoms, h=0.0, l=cc_dist)
    nAtoms2 = len(atoms)
    listInflectionPoint = [0, nAtoms, nAtoms2 - 1, 15]
    bend(atoms, listInflectionPoints=listInflectionPoint, bigAngleDeg=deg)
    return atoms
    #


def getBendedAtomsSpecies(nLines=3, sheet=False, deg=100, specieA='O', specieB='O'):
    cc_dist = 1.42
    vacuum = 9.0
    atoms = graphene_nanoribbon(nLines/2, 4, type='armchair', saturated=False, C_C=cc_dist, vacuum=vacuum, sheet=sheet)
    nAtoms = len(atoms)
    insertAtomAbove(specieA, 3, atoms, h=0.0, l=cc_dist )
    insertAtomAbove(specieB, 11, atoms, h=0.0, l=cc_dist )
    nAtoms2 = len(atoms)
    listInflectionPoint = [0, nAtoms, nAtoms2 - 1, 15]
    bend(atoms, listInflectionPoints=listInflectionPoint, bigAngleDeg=deg)
    return atoms
    #

def compareDFT_MTP():
    import matplotlib.pyplot as plt
    cvals = []
    energiesDFT = []
    energiesMTP = []
    #
    # PLOT CELLZ VS ENERGIES
    dictOfSpecies = {"0":"C", "1":"O"}
    #filename = "trainrelaxed.cfg"
    filename = "trainDFT.cfg"
    #
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    from cfg2ase import read_cfgs
    #
    dictCfgs = read_cfgs(filename, dictOfSpecies)
    #
    for i in range(len(dictCfgs)):
        energiesDFT.append(dictCfgs[i]['outputs']['energy'])
        cvals.append(dictCfgs[i]["cell"][2][2])
        #
        d = dictCfgs[i]
        species = d['species']
        cell = d['cell']
        positions = d['positions']
        # pbc=(1,0,1) VERY IMPORTANT!!!
        atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=(1,0,1))
        atoms.set_calculator( get_calculator() )
        energiesMTP.append(atoms.get_potential_energy())
        plt.ylim(-3359, -3341)
        plt.xlim(14, 20)
    #

    import matplotlib.pyplot as plt
    plt.plot(cvals, energiesDFT, "o", cvals, energiesMTP, '*')
    # 
    return cvals, energiesDFT, energiesMTP
#          

def predictUsingMTPfromAlreadyDFTrelaxed(dictOfSpecies, pbc=(1,0,1), filename="dftrlxO.cfg_76old"):
    import matplotlib.pyplot as plt
    listA = []
    listE = []
    #
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    from cfg2ase import read_cfgs
    #
    dictionary = read_cfgs(filename, dictOfSpecies)
    #
    for d in dictionary:
        listA.append(d["cell"][2][2])
        #
        species   = d['species']
        cell      = d['cell']
        positions = d['positions']
        
        atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=pbc)
        atoms.set_calculator( get_calculator() )
        listE.append(atoms.get_potential_energy())
    #
    return listA, listE
#
#%%
#
 
def compareDFT_MTP_2():
    import matplotlib.pyplot as plt
    cvals = []
    energiesDFT = []
    energiesMTP = []
    #
    # PLOT CELLZ VS ENERGIES
    dictOfSpecies = {"0":"C", "1":"O"}
    #filename = "trainrelaxed.cfg"
    filename = "trainDFT.cfg"
    #
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    from cfg2ase import read_cfgs
    #
    dictCfgs = read_cfgs(filename, dictOfSpecies)
    #
    for i in range(len(dictCfgs)):
        energiesDFT.append(dictCfgs[i]['outputs']['energy'])
        cvals.append(dictCfgs[i]["cell"][2][2])
        #
        d = dictCfgs[i]
        species = d['species']
        cell = d['cell']
        positions = d['positions']
        # pbc=(1,0,1) VERY IMPORTANT!!!
        atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=(1,0,1))
        atoms.set_calculator( get_calculator() )
        energiesMTP.append(atoms.get_potential_energy())
        plt.ylim(-3359, -3341)
        plt.xlim(14, 20)
    #
    return np.array(energiesDFT), np.array(energiesMTP)
#

def plot(Yactual, Ypredict):
    import matplotlib.pyplot as plt
    #plt.plot(cvals, energiesDFT, "o", cvals, energiesMTP, '*')
    offset = -3358.6579
    ymin = -3358.6579
    ymax = max(Ypredict)
    plt.figure(1, figsize=(8, 8))
    plt.plot(Yactual - offset, Ypredict- offset, 'ro', markersize=12, mfc='none')
    plt.plot([ymin - offset,ymax - offset],[ymin - offset, ymax - offset], 'k', linewidth=2)
    plt.xlabel('Actual Energy', fontsize=15)
    plt.ylabel('Predicted Energy', fontsize=15)
    plt.xlim((ymin - offset, ymax- offset))
    plt.ylim((ymin - offset, ymax - offset))
    #plt.legend(['Density (MPDB)','Ideal Performance'], loc='best', fontsize=15)
    # y1 = y_test.values.tolist()
    # for i in range(len(y_test)):
    #     print(y1[i] - offset, y_predict[i] - offset)
    # # #
#

#%%

def dict2atoms2D(dictCfg):
    species = dictCfg['species']
    cell = dictCfg['cell']
    positions = dictCfg['positions']
    atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=(1,0,1))
    return atoms
#
def plotDFTvsMTPrelaxations():
	import sys
	sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
	from cfg2ase import read_cfgs
	#
	angles = []
	listZcell = []
	dictOfSpecies = {"0":"C", "1":"O"}
	filename = "trainrelaxed.cfg"
	#filename = "mtprelaxed.cfg"
	dictCfgs = read_cfgs(filename, dictOfSpecies)
	indices = range(len(dictCfgs))
	for i in indices:
		atoms = dict2atoms2D( dictCfgs[i] )
		a, b, c, _, _, _ = atoms.cell.cellpar()
		listZcell.append( c )
		angles.append( atoms.get_angle(3, 16, 4) )
	#
	# sort 
	sortedIndices = np.argsort(listZcell)
	#
	sortedListZcell = []
	sortedAngles = []
	energies = []
	for i in sortedIndices:
		sortedListZcell.append( listZcell[i] )
		sortedAngles.append( angles[i] )
		energies.append( dictCfgs[i]['outputs']['energy'] )
	#
	import matplotlib.pyplot as plt
	#plt.plot(sortedListZcell, energies,"o")
	for i in range(len(sortedListZcell)):
	    print(sortedListZcell[i], energies[i]+3358.6579, sortedAngles[i])
	#
#

def getAtoms2D_from_cfgAsInput(filename, i):
	import sys
	sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
	import cfg2ase
	import importlib
	importlib.reload(cfg2ase)
	from cfg2ase import read_cfgs_asinput
	#
	dictOfSpecies = {"0":"C", "1":"O"}
	dictCfgs = read_cfgs_asinput(filename, dictOfSpecies)
	#
	atoms = dict2atoms2D( dictCfgs[i] )
	return atoms
#

def getAtoms2D_from_cfg(filename, i):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    dictOfSpecies = {"0":"C", "1":"O"}
    dictCfgs = read_cfgs(filename, dictOfSpecies)
    #
    atoms = dict2atoms2D( dictCfgs[i] )
    energy = dictCfgs[i]['outputs']['energy']
    return atoms, energy
#

def getCurvesFromRelaxed(filename, dictOfSpecies, xlim=[14.0, 16.5]):
    #dictOfSpecies = {"0":"C", "1":"B", "2":"N"}
    #dictOfSpecies = {"0":"C", "1":"N"}
    #dictOfSpecies = {"0":"C", "1":"B"}
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    dictCfgs = read_cfgs(filename, dictOfSpecies)
    #
    alatMin = xlim[0]
    alatMax = xlim[1]
    degrees = []
    alats = []
    energies = []
    for i in range(len(dictCfgs)):
        atoms = dict2atoms2D( dictCfgs[i] )
        alat = dictCfgs[i]["cell"][2][2]
        if belongs(alat, alatMin, alatMax):
            degrees.append(atoms.get_angle(3, 16, 4))
            alats.append(alat)
            energies.append(dictCfgs[i]['outputs']['energy'])
        #
        else:
            print(i+1, alat, filename)
        #
    #
    emin = min(energies)
    # emin = 0.0
    nAtoms = len(atoms)
    energies = [(energies[i] - emin) / nAtoms for i in range(len(energies))]
    return alats, energies, degrees
#

def getCurvesFromRelaxed2(filename, dictOfSpecies, xlim=[14.0, 16.5]):
    #dictOfSpecies = {"0":"C", "1":"B", "2":"N"}
    #dictOfSpecies = {"0":"C", "1":"N"}
    #dictOfSpecies = {"0":"C", "1":"B"}
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    dictCfgs = read_cfgs(filename, dictOfSpecies)
    #
    alatMin = xlim[0]
    alatMax = xlim[1]
    degrees = []
    alats = []
    energies = []
    for i in range(len(dictCfgs)):
        atoms = dict2atoms2D( dictCfgs[i] )
        alat = dictCfgs[i]["cell"][2][2]
        if belongs(alat, alatMin, alatMax):
            degrees.append(atoms.get_angle(3, 16, 4))
            alats.append(alat)
            energies.append(dictCfgs[i]['outputs']['energy'])
        #
        else:
            print(i+1, alat, filename)
        #
    #
    #emin = min(energies)
    ## emin = 0.0
    #nAtoms = len(atoms)
    #energies = [(energies[i] - emin) / nAtoms for i in range(len(energies))]
    return alats, energies, degrees
#
#%%

def getCurveRibbon(nLines, minAngl=90, maxAngl=180, dAngle=5, xlim=[14.0, 16.5]):
    alatMin = xlim[0]
    alatMax = xlim[1]
    energies = []
    alats = []
    for deg in np.arange(minAngl, maxAngl, dAngle):
        atoms = getBendedAtomsSpecies(nLines=nLines, sheet=False, deg=deg, specieA='O', specieB='O')
        a, b, c, _, _, _ = atoms.cell.cellpar()
        if belongs(c, alatMin, alatMax):
            alats.append(c)
            atoms.set_calculator( get_calculator() )
            energies.append(atoms.get_potential_energy())
        #
    #
    natoms = len(atoms)
    emin = min(energies)
    # emin = 0.0
    energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
    # #
    return alats, energies
#

def getCurveRibbon2(nLines, minAngl=90, maxAngl=180, dAngle=5, xlim=[14.0, 16.5]):
    alatMin = xlim[0]
    alatMax = xlim[1]
    energies = []
    alats = []
    for deg in np.arange(minAngl, maxAngl, dAngle):
        atoms = getBendedAtomsSpecies(nLines=nLines, sheet=False, deg=deg, specieA='O', specieB='O')
        a, b, c, _, _, _ = atoms.cell.cellpar()
        if belongs(c, alatMin, alatMax):
            alats.append(c)
            atoms.set_calculator( get_calculator() )
            energies.append(atoms.get_potential_energy())
        #
    #
    # #
    return alats, energies
#

def getCurveRibbon_dopants(nLines, minAngl=90, maxAngl=180, dAngle=5, xlim=[14.0, 16.5], dopante1="B", dopante2="N"):
    alatMin = xlim[0]
    alatMax = xlim[1]
    energies = []
    alats = []
    for deg in np.arange(minAngl, maxAngl, dAngle):
        atoms = getBendedAtomsSpecies(nLines=nLines, sheet=False, deg=deg, specieA=dopante1, specieB=dopante2)
        a, b, c, _, _, _ = atoms.cell.cellpar()
        if belongs(c, alatMin, alatMax):
            alats.append(c)
            atoms.set_calculator( get_calculator() )
            energies.append(atoms.get_potential_energy())
        #
    #
    natoms = len(atoms)
    emin = min(energies)
    # emin = 0.0
    energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
    # #
    return alats, energies
#

def plotCurvesRibbons(listnlines=[3, 5, 7, 25], xlim=[14.0, 16.5]):
    import numpy as np
    listA = []
    listE = []
    minAngl, maxAngl, dAngle = [90, 180, 2]
    # listnlines = [3, 5, 10, 23] # np.arange(3,500, 200):
    for nLines in listnlines:
        alats, energies = getCurveRibbon(nLines, minAngl, maxAngl, dAngle, xlim)
        listA.append(alats)
        listE.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(listE)):
        plt.plot(listA[i], listE[i])
    #
    plt.ylim(0.0, 0.02)
    return listA, listE 
#

def plotCurvesRibbons2(listnlines=[3, 5, 7, 25], xlim=[14.0, 16.5]):
    import numpy as np
    listA = []
    listE = []
    minAngl, maxAngl, dAngle = [90, 180, 2]
    # listnlines = [3, 5, 10, 23] # np.arange(3,500, 200):
    for nLines in listnlines:
        alats, energies = getCurveRibbon2(nLines, minAngl, maxAngl, dAngle, xlim)
        listA.append(alats)
        listE.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(listE)):
        plt.plot(listA[i], listE[i])
    #
    return listA, listE 
#


# listA, listE = plotCurvesRibbons_dopants(listnlines=[3, 5, 7, 25], xlim=[14.0, 16.5], dopante1="B", dopante2="N")
def plotCurvesRibbons_dopants(listnlines=[3, 5, 7, 25], xlim=[14.0, 16.5], dopante1="B", dopante2="N"):
    import numpy as np
    listA = []
    listE = []
    minAngl, maxAngl, dAngle = [90, 180, 2]
    # listnlines = [3, 5, 10, 23] # np.arange(3,500, 200):
    for nLines in listnlines:
        alats, energies = getCurveRibbon_dopants(nLines, minAngl, maxAngl, dAngle, xlim, dopante1, dopante2)
        listA.append(alats)
        listE.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(listE)):
        plt.plot(listA[i], listE[i])
    #
    plt.ylim(0.0, 0.02)
    return listA, listE 
#

def generate_struct_of_nLines(atoms2D, nLines):
    nRepeat = (nLines + 1) / 2 # **** NLines MUST BE EVEN !!! @@@@@@@@@@@ *********
    nRepeat = int(nRepeat) 
    #
    width, b, alat, _, _, _ = atoms2D.cell.cellpar()

    new_chemical_symbols = atoms2D.get_chemical_symbols()
    new_positions = atoms2D.get_positions()

    # xmax = max( [ atoms2D.positions[i][0] ] for i in range(len(atoms2D)))[0]
    # new_structs_of_nLines = []

    lowx = min( [ atoms2D.positions[i][0] ] for i in range(len(atoms2D)))[0]
    xmax = lowx - 0.1
    xmin = xmax - ( width * (nLines - 2) /2 ) + 0.1

    for i in range(nRepeat-1):
        for atom in atoms2D:
            x, y, z = atom.position
            # print(x, width, x + ((i+1) * width), xmax)
            x -= ((i+1) * width)
            # if belongs(x, xmax - 0.1, width*nRepeat - width/2 ):

            if belongs(x, xmin, xmax):
                new_positions = np.append(new_positions, [[x, y, z]], axis=0)
                new_chemical_symbols = np.append(new_chemical_symbols, atom.symbol)
            #
        #
    #
    

    new_width = (width * nRepeat) + 20.0
    s = Atoms(
            symbols = new_chemical_symbols,
            cell = [new_width, b, alat],
            positions = new_positions,
            pbc = [0, 0, 1]
            )
    #

    newlowx = min( [ s.positions[i][0] ] for i in range(len(s)))[0]
    dx = lowx - newlowx
    for atom in s:
        atom.position[0] += dx
    #


    # s.wrap(pbc=[True, True, True])
    return s
#
#%%

def getCurves_2(list_nLines = [3, 5, 7, 25], filename2D = "dftrlxO.cfg", xlim=[14.0, 16.5]):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    # filename2D = "dftrlxO.cfg"
    dictOfSpecies = {"0":"C", "1":"O"}
    dictCfgs      = read_cfgs(filename2D, dictOfSpecies)

    alatMin, alatMax = xlim
    #
    structs = [ dict2atoms2D(d) for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    alats   = [ d['cell'][2][2] for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    sortedIndices = np.argsort(alats)
    #
    list_alats = []
    list_energies = []
    list_newStrs = []
    for nLines in list_nLines:
        alats = []
        energies = []
        for struct in structs:
            s = generate_struct_of_nLines(atoms2D=struct, nLines=nLines)
            list_newStrs.append(s)
            s.set_calculator( get_calculator() )
            natoms = len(s)
            # natoms = 1
            e = s.get_potential_energy()
            # print(e)
            energies.append(e)
            a, b, alat, _, _, _ = s.cell.cellpar()
            alats.append(alat)
        #
        emin = min(energies)
        # emin = 0.0
        energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
        #
        alats = [ alats[i] for i in sortedIndices]
        energies = [ energies[i] for i in sortedIndices]
        #
        list_alats.append(alats)
        list_energies.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(list_energies)):
        # plt.plot(list_alats[i], list_energies[i], '.')
        plt.plot(list_alats[i], list_energies[i])
    #
    return list_alats, list_energies, list_newStrs
#


def getCurves_2b(list_nLines = [3, 5, 7, 25], filename2D = "dftrlxO.cfg", xlim=[14.0, 16.5]):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    # filename2D = "dftrlxO.cfg"
    dictOfSpecies = {"0":"C", "1":"O"}
    dictCfgs      = read_cfgs(filename2D, dictOfSpecies)

    alatMin, alatMax = xlim
    #
    structs = [ dict2atoms2D(d) for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    alats   = [ d['cell'][2][2] for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    sortedIndices = np.argsort(alats)
    #
    list_alats = []
    list_energies = []
    list_newStrs = []
    for nLines in list_nLines:
        alats = []
        energies = []
        for struct in structs:
            s = generate_struct_of_nLines(atoms2D=struct, nLines=nLines)
            list_newStrs.append(s)
            s.set_calculator( get_calculator() )
            natoms = len(s)
            # natoms = 1
            e = s.get_potential_energy()
            # print(e)
            energies.append(e)
            a, b, alat, _, _, _ = s.cell.cellpar()
            alats.append(alat)
        #
        # emin = min(energies)
        # emin = 0.0
        # energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
        #
        alats = [ alats[i] for i in sortedIndices]
        energies = [ energies[i] for i in sortedIndices]
        #
        list_alats.append(alats)
        list_energies.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(list_energies)):
        # plt.plot(list_alats[i], list_energies[i], '.')
        plt.plot(list_alats[i], list_energies[i])
    #
    return list_alats, list_energies, list_newStrs
#
# alats2D, energies2D, degrees2D = getCurvesFromRelaxed("dftrlxO.cfg", dictOfSpecies, xlim=[14.0, 16.5])
# list_alats, list_energies = getCurves_2(list_nLines = [3, 5, 7], filename2D = "dftrlxO.cfg", xlim=[14.0, 16.5])
# for i in range(len(list_energies)):
#     plt.plot(list_alats[i], list_energies[i])
# #
# plt.plot(alats2D, energies2D, '+')
# # plt.ylim(-3358 -0.68 , -3358 -0.45)

def shorten(i1, i2, atoms, distance):
    v = atoms[i2].position - atoms[i1].position
    d0 = np.linalg.norm(v)
    u = v / d0
    #
    dr = 0.5 * distance * u
    atoms[i1].position += dr
    atoms[i2].position -= dr

    d1 = np.linalg.norm( atoms[i2].position - atoms[i1].position )
    if d1 > d0:
        atoms[i1].position -= 2 * dr
        atoms[i2].position += 2 * dr
    #
#    

def getCurves_3_becareful(list_nLines = [3, 5, 7, 25], filename2D = "dftrlxO.cfg", xlim=[14.0, 16.5]):
    # be careful because this is only intended for "dftrlxO.cfg": already ordered atomic positions
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    # filename2D = "dftrlxO.cfg"
    dictOfSpecies = {"0":"C", "1":"O"}
    dictCfgs      = read_cfgs(filename2D, dictOfSpecies)

    alatMin, alatMax = xlim
    #
    structs = [ dict2atoms2D(d) for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    alats   = [ d['cell'][2][2] for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    sortedIndices = np.argsort(alats)
    #
    list_alats = []
    list_energies = []
    list_newStrs = []
    for nLines in list_nLines:
        alats = []
        energies = []
        for struct in structs:
            s = generate_struct_of_nLines(atoms2D=struct, nLines=nLines)
            ############
            
            if nLines > 3: # distances for nlines=3 are 1.42, according to DFT hummm, and 2Dstructure is somewhat different to 1.42, and change from site to site
                # First edge
                for k in range(1, 14, 4):
                    i1 = k
                    i2 = i1 + 1
                    shorten(i1, i2, s, 0.20) #0.20Angstrom
                #
                nRepeat = (nLines + 1) / 2 # **** NLines MUST BE EVEN !!! @@@@@@@@@@@ *********
                nRepeat = int(nRepeat) 

                # Second edge
                for k in [0, 2, 4, 6]:
                    i1 = k + ((nRepeat - 1) * 18)
                    i2 = i1 + 1
                    shorten(i1, i2, s, 0.20) #0.20Angstrom
                #
            #
            ############
            list_newStrs.append(s)

            s.set_calculator( get_calculator() )
            natoms = len(s)
            # natoms = 1
            e = s.get_potential_energy()
            # print(e)
            energies.append(e)
            a, b, alat, _, _, _ = s.cell.cellpar()
            alats.append(alat)
        #
        emin = min(energies)
        # emin = 0.0
        energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
        #
        alats = [ alats[i] for i in sortedIndices]
        energies = [ energies[i] for i in sortedIndices]
        #
        list_alats.append(alats)
        list_energies.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(list_energies)):
        # plt.plot(list_alats[i], list_energies[i], '.')
        plt.plot(list_alats[i], list_energies[i])
    #
    return list_alats, list_energies, list_newStrs
#


#%%

def writeOnFileSortedArray(alats, energies, filename='dat'):
    sortedIndices = np.argsort(alats)
    f = open(filename, 'w')
    for i in sortedIndices:
        f.write("%3.3f  %3.3f" % (alats[i], 1000*energies[i]) + '\n')
    #
    f.close()    
#

def get_avg_charges_espressoOut(filename):
    import myread
    from myread import read_espresso_out
    atoms = read_espresso_out(filename)
    
    a, b, alat, _, _, _ = atoms.cell.cellpar()
    charges_at_end = atoms.get_initial_charges() # not initial!
    #
    sum = 0.0
    cont = 0
    for (charge, symbol) in zip(charges_at_end, atoms.get_chemical_symbols()):
        if symbol == "C":
            sum += charge
            cont += 1
    #
    avg = sum / cont
    return alat, avg
#    

# filename = "relaxed.cfg_0"
# i = 0
# atoms, energy = getAtoms_from_cfg(filename, i)
def getAtoms_from_cfg(filename, i):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    # filename = "relaxed.cfg_0"
    dictOfSpecies = {"0":"C", "1":"O"}
    dictCfgs = read_cfgs(filename, dictOfSpecies)

    atoms = dict2atoms( dictCfgs[i] )
    energy = dictCfgs[i]['outputs']['energy']
    return atoms, energy
#


# f = open('outs/lista', 'r')
# lista = f.read().splitlines()
# f.close()
# alats_ch = []
# avg_ch = []
# for filename in lista:
#     alat, avg = get_avg_charges_espressoOut('outs/' + filename)
#     # print(alat, avg)
#     alats_ch.append(alat)
#     avg_ch.append(avg)
# #
# import matplotlib.pyplot as plt
# plt.plot(alats_ch, avg_ch, 'o')

# alats, energies, structsWithVacancies = vacancies1(nVacancies=1, xlim=[14.0, 16.5], nLines=401, n_times=1)
def vacancies1(nVacancies=1, xlim=[14.0, 16.5], nLines=401, n_times=1):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs

    alatMin = xlim[0]
    alatMax = xlim[1]
    minAngl, maxAngl, dAngle = [90, 180, 2]

    alats = []
    energies = []
    structsWithVacancies = []

    for deg in np.arange(minAngl, maxAngl, dAngle):
        atoms = getBendedAtomsSpecies(nLines=nLines, sheet=False, deg=deg, specieA='O', specieB='O')
        _, _, alat, _, _, _ = atoms.cell.cellpar()
        if belongs(alat, alatMin, alatMax):
            for _ in range(n_times):
                s = atoms.copy()
                for _ in range(nVacancies):
                    lchem = s.get_chemical_symbols()
                    listIndxOxygens = [ i for i in range(len(lchem)) if lchem[i] == 'O' ]
                    random.shuffle(listIndxOxygens)
                    i = listIndxOxygens[0:1][0]
                    # print(i)
                    s.pop(i)
                #
                structsWithVacancies.append(s)
                # 
                s.set_calculator( get_calculator() )
                natoms = len(s)
                e = s.get_potential_energy()
                energies.append(e)
                alats.append(alat)
            #
        #
    #

    # for _ in range(n_times):
    #     for deg in np.arange(minAngl, maxAngl, dAngle):
    #         atoms = getBendedAtomsSpecies(nLines=nLines, sheet=False, deg=deg, specieA='O', specieB='O')
    #         _, _, alat, _, _, _ = atoms.cell.cellpar()
    #         if belongs(alat, alatMin, alatMax):
    #             s = atoms.copy()
    #             for _ in range(nVacancies):
    #                 lchem = s.get_chemical_symbols()
    #                 listIndxOxygens = [ i for i in range(len(lchem)) if lchem[i] == 'O' ]
    #                 random.shuffle(listIndxOxygens)
    #                 i = listIndxOxygens[0:1][0]
    #                 print(i)
    #                 s.pop(i)
    #             #
    #             structsWithVacancies.append(s)
    #             # 
    #             s.set_calculator( get_calculator() )
    #             natoms = len(s)
    #             e = s.get_potential_energy()
    #             energies.append(e)
    #             alats.append(alat)
    #         #
    #     #
    # #
    # emin = min(energies)
    # emin = 0.0
    # energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
    energies = [ energies[i] / natoms for i in range(len(energies))]
    #
    sortedIndices = np.argsort(alats)
    alats = [ alats[i] for i in sortedIndices]
    energies = [ energies[i] for i in sortedIndices]
    structsWithVacancies = [ structsWithVacancies[i] for i in sortedIndices]
    #
    return alats, energies, structsWithVacancies
#


# alats, energies, structsWithVacancies = dope1(nVacancies=1, xlim=[14.0, 20.0], nLines=401, n_times=1, dopante="B")
def dope1(nVacancies=1, xlim=[14.0, 20.0], nLines=401, n_times=1, dopante="B"):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs

    alatMin = xlim[0]
    alatMax = xlim[1]
    minAngl, maxAngl, dAngle = [90, 180, 2]

    alats = []
    energies = []
    structsWithVacancies = []

    for _ in range(n_times):
        for deg in np.arange(minAngl, maxAngl, dAngle):
            atoms = getBendedAtomsSpecies(nLines=nLines, sheet=False, deg=deg, specieA='O', specieB='O')
            _, _, alat, _, _, _ = atoms.cell.cellpar()
            if belongs(alat, alatMin, alatMax):
                s = atoms.copy()
                for _ in range(nVacancies):
                    lchem = s.get_chemical_symbols()
                    listIndxOxygens = [ i for i in range(len(lchem)) if lchem[i] == 'O' ]
                    random.shuffle(listIndxOxygens)
                    i = listIndxOxygens[0:1][0]
                    # print(i)
                    # s.pop(i)
                    s[i].symbol = dopante # = 'B'
                #
                structsWithVacancies.append(s)
                # 
                s.set_calculator( get_calculator() )
                natoms = len(s)
                e = s.get_potential_energy()
                energies.append(e)
                alats.append(alat)
            #
        #
    #
    # emin = min(energies)
    # emin = 0.0
    # energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
    energies = [ energies[i] / natoms for i in range(len(energies))]
    #
    sortedIndices = np.argsort(alats)
    alats = [ alats[i] for i in sortedIndices]
    energies = [ energies[i] for i in sortedIndices]
    structsWithVacancies = [ structsWithVacancies[i] for i in sortedIndices]
    #
    return alats, energies, structsWithVacancies
#
#%%

def getCurves_dope(dictOfSpecies={"0":"C", "1":"O", "2":"B"}, nDopants=1, dopeType='B', list_nLines = [25], filename2D = "dftrlxO.cfg", xlim=[14.0, 16.5]):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs
    #
    # filename2D = "dftrlxO.cfg"
    # dictOfSpecies = {"0":"C", "1":"O"}
    dictCfgs      = read_cfgs(filename2D, dictOfSpecies)

    alatMin, alatMax = xlim
    #
    structs = [ dict2atoms2D(d) for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    alats   = [ d['cell'][2][2] for d in  dictCfgs 
                if belongs( d['cell'][2][2], alatMin, alatMax ) ]
    #
    sortedIndices = np.argsort(alats)
    #
    list_alats = []
    list_energies = []
    dopedStructures = []
    for nLines in list_nLines:
        alats = []
        energies = []
        for struct in structs:
            s = generate_struct_of_nLines(atoms2D=struct, nLines=nLines)
            #
            lchem = s.get_chemical_symbols()
            listIndxOxygens = [ i for i in range(len(lchem)) if lchem[i] == 'O' ]
            random.shuffle(listIndxOxygens)
            chosenList = listIndxOxygens[0:nDopants]
            for i in chosenList:
                s[i].symbol = dopeType # = 'B'
            #
            dopedStructures.append(s)
            #
            s.set_calculator( get_calculator() )
            natoms = len(s)
            # natoms = 1
            e = s.get_potential_energy()
            # e = 100
            # print(e)
            energies.append(e)
            a, b, alat, _, _, _ = s.cell.cellpar()
            alats.append(alat)
        #
        emin = min(energies)
        # emin = 0.0
        energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
        #
        alats = [ alats[i] for i in sortedIndices]
        energies = [ energies[i] for i in sortedIndices]
        #
        list_alats.append(alats)
        list_energies.append(energies)
    #
    import matplotlib.pyplot as plt
    for i in range(len(list_energies)):
        # plt.plot(list_alats[i], list_energies[i], '.')
        plt.plot(list_alats[i], list_energies[i], '.')
    #
    return list_alats, list_energies, dopedStructures
#

def write2file(filaname, xlist, ylist):
    s = ""
    for i in range(len(ylist)):
        s += str(xlist[i]) + "  " + str(ylist[i]) + "\n"
    #
    f = open(filaname, "w")
    f.write(s)
    f.close()
#

# T00, T, atoms = get_T00_T_TD(3, 2, "armchair")
# T00, T, atoms = get_T00_T_TD(4, 1, "zigzag")
def get_T00_T_TD(lineas=3, cells=2, type='armchair'):
    if type == 'armchair':
        atoms = graphene_nanoribbon(lineas/2, cells, type, saturated=False, C_H=1.1, C_C=1.4, vacuum=10.0, sheet=False)
    elif type == 'zigzag':
        atoms = graphene_nanoribbon(lineas, cells, type, saturated=False, C_H=1.1, C_C=1.4, vacuum=10.0, sheet=False)
    #
    # d = atoms.copy()
    # d *= [2,1,2]
    # visualize(d, enumerar=True)
    #
    from ase import neighborlist
    cutOff = neighborlist.natural_cutoffs(atoms)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighborList.update(atoms)
    # T00 = neighborList.get_connectivity_matrix(sparse=False)
    #
    import numpy as np
    n = len(atoms)

    T00 = np.zeros((n,n), dtype=int)
    T   = np.zeros((n,n), dtype=int)
    for i in range(n):
        indices, offsets = neighborList.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            z = offset[2]
            if z == 0:
                T00[i,j] = 1
            #
            if z == 1:
                T[i,j] = 1
            #            
        #
    #



    # T = np.zeros((n,n), dtype=int)
    # for i in range(n):
    #     indices, offsets = neighborList.get_neighbors(i)
    #     for j, offset in zip(indices, offsets):
    #         z = offset[2]
    #         if z == 1:
    #             T[i,j] = 1
    #         #
    #     #
    # #
    # TD = np.transpose(T)


    #
    T00_jl = []
    T_jl = []
    l = 0
    for j in range(n):
        for i in range(n):
            l += 1
            if T00[i,j] == 1:
                T00_jl.append(l)
            #
            if T[i,j] == 1:
                T_jl.append(l)
            #
        #
    #
    f = open("jl_T00", "w")
    for i in range(len(T00_jl)):
        f.write(str(T00_jl[i]) + "\n")
    #
    f.close()
    
    f = open("jl_T", "w")
    for i in range(len(T_jl)):
        f.write(str(T_jl[i]) + "\n")
    #
    f.close()

    f = open("jl_n", "w")
    f.write(str(n))
    f.close()
    
    return T00, T, atoms

#

#T00, T, atoms = get_T00_T_TD_2(atoms)
def get_T00_T_TD_2(atoms):
    # if type == 'armchair':
    #     atoms = graphene_nanoribbon(lineas/2, cells, type, saturated=False, C_H=1.1, C_C=1.4, vacuum=10.0, sheet=False)
    # elif type == 'zigzag':
    #     atoms = graphene_nanoribbon(lineas, cells, type, saturated=False, C_H=1.1, C_C=1.4, vacuum=10.0, sheet=False)
    #
    # d = atoms.copy()
    # d *= [2,1,2]
    # visualize(d, enumerar=True)
    #
    from ase import neighborlist
    cutOff = neighborlist.natural_cutoffs(atoms)
    # cutOff = [cutOff[i]*4 for i in range(len(cutOff))]
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighborList.update(atoms)
    # T00 = neighborList.get_connectivity_matrix(sparse=False)
    #
    import numpy as np
    n = len(atoms)

    T00 = np.zeros((n,n), dtype=int)
    T   = np.zeros((n,n), dtype=int)
    for i in range(n):
        indices, offsets = neighborList.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            z = offset[2]
            if z == 0:
                T00[i,j] = 1
            #
            if z == 1:
                T[i,j] = 1
            #            
        #
    #



    # T = np.zeros((n,n), dtype=int)
    # for i in range(n):
    #     indices, offsets = neighborList.get_neighbors(i)
    #     for j, offset in zip(indices, offsets):
    #         z = offset[2]
    #         if z == 1:
    #             T[i,j] = 1
    #         #
    #     #
    # #
    # TD = np.transpose(T)


    #
    T00_jl = []
    T_jl = []
    l = 0
    for j in range(n):
        for i in range(n):
            l += 1
            if T00[i,j] == 1:
                T00_jl.append(l)
            #
            if T[i,j] == 1:
                T_jl.append(l)
            #
        #
    #
    f = open("jl_T00", "w")
    for i in range(len(T00_jl)):
        f.write(str(T00_jl[i]) + "\n")
    #
    f.close()
    
    f = open("jl_T", "w")
    for i in range(len(T_jl)):
        f.write(str(T_jl[i]) + "\n")
    #
    f.close()

    f = open("jl_n", "w")
    f.write(str(n))
    f.close()
    
    return T00, T, atoms

#

#%%
def getDx(atoms, i, j):
    return (atoms[i].position - atoms[j].position)[0]
#

def sacudeParalelo(atoms, i, j, eps, a):
    dx = getDx(atoms, i, j)
    if not belongs(dx, -eps, eps):
        atoms[j].position[0] += a
        dx = getDx(atoms, i, j)
        if not belongs(dx, -eps, eps):
            atoms[j].position[0] -= (2*a)
#

def zigzagear(atoms):
    a, b, alat, _, _, _ = atoms.cell.cellpar()
    eps = 0.2

    sacudeParalelo(atoms, 1, 2, eps, a)
    sacudeParalelo(atoms, 1, 5, eps, a)
    sacudeParalelo(atoms, 1, 6, eps, a)
    sacudeParalelo(atoms, 1, 9, eps, a)
    sacudeParalelo(atoms, 1, 10, eps, a)
    sacudeParalelo(atoms, 1, 13, eps, a)
    sacudeParalelo(atoms, 1, 14, eps, a)

    sacudeParalelo(atoms, 0, 3, eps, a)
    sacudeParalelo(atoms, 0, 4, eps, a)
    sacudeParalelo(atoms, 0, 7, eps, a)
    sacudeParalelo(atoms, 0, 8, eps, a)
    sacudeParalelo(atoms, 0, 11, eps, a)
    sacudeParalelo(atoms, 0, 12, eps, a)
    sacudeParalelo(atoms, 0, 15, eps, a)

    # Now oxygens:
    sacudeParalelo(atoms, 0, 16, eps, a)
    sacudeParalelo(atoms, 0, 17, eps, a)
#

# atoms = mywrap(dictCfgs, 18)
# visualize(atoms, enumerar=True)
def mywrap(dictCfgs, i):
    atoms = dict2atoms( dictCfgs[i] )

    # Wrap positions to unit cell.
    # See https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.set_scaled_positions
    # See ase.geometry.wrap_positions() in https://wiki.fysik.dtu.dk/ase/ase/geometry.html
    atoms.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)

    zigzagear(atoms)
    return atoms
#

# filename = "train.cfg"
# dictCfgs = read_cfgs_asinput(filename, dictOfSpecies)
#
# atoms = mywrap(dictCfgs, 300)
# visualize(atoms, enumerar=True)
def newCfgFixed(dictCfgs, dictOfSpecies):
    cfgString = ""
    for d in dictCfgs:
        atoms = dict2atoms(d)

        ####################
        # Wrap positions to unit cell.
        # See https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.set_scaled_positions
        # See ase.geometry.wrap_positions() in https://wiki.fysik.dtu.dk/ase/ase/geometry.html
        atoms.wrap(pbc=[True, True, True]) # <<-- MTP and QE will receive more work or even will give erroneous mindist (mlp mindist) or bad convergence (QE)
        zigzagear(atoms)


        d["positions"] = atoms.positions
        cfgString += dict2cfg(d, dictOfSpecies)
    #
    f = open("fixedTrainingSet.cfg", "w")
    f.write(cfgString)
    f.close()    
#    



# filename = "dftrlxO.cfg_76old"
# dic2D   = read_cfgs(filaname, dictOfSpecies)
# a2, e2, degrees = getCurvesFromRelaxed(filename, dictOfSpecies, xlim=[14.0, 20.5])
# sortIndx = np.argsort(a2)
# a2 = [ a2[i] for i in sortIndx ]
# e2 = [ e2[i] for i in sortIndx ]
# dic2D  = [ dic2D[i] for i in sortIndx ]
#
# filename = "train.cfg"
# dicTrain = read_cfgs(filename, dictOfSpecies)
# alats, energies, degrees = getCurvesFromRelaxed(filename, dictOfSpecies, xlim=[14.0, 20.5])
# sortIndx = np.argsort(alats)
# alats = [ alats[i] for i in sortIndx ]
# energies = [ energies[i] for i in sortIndx ]
# dicTrain  = [ dicTrain[i] for i in sortIndx ]
#
# atoms2D = dict2atoms2D(dic2D[8])
# atomsTrain = dict2atoms2D(dicTrain[76])

#

#%%


def predictFromFile(dictOfSpecies, filename="dftrlx_3lines_curated.cfg"):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs

    dto = read_cfgs(filename, dictOfSpecies)
    eto = []
    ato = []
    for i in range(len(dto)):
        ato.append(dto[i]["cell"][2][2])
        #
        d = dto[i]
        species = d['species']
        cell = d['cell']
        positions = d['positions']
        # pbc=(1,0,1) VERY IMPORTANT!!!
        atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=(1,0,1))
        atoms.set_calculator( get_calculator() )
        eto.append(atoms.get_potential_energy())
    #
    natoms = dto[0]["size"]
    emin3l = min(eto)
    # emin = 0.0
    eto = [ (eto[i] - emin3l) / natoms for i in range(len(eto))]
    #
    sortedIndices = np.argsort(ato)
    ato = [ ato[i] for i in sortedIndices]
    eto = [ eto[i] for i in sortedIndices]

    return ato, eto
#

def predictFromFileAsInput(dictOfSpecies, filename="to_relax.cfg"):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs_asinput

    dto = read_cfgs_asinput(filename, dictOfSpecies)
    eto = []
    ato = []
    for i in range(len(dto)):
        ato.append(dto[i]["cell"][2][2])
        #
        d = dto[i]
        species = d['species']
        cell = d['cell']
        positions = d['positions']
        # pbc=(1,0,1) VERY IMPORTANT!!!
        atoms = Atoms("".join(species), cell=cell, positions=positions, pbc=(1,0,1))
        atoms.set_calculator( get_calculator() )
        eto.append(atoms.get_potential_energy())
    #
    return ato, eto
#   

#%%

def getListBendedAtoms(nLines=2, sheet=True):
    cc_dist = 1.42
    vacuum = 9.0
    atoms = graphene_nanoribbon(nLines/2, 4, type='armchair', saturated=False, C_C=cc_dist, vacuum=vacuum, sheet=sheet)
    nAtoms = len(atoms)

    atomsTemp = atoms.copy()

    hmin = - cc_dist / 2
    hmax = + cc_dist / 2
    n = 10
    dh = (hmax - hmin) / n
    listH = [ hmin + (i * dh) for i in range(n+1)]

    lmin = 0.5 * cc_dist
    lmax = 1.5 * cc_dist
    dl = (lmax - lmin) / n    
    listL = [ lmin + (i * dl) for i in range(n+1)]

    listDeg = [90.0 + (i * 3) for i in range(31)]

    # listBendedStructs = []
    
    filename = "ListBended_nlines_" + str(nLines) + ".cfg"
    f = open(filename, "a")
    for h in listH:
    	for l in listL:
            for deg in listDeg:
                insertOxygensAbove(3, atomsTemp, h=h,  l=l)
                insertOxygensAbove(11, atomsTemp, h=h, l=l)

                nAtoms2 = len(atomsTemp)
                listInflectionPoint = [0, nAtoms, nAtoms2 - 1, 15]
                bend(atomsTemp, listInflectionPoints=listInflectionPoint, bigAngleDeg=deg)
                #
                _, _, alat, _, _, _ = atomsTemp.cell.cellpar()
                if belongs(alat, 14.0, 20.0):
                    # listBendedStructs.append(atomsTemp)
                    s = atoms2cfg(atomsTemp)
                    f.write(s)
                #
                atomsTemp = atoms.copy()
            #
        #
	#
    f.close()
    # return listBendedStructs
    #
#%%
def filterCfgFile(filenameIn, dictOfSpecies, filenameOut):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs

    dictCfgs = read_cfgs(filenameIn, dictOfSpecies)
    alatMin = 14.0
    alatMax = 16.5
    newListDict = []
    for d in dictCfgs:
        alat = d["cell"][2][2]
        if belongs(alat, alatMin, alatMax):
            newListDict.append(d)
        #
    #

    cfgString = ""
    for d in newListDict:
        cfgString += dict2cfg(d, dictOfSpecies)
    #
    f = open(filenameOut, "w")
    f.write(cfgString)
    f.close()    
#


#%%
def isOxygenFound(atoms):
    iAtom = -1
    for (i, atom) in enumerate(atoms):
        if atom.symbol == "O":
            iAtom = i
            break
        #
    #
    return iAtom
#

def getRidOfOxygens(atoms):
    n = len(atoms)
    for _ in range(n):
        iAtom = isOxygenFound(atoms)
        if iAtom != -1:
            # print(iAtom)
            atoms.pop(iAtom)
        #
    #
    return atoms
#

def getLargestX(atoms):
    xlist = [ atom.position[0] for atom in atoms ]
    return max(xlist)
#

def getMinX(atoms):
    xlist = [ atom.position[0] for atom in atoms ]
    return min(xlist)
#

# def getShiftZ(atoms, alat, Zcenter):
#     l = [ atoms.position[2] for atom in atoms ]
#     d = max(l) - min(l)
#     space = alat - d
#     zshift = (d +  alat - d) / 2
#     return zshift


def getBendedAtomsSpecies2(nLines=3, sheet=False, deg=100, h=0.0, l=1.42):
    cc_dist = 1.42
    vacuum = 9.0
    atoms = graphene_nanoribbon(nLines/2, 4, type='armchair', saturated=False, C_C=cc_dist, vacuum=vacuum, sheet=sheet)
    nAtoms = len(atoms)
    insertOxygensAbove2( 3, atoms, h=h, l=l )
    insertOxygensAbove2(11, atoms, h=h, l=l )
    nAtoms2 = len(atoms)
    listInflectionPoint = [0, nAtoms, nAtoms2 - 1, 15]
    bend(atoms, listInflectionPoints=listInflectionPoint, bigAngleDeg=deg)
    return atoms
    #
#
#%%

def getZcenter(atoms):
    Zcenter = (atoms[15].position[2] + atoms[0].position[2]) / 2
    return Zcenter
#
def getYcenter(atoms):
    Ycenter = (atoms[15].position[1] + atoms[0].position[1]) / 2
    return Ycenter
#

# Ycenter_shifted = getYcenter_shifted(atoms)
def getYcenter_shifted(atoms):
    Ycenter_shifted = (atoms[3].position[1] + atoms[4].position[1]) / 2
    return Ycenter_shifted
#
def getZcenter_shifted(atoms):
    Zcenter_shifted = (atoms[3].position[2] + atoms[4].position[2]) / 2
    return Zcenter_shifted
#

# def add2lines_b(atoms1, angle, deleteOxyen, shift, Y1center, Z1center):
#     cc_dist = 1.42
#     a, b, alat1, _, _, _ = atoms1.cell.cellpar()
#     # xAtoms1 = atoms1[1].position[0]
#     xAtoms1 = getLargestX(atoms1)

#     nLines = 2 # brings 1 oxygen atom rows
#     atoms2 = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=angle, h=0.0, l=cc_dist)
#     a2, _, alat2, _, _, _ = atoms2.cell.cellpar()



#     atoms2 = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=angle2, h=0.0, l=l)
#     if deleteOxyen:
#         atoms2 = getRidOfOxygens(atoms2)
#     #





def add2lines(atoms1, angle, deleteOxyen, shift, Y1center, Z1center):
    cc_dist = 1.42

    a, b, alat1, _, _, _ = atoms1.cell.cellpar()
    # xAtoms1 = atoms1[1].position[0]
    xAtoms1 = getLargestX(atoms1)

    nLines = 2 # brings 1 oxygen atom rows
    atoms2 = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=angle, h=0.0, l=cc_dist)
    a2, _, alat2, _, _, _ = atoms2.cell.cellpar()

    # angle1 = angleInput
    angle2 = angle
    # beta1 = (90.0 - (angle1 / 2)) * np.pi / 180.0
    beta2 = (90.0 - (angle2 / 2)) * np.pi / 180.0
    d1 = (alat2 - alat1) / 2
    d1 /= np.cos(beta2)
    l =  cc_dist - d1  

    atoms2 = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=angle2, h=0.0, l=l)
    if deleteOxyen:
        atoms2 = getRidOfOxygens(atoms2)
    #
    # Z1center = getZcenter(atoms1)
    Z2center = getZcenter(atoms2)
    dZ = Z2center - Z1center
    #
    # Y1center = getYcenter(atoms1)
    Y2center = getYcenter(atoms2)
    dY = Y2center - Y1center

    a += a2 / 2
    xAtoms2 = atoms2[1].position[0]
    #
    dX = -xAtoms2 + xAtoms1 + (cc_dist * np.sqrt(3))
    #
    # shiftZ = getShiftZ(atoms1) + 
    #
    Y1center_shifted = getYcenter_shifted(atoms1)
    dY_shifted = Y2center - Y1center_shifted
    Z1center_shifted = getZcenter_shifted(atoms1)
    dZ_shifted = Z2center - Z1center_shifted

    for i in range(len(atoms2)):
        atoms2[i].position[0] += dX
        ####################################### centering Y and Z #######################################
        # atoms2[i].position[1] -= dY
        # atoms2[i].position[2] -= dZ
        #################################################################################################
        # atoms2[i].position[2] += shift * alat1 / 2
        # atoms2[i].position[2] += shift * alat2 / 2
        if shift == 0.5:
            x=5
            atoms2[i].position[2] += shift * alat1 / 2
            atoms2[i].position[0] += 0.12 # <<< I had to give this extra space, by random gessing until all distances were 1.42Angstroms (mindist)
            # atoms2[i].position[1] += dY
            # atoms2[i].position[2] += dZ
            # atoms2[i].position[1] -= dY_shifted
            # atoms2[i].position[2] -= dZ_shifted
        elif shift == 1.0:
            x = 5
            # atoms2[i].position[2] += shift * alat1 / 2
            # atoms2[i].position[1] += dY
            # atoms2[i].position[1] -= dY

            # atoms2[i].position[0] += 0.0875 # <<< I had to give this extra space, by random gessing until all distances were 1.42Angstroms (mindist)

            # atoms2[i].position[2] += dZ
            # atoms2[i].position[1] -= dY_shifted
            # atoms2[i].position[2] -= dZ_shifted
        #


    #
    ####



    atoms1.extend(atoms2)    
    atoms1.cell = a, b, alat1
    return atoms1    
#

#%%

def add2lines_c(atoms1, angle2, deleteOxyen, shift, Y1center, Z1center, delta, deltaY):
    cc_dist = 1.42
    a, b, alat1, _, _, _ = atoms1.cell.cellpar()
    # xAtoms1 = atoms1[1].position[0]
    xAtoms1 = getLargestX(atoms1)


    nLines = 2 # brings 1 oxygen atom rows
    # l1 = cc_dist
    # l2 = cc_dist
    atoms2 = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=angle2, h=0.0, l=cc_dist)
    # atoms2 = getBendedAtomsSpecies_c(nLines=nLines, sheet=False, deg=angle2, h=0.0, l1=l1, l2=l2)
    a2, _, alat2, _, _, _ = atoms2.cell.cellpar()

    # angle2 = angle
    # beta1 = (90.0 - (angle1 / 2)) * np.pi / 180.0
    beta2 = (90.0 - (angle2 / 2)) * np.pi / 180.0
    d = (alat2 - alat1) / 2
    d /= np.cos(beta2)
    l =  cc_dist - d

    # l1 = (l*0.5) #+ delta
    # l2 = (l*0.5) #- delta
    l1 = delta * l
    l2 = (1 - delta) * l
    atoms2 = getBendedAtomsSpecies_c(nLines=nLines, sheet=False, deg=angle2, h=0.0, l1=l1, l2=l2)
    if deleteOxyen:
        atoms2 = getRidOfOxygens(atoms2)
    #
    # Y1center = getYcenter(atoms1)
    Y2center = getYcenter(atoms2)
    dY = Y2center - Y1center
    # Z1center = getZcenter(atoms1)
    Z2center = getZcenter(atoms2)
    dZ = Z2center - Z1center    

    a += a2 / 2
    xAtoms2 = atoms2[1].position[0]
    #
    dX = -xAtoms2 + xAtoms1 + (cc_dist * np.sqrt(3))
    for i in range(len(atoms2)):
        atoms2[i].position[0] += dX
        ####################################### centering Y and Z #######################################
        atoms2[i].position[1] -= dY + deltaY
        atoms2[i].position[2] -= dZ

    #
    


    
    atoms1.extend(atoms2)    
    atoms1.cell = a, b, alat1
    return atoms1    

#%%
def addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter):
    for (i, angle) in enumerate(listAngles):
        shift = listShifts[i]
        atoms = add2lines(atoms,  angle, deleteOxyen, shift, Ycenter, Zcenter)
    #
    return atoms
#

def getListAnglesAndShifts(degInput, nLinesG):
    n = int( (nLinesG + 3) / 4 ) # nLinesG can be 1, 5, 9
    alphaComplement = 180 - degInput
    dBeta = alphaComplement / n
    list = [ degInput + ( (i + 1) * dBeta) for i in range(n - 1) ]
    listAngles = list + [ 180.0 ] + list[::-1]
    listShifts = [0 for i in range(len(list) + 1)] + [0.5 for i in range(len(list))]
    return listAngles, listShifts
#

#%%
# atoms = get_notHighlyOrdrdOxRibb3(nLines=7, degInput=120, nLinesG=3):
def get_notHighlyOrdrdOxRibb3(nLines=7, degInput=120, nLinesG=3):
    cc_dist = 1.42
    atoms = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=degInput, h=0.0, l=cc_dist)
    Zcenter = getZcenter(atoms)
    Ycenter = getYcenter(atoms)

    add2lines(atoms, angle=degInput, deleteOxyen=False, shift=0.5, Y1center=Ycenter, Z1center=Zcenter)

    # print(getMinDist(atoms))
    # visualize(atoms, enumerar=True)
    # distMatrix = atoms.get_all_distances()
    # for i in range(len(distMatrix[0])):
    #     for j in range(len(distMatrix[0])):
    #         if belongs(distMatrix[i][j], 0.5, 1.41):
    #             print(i,j)
    
    atoms.wrap()
    return atoms

#%%
def get_notHighlyOrdrdOxRibb2(nLines=7, degInput=120, nLinesG=3):
    cc_dist = 1.42
    
    # nLines = 7 # brings 3 oxygen atom rows

    deg = degInput
    atoms = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=deg, h=0.0, l=cc_dist)
    
    Zcenter = getZcenter(atoms)
    Ycenter = getYcenter(atoms)
    listAngles0, listShifts0 = getListAnglesAndShifts(degInput, nLinesG)
    # listYshift = getListYshift()

    listAngles = listAngles0.copy()
    listShifts = listShifts0.copy()
    deleteOxyen = True
    atoms = addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter)

    # nRepeat = int((nLines + 1) / 2)  # **** NLines MUST BE EVEN !!! @@@@@@@@@@@ *********
    # listAngles = degInput * np.ones(nRepeat - 1)
    # listShifts = 0.5 * np.ones(nRepeat - 1)
    # deleteOxyen = False
    # atoms = addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter)


    # deleteOxyen = True
    # listAngles = listAngles0.copy()
    # listShifts = 0.5 * np.ones(len(listShifts0))
    # listShifts[-1] = 1.0
    # atoms = addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter)


    atoms.wrap()

    a, b, alat1, _, _, _ = atoms.cell.cellpar()
    a = getLargestX(atoms) - getMinX(atoms) + 30.0
    atoms.cell = a, b, alat1


    return atoms
#

#%%


def get_notHighlyOrdrdOxRibb(nLines=7, degInput=120, nLinesG=3):
    cc_dist = 1.42
    
    # nLines = 7 # brings 3 oxygen atom rows

    deg = degInput
    atoms = getBendedAtomsSpecies2(nLines=nLines, sheet=False, deg=deg, h=0.0, l=cc_dist)
    
    Zcenter = getZcenter(atoms)
    Ycenter = getYcenter(atoms)
    listAngles0, listShifts0 = getListAnglesAndShifts(degInput, nLinesG)

    listAngles = listAngles0.copy()
    listShifts = listShifts0.copy()
    deleteOxyen = True
    atoms = addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter)

    # nRepeat = int((nLines + 1) / 2)  # **** NLines MUST BE EVEN !!! @@@@@@@@@@@ *********
    # listAngles = degInput * np.ones(nRepeat - 1)
    # listShifts = 0.5 * np.ones(nRepeat - 1)
    # deleteOxyen = False
    # atoms = addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter)


    # deleteOxyen = True
    # listAngles = listAngles0.copy()
    # listShifts = 0.5 * np.ones(len(listShifts0))
    # listShifts[-1] = 1.0
    # atoms = addSeveralLines(atoms, listAngles, deleteOxyen, listShifts, Ycenter, Zcenter)


    atoms.wrap()

    a, b, alat1, _, _, _ = atoms.cell.cellpar()
    a = getLargestX(atoms) - getMinX(atoms) + 30.0
    atoms.cell = a, b, alat1


    return atoms
#
#%%

# alats, energies = getCurvesNotHighlyOrdrd(xlim=[14.0, 20.0], nLines=401, nLinesG=11)
def getCurvesNotHighlyOrdrd(xlim=[14.0, 20.0], nLines=401, nLinesG=11):
    import sys
    sys.path.insert(1, '/Users/chinchay/Documents/10_Canadawork/MTPy')
    import cfg2ase
    import importlib
    importlib.reload(cfg2ase)
    from cfg2ase import read_cfgs

    alatMin = xlim[0]
    alatMax = xlim[1]
    minAngl, maxAngl, dAngle = [90, 180, 2]

    alats = []
    energies = []
    # structsWithVacancies = []

    for deg in np.arange(minAngl, maxAngl, dAngle):
        atoms = get_notHighlyOrdrdOxRibb(nLines=nLines, degInput=deg, nLinesG=nLinesG)
        _, _, alat, _, _, _ = atoms.cell.cellpar()
        if belongs(alat, alatMin, alatMax):
            if getMinDist(atoms) > 1.41:
                atoms.set_calculator( get_calculator() )
                natoms = len(atoms)
                e = atoms.get_potential_energy()
                energies.append(e)
                alats.append(alat)
            else:
                print(deg)
            #

        #
    #
    #
    emin = min(energies)
    # emin = 0.0
    energies = [ (energies[i] - emin) / natoms for i in range(len(energies))]
    # energies = [ energies[i] / natoms for i in range(len(energies))]
    #
    sortedIndices = np.argsort(alats)
    alats = [ alats[i] for i in sortedIndices]
    energies = [ energies[i] for i in sortedIndices]
    # structsWithVacancies = [ structsWithVacancies[i] for i in sortedIndices]
    #
    return alats, energies #, structsWithVacancies
#

