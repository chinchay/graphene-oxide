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
import generate as gn


# atoms = getMirrorChain(deg, cc_dist)
def getMirrorChain(deg, cc_dist):
    # atoms1 = getBendedAtomsSpecies2(nLines=2, sheet=False, deg=deg, h=0.0, l=cc_dist)
    # for i in range(len(atoms2)):
    #     atoms1[i].position[0] -= 2*cc_dist * np.sqrt(3)
    # #
    atoms2 = gn.getBendedAtomsSpecies2(nLines=2, sheet=False, deg=deg, h=0.0, l=cc_dist)
    _, _, alat2, _, _, _ = atoms2.cell.cellpar()
    for i in range(len(atoms2)):
        # atoms2[i].position[2] += 0.125 * alat2
        # atoms2[i].position[2] += shift * alat2
        if i in [1,2, 5,6, 9,10, 13,14]:
            atoms2[i].position[0] -=  cc_dist * np.sqrt(3)
        #
    #
    # atoms2.wrap()
    x0 = atoms2[0].position[0]
    for i in range(len(atoms2)):
        atoms2[i].position[0] -= x0 
    #

    # for i in range(len(atoms2)):
    #     atoms2[i].position[0] += (ixshift * cc_dist * np.sqrt(3)) + dx
    # #
    return atoms2

#

# # atoms = mirrorChain(ixshift, deg, shift, cc_dist)
# def mirrorChain(deg, shift, cc_dist):
#     # atoms1 = getBendedAtomsSpecies2(nLines=2, sheet=False, deg=deg, h=0.0, l=cc_dist)
#     # for i in range(len(atoms2)):
#     #     atoms1[i].position[0] -= 2*cc_dist * np.sqrt(3)
#     # #
#     atoms2 = getBendedAtomsSpecies2(nLines=2, sheet=False, deg=deg, h=0.0, l=cc_dist)
#     _, _, alat2, _, _, _ = atoms2.cell.cellpar()
#     for i in range(len(atoms2)):
#         # atoms2[i].position[2] += 0.125 * alat2
#         atoms2[i].position[2] += shift * alat2
#         if i in [1,2, 5,6, 9,10, 13,14]:
#             atoms2[i].position[0] -=  cc_dist * np.sqrt(3)
#         #
#     #
#     # atoms2.wrap()
#     x0 = atoms2[0].position[0]
#     for i in range(len(atoms2)):
#         atoms2[i].position[0] -= x0 
#     #

#     # for i in range(len(atoms2)):
#     #     atoms2[i].position[0] += (ixshift * cc_dist * np.sqrt(3)) + dx
#     # #
#     return atoms2

#
#
# atoms, ixshift, dx, previousChain = addChain(option, previousOption, ixshift, atoms, previousChain, deg)
def addChain(option, previousOption, ixshift, atoms1, previousChain, deg, atomsMirror, atomsBended):
    # atoms_original = atoms1.copy()
    previousChain_original = previousChain.copy()

    # ixshift = 0
    # nChose = len(atoms1)-1-2
    # xfinal = atoms1[nChose].position[0]
    # print("nChose=", nChose)
    

    cc_dist = 1.42

    mydict = { 0:0.0, 1:((1.5) / 14), 2:0.25, 3:0.5 }
    shift = -100

    # # print(ixshift, option, "====")
    # if previousOption == 1:
    #     option = 2
    # #
    # print(previousOption, option)
    # if (option - previousOption == 1) and (previousOption != 0):
    #     shift = mydict[option]
    #     ixshift += 0.5
    #     print("a0")
    if option in [0, 2, 3]:
        if previousOption != 1:
            shift = mydict[option]
            ixshift += 1
            # print("a")
        else:
            option = 2 # already done above!
            shift = mydict[option]
            ixshift += 0.5
            # print("b")
    elif option == 1:
        shift = mydict[option]
        if previousOption != 1:
            ixshift += 1.5
            # print("c")
        else:
            ixshift += 1
            # print("d")
        #
    #
    
    # aa = xfinal + (ixshift * cc_dist * np.sqrt(3))
    aa = ixshift * cc_dist * np.sqrt(3)
    # print(option, xfinal, aa, shift, ixshift)
    # print(option, aa, shift, ixshift)
    

    if option != 1:
        atoms2_0 = atomsBended.copy()
    else:
        atoms2_0 = atomsMirror.copy()
    #

    atoms2 = atoms2_0.copy()
    #
    _, _, alat2, _, _, _ = atoms2.cell.cellpar()
    n2 = len(atoms2)

    listDx = np.arange(aa-0.5, aa+0.5, 0.05)
    # listDx = [aa]
    
    # print(listDx)
    dxexit = 0
    mindist = -1
    for dx in listDx:
        atoms2 = atoms2_0.copy()
        if mindist < 1.419:
            dxexit = dx
            # print(atoms2[0].position[0], dxexit, aa-cc_dist, aa+cc_dist, "+xx")
            for i in range(n2):
                atoms2[i].position[0] += dx
                atoms2[i].position[2] += shift * alat2   # 0.5 * alat2
            #
            ############################################################
            previousChain.extend(atoms2)
            mindist = gn.getMinDist(previousChain)
            previousChain = previousChain_original.copy()
        else:
            break
        #
    #
    
    # print(atoms2[0].position[0], dxexit, "...")

    # previousChain = previousChain_original.copy()
    # listDx = np.arange(dxexit-1.0, dxexit+1.0, 0.05)
    # for dx in listDx:
    #     dxexit = dx
    #     atom2 = atoms2_0.copy()
    #     for i in range(len(atoms2)):
    #         atoms2[i].position[0] += dx
    #         atoms2[i].position[2] += shift * alat2   # 0.5 * alat2
    #     #
    #     ############################################################
    #     previousChain.extend(atoms2)
    #     mindist = getMinDist(previousChain)
    #     if mindist < 1.419:
    #         # dx += 0.01
    #         previousChain = previousChain_original.copy()
    #     else:
    #         break
    #     #
    # #
       
    # previousChain = previousChain_original.copy()
    # listDx = np.arange(dxexit-0.1, dxexit+0.1, 0.01)
    # for dx in listDx:
    #     dxexit = dx
    #     atom2 = atoms2_0.copy()
    #     for i in range(len(atoms2)):
    #         atoms2[i].position[0] += dx
    #         atoms2[i].position[2] += shift * alat2   # 0.5 * alat2
    #     #
    #     ############################################################
    #     previousChain.extend(atoms2)
    #     mindist = getMinDist(previousChain)
    #     if mindist < 1.419:
    #         # dx += 0.01
    #         previousChain = previousChain_original.copy()
    #     else:
    #         break
    #     #
    # #

    atoms2 = atoms2_0.copy()
    for i in range(n2):
        atoms2[i].position[0] += dxexit
        atoms2[i].position[2] += shift * alat2   # 0.5 * alat2
    #############################################################
    
    # print(atoms2[0].position[0], dxexit, "...")

    atoms1.extend(atoms2)
    ixshift = dxexit / (cc_dist * np.sqrt(3))
    return atoms1, ixshift, dxexit, atoms2, option 
#

# _, _, alat2, _, _, _ = atoms2.cell.cellpar()
# atoms2[i].position[2] += shift * alat2

def zerarX(atoms):
    x0 = atoms[0].position[0]
    for i in range(len(atoms)):
        atoms[i].position[0] -= x0 
    #
    return atoms
#

# atoms = generateRandomStr(atoms, deg, nLines, nRepeat)
def generateRandomStr(atoms, deg, nLines, nRepeat):
    a, b, alat, _, _, _ = atoms.cell.cellpar()
    cc_dist = 1.42
    atomsMirror = getMirrorChain(deg, cc_dist)
    atomsMirror = zerarX(atomsMirror)
    atomsBended = gn.getBendedAtomsSpecies2(nLines=2, sheet=False, deg=deg, h=0.0, l=cc_dist)
    atomsBended  = zerarX(atomsBended)

    import random
    previousOption = 0
    ixshift = 0
    previousChain = atoms.copy()
    for _ in range(nLines):
        option = random.randint(0,3)
        # atoms, ixshift, dx, previousChain, option = addChain(option, previousOption, ixshift, atoms, previousChain, deg, atomsMirror, atomsBended)
        # previousOption = option
        
        ########################################
        if nRepeat > 1:
            for _ in range(nRepeat):
                option0 = option
                atoms, ixshift, dx, previousChain, option = addChain(option, previousOption, ixshift, atoms, previousChain, deg, atomsMirror, atomsBended)
                previousOption = option
                option = option0
            #
        else:
            atoms, ixshift, dx, previousChain, option = addChain(option, previousOption, ixshift, atoms, previousChain, deg, atomsMirror, atomsBended)
            previousOption = option
        #
        ###########################################

        # print(len(atoms))
        # print(getMinDist(atoms), ixshift, dx)
    #
    #
    a += 10.0 + ixshift * (cc_dist * np.sqrt(3))
    atoms.set_cell([a, b, alat])
    atoms.wrap()
    return atoms
#

# atoms =  generateSurface(nLines, deg, nRepeat)
def generateSurface(nLinesPlusOne, deg, nRepeat):
    cc_dist = 1.42
    atoms = gn.getBendedAtomsSpecies2(nLines=2, sheet=False, deg=deg, h=0.0, l=cc_dist)
    atoms  = zerarX(atoms)
    #
    atoms = generateRandomStr(atoms, deg, nLinesPlusOne-1, nRepeat)
    atoms.wrap()
    #
    return atoms
#    

# alats, energies = getCurvesNotHighlyOrdrd(xlim=[14.0, 20.0], nLines=9, nRepeat=1)
def getCurvesNotHighlyOrdrd(xlim=[14.0, 20.0], nLines=9, nRepeat=1):
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
        atoms =  generateSurface(nLines, deg, nRepeat)
        _, _, alat, _, _, _ = atoms.cell.cellpar()
        if gn.belongs(alat, alatMin, alatMax):
            if gn.getMinDist(atoms) > 1.41:
                # print(deg)
                atoms.set_calculator( gn.get_calculator() )
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
#%%

#deg    = 120
#nLines = 9
#nRepeat = 1
#atoms =  generateSurface(nLines, deg, nRepeat)

#print(gn.getMinDist(atoms))
#gn.visualize(atoms)
# %%

