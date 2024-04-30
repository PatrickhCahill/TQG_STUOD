from firedrake import *
import numpy as np


def tqg_energy(q, psi, f, h, b):
    return 0.5*assemble( (psi*(q-f) + h*b)*dx )


def tqg_kinetic_energy(q, psi, f):
    return 0.5 * assemble( psi*(q-f)*dx )


def tqg_potential_energy(h, b):
    return 0.5 * assemble( h*b*dx )


def tqg_mesh_grid(res):
    xx = np.linspace(0, 1, res+1)
    yy = np.linspace(0, 1, res+1)

    mygrid = []
    for i in range(len(xx)):
        for j in range(len(yy)):
            mygrid.append([xx[i], yy[j]])
    return np.asarray(mygrid) # now i have a list of grid values
