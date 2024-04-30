from firedrake import *


def tqg_energy(q, psi, f, h, b):
    return 0.5*assemble( (psi*(q-f) + h*b)*dx )


def tqg_kinetic_energy(q, psi, f):
    return 0.5 * assemble( psi*(q-f)*dx )


def tqg_potential_energy(h, b):
    return 0.5 * assemble( h*b*dx )
