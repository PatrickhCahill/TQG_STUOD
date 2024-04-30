#   Copyright 2020
#
#   STUOD Horizontal Rayleigh Benard convection
#   
#
import sys 
from firedrake import *
from utility import Workspace
from tqg.solver import TQGSolver
from firedrake_utility import TorusMeshHierarchy
from firedrake import pi, ensemble,COMM_WORLD, PETSc

from tqg.example2 import TQGExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":
     
    nx = int(sys.argv[1] )
    alpha = None
    procno = ensemble_comm.rank 
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=comm).get_fine_mesh()
    gradperp = lambda _u : as_vector((-_u.dx(1), _u.dx(0))) 
    V = FunctionSpace(mesh, "CG", 1)
    Vu = VectorFunctionSpace(mesh, "DG", 1)
    Vucg = VectorFunctionSpace(mesh, "CG", 1)
    q = Function(V)
    v = Function(Vu)
    vcg = Function(Vucg)
    x, y = SpatialCoordinate(mesh)
    q.interpolate(sin(2*pi*x)*sin(2*pi*y))
    v.project(gradperp(q))
    vcg.project(gradperp(q))
    PETSc.Sys.Print(nx, errornorm(v, vcg), flush=True)
    PETSc.Sys.Print(nx, norm(v)- norm(vcg), flush=True)

