#   Copyright 2020
#
#   STUOD Horizontal Rayleigh Benard convection
#   
#

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
    nx = 512
    alpha = None
    procno = ensemble_comm.rank 
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=comm).get_fine_mesh()
    params = Example2params(1.,0.0001,mesh, bc='x', alpha=alpha)
    solver = TQGSolver(params)
    workspace = Workspace("/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha if alpha != None else 'none'))
    fileset = workspace.list_all_h5_files("data")
    for filename in fileset:
        PETSc.Sys.Print(filename, flush=True)
        solver.save_velocity_grid_data(filename[:-3], res=nx)


