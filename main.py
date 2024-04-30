#   Copyright 2020
#
#   STUOD Horizontal Rayleigh Benard convection
#   
#
import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import pi, ensemble,COMM_WORLD
from firedrake.petsc import PETSc

from tqg.example2 import TQGExampleTwo as Example2params
from tqg.example3l import TQGExampleThreeL as Example3lparams

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm
ensemble_comm = ensemble


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = opts['time']
    nx = opts['resolution']
    dt = opts['time_step']
    dump_freq = opts['dump_freq']
    alpha = opts['alpha']
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']

    PETSc.Sys.Print("time horizon: {}, res: {}, dt: {}, dump_req: {}, alpha: {}".format(T, nx, dt, dump_freq, alpha), flush=True)

    #workspace = Workspace("/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha if alpha != None else 'none'))

    # workspace = Workspace("/home/wpan1/Development/Data/AngryDolphin")
    workspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
    
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=comm).get_fine_mesh()
    
    params = Example2params(T,dt,mesh, bc='x', alpha=alpha)
    # params = Example3lparams(T,dt,TorusMeshHierarchy(nx,nx,1.,1.,0,"both",comm=comm).get_fine_mesh(), bc='',alpha=alpha)
    solver = RTQGSolver(params) if alpha!=None else TQGSolver(params)

    # visual_output_name = workspace.output_name("_pde_visuals", "visuals")
    # data_output_name   = workspace.output_name("_pde_data", "data")
    # h5_data_name = workspace.output_name("pde_data_2062", "data")

    visual_output_name = workspace.output_name("pde_visuals", "PDESolution-random-bathymetry/visuals")
    data_output_name  = workspace.output_name("pde_data", "PDESolution-random-bathymetry")
    h5_data_name = workspace.output_name("pde_data_0", "PDESolution")
    solver.load_initial_conditions_from_file(h5_data_name)
    solver.perturb_bathymetry() 
    # solver.perturb_bathymetry_deterministic(freq=16.)

    solver.solve(dump_freq, visual_output_name, data_output_name, ensemble_comm, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx)
