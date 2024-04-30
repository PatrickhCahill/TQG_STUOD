
#   Copyright 2020
#
#   STUOD Horizontal Rayleigh Benard convection
#   
#
import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
from stqg.solver_forcing_buoyancy import STQGSolver
from firedrake import pi, ensemble,COMM_WORLD, File
from firedrake.petsc import PETSc 
from tqg.example2 import TQGExampleTwoTest as Example2params

spatial_size = 1
ensemble = ensemble.Ensemble(COMM_WORLD, spatial_size)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = opts['time']
    nx = opts['resolution']
    dt = opts['time_step']
    dump_freq = opts['dump_freq']
    alpha = opts['alpha']
    alpha = None
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']
    do_save_spectrum = None

    PETSc.Sys.Print("time horizon: {}, res: {}, dt: {}, dump_req: {}, alpha: {}".format(T, nx, dt, dump_freq, alpha), flush=True)

    workspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
    
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()
    
    params = Example2params(T,dt,mesh, bc='x', alpha=alpha)
    # params = Example3lparams(T,dt,TorusMeshHierarchy(nx,nx,1.,1.,0,"both",comm=comm).get_fine_mesh(), bc='',alpha=alpha)
    solver = STQGSolver(params) 
    zeta_fname = workspace.output_name("zetas.npy", "ParamFiles")
    visual_output_name = workspace.output_name("spde_visuals", "TestFiles/visuals")
    data_output_name   = workspace.output_name("spde_data", "TestFiles/data")

    if 1:
        #solver.load_initial_conditions_from_file(workspace.output_name("ensemble_member_0", "EnsembleMembers"), comm=spatial_comm)
        #solver.solve(dump_freq, visual_output_name, data_output_name, ensemble, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx, zetas_file_name=zeta_fname, xi_scaling=10000. )
        solver.solve(dump_freq, visual_output_name, data_output_name, ensemble, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx, xi_scaling=.001, bathymetry_xi=True )


    if 0:
        visual_out = File(workspace.output_name("cpde_data.pvd", "cPDESolution/visuals"), comm=spatial_comm)

        for index in range(1501):
            solver.load_initial_conditions_from_file(workspace.output_name("cpde_data_{}".format(index), "cPDESolution"), comm=spatial_comm)
            _t = round(dt*dump_freq*index, 5)
            print(index, _t)
            visual_out.write(solver.initial_cond, solver.initial_b, solver.psi0, solver.ssh, time=_t)


