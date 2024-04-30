import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
import euler.euler as euler
from firedrake import pi, ensemble,COMM_WORLD, SpatialCoordinate, cos, div, grad, assemble, norm, Function
from firedrake.petsc import PETSc

from tqg.example2 import TQGExampleTwo as Example2params

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

    workspace = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")
    
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=comm).get_fine_mesh()
    x= SpatialCoordinate(mesh)
    
    params = euler.EulerParams(T,dt,mesh)
    solver = euler.EulerSolver(params)

    visual_output_name = workspace.output_name("pde_visuals", "PDESolution-euler/visuals")
    data_output_name  = workspace.output_name("pde_data", "PDESolution-euler")

    bathymetry = Function(solver.psi0.function_space()).interpolate(cos(2*pi*x[0]) + 0.5 * cos(4.0*pi * x[0]) + cos(6.0*pi*(x[0]))/3.0)
    solver.solve_for_q_given_psi(bathymetry, solver.initial_cond)
    solver.psi_solve_given_q(solver.initial_cond)
    #print(norm(solver.psi0 - bathymetry))
    

    solver.solver(dump_freq, solver.initial_cond, output_name=visual_output_name, output_visual_flag=True, chkpt_flag=False) #data_output_name, ensemble_comm, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx)
