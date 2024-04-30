
import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import pi, ensemble,COMM_WORLD
from firedrake.petsc import PETSc

from tqg.example_james_woodfield import TQGExampleJames as Example2params

from tqg.example_james_woodfield import TQGExampleJames2 as Example2_params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = 1000 
    nx = 64
    dt = 0.0001 
    dump_freq = 10000
    alpha = None 
    do_save_data = True 
    do_save_visual = True 
    do_save_spectrum = False 

    PETSc.Sys.Print("time horizon: {}, res: {}, dt: {}, dump_req: {}, alpha: {}".format(T, nx, dt, dump_freq, alpha), flush=True)

    #workspace = Workspace("/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha if alpha != None else 'none'))

    workspace = Workspace("/home/wpan1/Development/Data/JamesWoodfield")
    
    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="both",comm=comm, quad=True).get_fine_mesh()
    
    params = Example2params(T,dt,mesh, bc='none', alpha=alpha)
    # params = Example3lparams(T,dt,TorusMeshHierarchy(nx,nx,1.,1.,0,"both",comm=comm).get_fine_mesh(), bc='',alpha=alpha)
    solver = TQGSolver(params)

    visual_output_name = workspace.output_name("pde_visuals", "visuals")
    data_output_name   = workspace.output_name("pde_data", "data")
    solver.solve(dump_freq, visual_output_name, data_output_name, ensemble, do_save_data=do_save_data, do_save_visual=do_save_visual, do_save_spectrum=do_save_spectrum, res=nx)
