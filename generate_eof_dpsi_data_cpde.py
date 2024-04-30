# Generates dpsi data for eof computation

from firedrake import *
from utility import Workspace
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake_utility import TorusMeshHierarchy
# from firedrake import pi, ensemble,COMM_WORLD, PETSc
from firedrake.petsc import PETSc

from tqg.example2 import TQGExampleTwo as Example2params
import numpy as np

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm

cpde_directory = "cPDESolution128"
cpde_calibration_directory = "CalibrationData128"


def load_from_file(fdrake_function, filename, fieldname):
    with DumbCheckpoint(filename, mode=FILE_READ) as chk:
        chk.load(fdrake_function, name=fieldname)

def save_to_file(fdrake_function, filename):
    with DumbCheckpoint(filename, single_file=True, mode=FILE_CREATE) as chk:
        chk.store(fdrake_function)

def psi_file_name(file_index, workspace):
    return workspace.output_name("cpde_data_{}".format(file_index), cpde_directory)

def rpsi_file_name(file_index, workspace, increment_id=None):
    _fname_= "_regularised_psi_increment_{}".format(file_index)
    _fname = _fname_ + "_{}".format(increment_id) if increment_id !=None else _fname_ 
    return workspace.output_name(_fname, cpde_calibration_directory) 

if __name__ == "__main__":
    nx = 512
    cnx =128 
    T = 0.001  # 0.001 = ~30 minutes
    dt = 0.0002 # 0.0002 ~ twice fine dt ~ 3.5 minutes
    dump_freq = 5
    solve_flag = True 
    write_visual = False 
    psi_name = "Streamfunction"
    rpsi_reg_name = "StreamfunctionRegularised"
    rpsi_name = "Streamfunction"
    #grid_point = [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]] 

    procno = ensemble_comm.rank 
    visual_output_name = ''
    #mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y", comm=comm).get_fine_mesh()
    cmesh = TorusMeshHierarchy(cnx, cnx, 1., 1., 0, period="y", comm=comm).get_fine_mesh()

    #params = Example2params(T,dt,mesh, bc='x', alpha=None)
    #params_reg = Example2params(T,dt,cmesh, bc='x', alpha=None)
    params_c = Example2params(T, dt, cmesh, bc='x', alpha=None)

    #fname =       "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, alpha)
    #input_fname = "/home/wpan1/Data/PythonProjects/TQGExample2/res_{}_alpha_{}".format(nx, 'none')
    input_fname = "/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin"

    workspace = Workspace(input_fname)
    #streamfunction_increment_error_gridfname = workspace.output_name("streamfunction_increment_error_grid_value_time_series", "visuals")
    #streamfunction_increment_regularised_error_gridfname = workspace.output_name("__streamfunction_increment_regularised_error_grid_value_time_series", "visuals")
    #streamfunction_increment_error_time_series = []
    #streamfunction_increment_regularised_error_time_series = []
    
    myprint = PETSc.Sys.Print

    #if procno == 0 and write_visual: 
    #    output_file = File(workspace.output_name("streamfunction_increments","visuals") + ".pvd")

    for f_index in range(0,4275-2775):  

        psi_filename_t_n = psi_file_name(f_index, workspace) # workspace.output_name("_pde_data_{}".format(f_index), "data")
        psi_filename_t_np = psi_file_name(f_index+1, workspace)

        rpsi_filename = rpsi_file_name(f_index, workspace) # workspace.output_name("_regularised_psi_increment_{}".format(f_index), "data")
        rpsi_filename_t_n = rpsi_file_name(f_index, workspace, 0)
        rpsi_filename_t_np = rpsi_file_name(f_index, workspace, 1)

        myprint(psi_filename_t_n)

        solver = TQGSolver(params_c)
        #solver_reg = RTQGSolver(params_reg) 
        #solver_c = TQGSolver(params_c)

        # RTQGSolver is used for StreamfunctionRegularised 
        solver_c = RTQGSolver(params_c)

        # solver_reg starts at t(n), and solves until t(n+1)
        if solve_flag:
            #solver_reg.load_initial_conditions_from_file(psi_filename_t_n)
            #solver_reg.solve(dump_freq, visual_output_name, rpsi_filename, ensemble_comm, do_save_data=True, do_save_visual=False, do_save_spectrum=False, res=0)

            # coarse grain truth
            # solve cpde using coarse graned truth
            solver_c.load_initial_conditions_from_file(psi_filename_t_n)

            # res=0 below is for meshgrid, which is used for spectrum plotting 
            solver_c.solve(dump_freq, visual_output_name, rpsi_filename, ensemble_comm, do_save_data=True, do_save_visual=False, do_save_spectrum=False, res=0)

        if 0:
            psi_t_np = Function(solver.psi0.function_space(), name="Streamfunction")
            psi_t_n  = Function(solver.psi0.function_space())
            rpsi_t_np = Function(solver_c.psi0.function_space())
            rpsi_t_n  = Function(solver_c.psi0.function_space())
            rpsi_reg_t_np = Function(solver_c.psi0.function_space(), name="StreamfunctionReg")
            rpsi_reg_t_n  = Function(solver_c.psi0.function_space())
            # dpsi_error = Function(solver.psi0.function_space(), name="StreamfunctionIncrementError")
            dpsi_reg_error = Function(solver.psi0.function_space(), name="StreamfunctionIncrementError")
                    
            load_from_file(psi_t_np, psi_filename_t_np , psi_name)
            load_from_file(psi_t_n , psi_filename_t_n , psi_name)
            load_from_file(rpsi_t_np, rpsi_filename_t_np , rpsi_name)
            load_from_file(rpsi_t_n,  rpsi_filename_t_n , rpsi_name)
            load_from_file(rpsi_reg_t_np, rpsi_filename_t_np, rpsi_reg_name)
            load_from_file(rpsi_reg_t_n,  rpsi_filename_t_n , rpsi_reg_name)

            if procno == 0 and write_visual:
                output_file.write(psi_t_np, rpsi_reg_t_np , time=f_index)
            # dpsi_error.assign(0.5*dt*dump_freq*((psi_t_np + psi_t_n) - (rpsi_t_n + rpsi_t_np)))
            # myprint(norm(dpsi_error), norm(psi_t_np - psi_t_n), 0.5*dt*dump_freq*norm(rpsi_t_n + rpsi_t_np))
            # dpsi_reg_error.assign(0.5*dt*dump_freq*((psi_t_np + psi_t_n) - (rpsi_reg_t_n + rpsi_reg_t_np))) # integral error, fname: _
            # myprint(norm(dpsi_reg_error), 0.5*dt*dump_freq*norm(psi_t_np + psi_t_n), 0.5*dt*dump_freq*norm(rpsi_reg_t_np + rpsi_reg_t_n), "\n")

            # dpsi_reg_error.assign((psi_t_np - psi_t_n) - (rpsi_reg_t_np - rpsi_reg_t_n)) # white noisefname: no prefix 
            # myprint(norm(dpsi_reg_error), norm(psi_t_np - psi_t_n), norm(rpsi_reg_t_np - rpsi_reg_t_n))

            dpsi_reg_error.assign(psi_t_np - rpsi_reg_t_np) # white noise, fname: __
            myprint(norm(dpsi_reg_error), norm(psi_t_np), norm(rpsi_reg_t_np))

            # dpsi_error_fname = workspace.output_name("_streamfunction_increment_error_{}".format(f_index), "data")  
            # dpsi_reg_error_fname = workspace.output_name("_streamfunction_increment_regularised_error_{}".format(f_index), "data") 
            # save_to_file(dpsi_error, dpsi_error_fname)  # _streamfunction_increment_regularised_error_{}
            #save_to_file(dpsi_reg_error, dpsi_reg_error_fname)  # _streamfunction_increment_regularised_error_{}
            
            # streamfunction_increment_error_time_series.append(dpsi_error.at(grid_point, tolerance=1e-10))
    #        streamfunction_increment_regularised_error_time_series.append(dpsi_reg_error.at(grid_point, tolerance=1e-10))
#
#myprint(len(streamfunction_increment_regularised_error_time_series))
#
#if procno == 0:
#    np.save( streamfunction_increment_regularised_error_gridfname , np.asarray(streamfunction_increment_regularised_error_time_series))
#    # np.save( streamfunction_increment_error_gridfname , np.asarray(streamfunction_increment_error_time_series))

