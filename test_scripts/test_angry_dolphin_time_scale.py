from relative_import_header import *
from firedrake import *
from utility import Workspace
from firedrake_utility import TorusMeshHierarchy
from tqg.solver import TQGSolver
#import numpy as np
from random import randint, sample
from tqg.example2 import TQGExampleTwo as Example2params
import matplotlib.pyplot as plt


desired_spatial_rank = 1
ensemble = ensemble.Ensemble(COMM_WORLD, desired_spatial_rank)
spatial_comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


def pde_data_fname(wspace, file_index, sub_dir=''):
    return wspace.output_name("pde_data_{}".format(file_index), sub_dir) 


class TimeScale(TQGSolver):

    """ We assume tqg_params is defined using PDE solver settings """
    def __init__(self, tqg_params, ensemble_size, comm_manager, input_data_dir=''):
        ensemble_comm = comm_manager.ensemble_comm
        spatial_comm  = comm_manager.comm

        TQGSolver.__init__(self, tqg_params)
        
        self.file_index_range_for_perturbation = range(2000, 2775)  # file indices for perturbation
        self.file_index_range_for_simulation   = range(0, 1500)
        self.file_work_space = Workspace("/home/wpan1/Data2/PythonProjects/DataAssimilationAngryDolphin")

        if ensemble_comm.rank == 0:
            print("load initial conditions", flush=True)

        #self.load_initial_conditions_from_file(pde_data_fname(self.file_work_space, 0, "PDESolution"), spatial_comm)

        if ensemble_comm.rank == 0:
            print("initialiser", flush=True)
        
        #print("ensemble_comm rank: ", ensemble_comm.rank, ", norm of pv: " , norm(self.initial_cond))

def plot_values(workspace):
    data = np.load(workspace.output_name("dxs.npy", "TestFiles"))
    fig, ax = plt.subplots()
    ax.set(xlabel='data file index', ylabel='1 time step arc length')
    xs = np.arange(1,len(data)+1)
    ax.scatter(xs, data, s=0.1)
    
    plt.savefig(workspace.output_name("dxs.png", "TestFiles"))

    


if __name__ == "__main__":
    nx = 512
    cnx = 256 #64
    T = 0.05
    dt = 0.0001
    dump_freq =50 
    solve_flag = False
    write_visual = False 

    ensemble_size = ensemble_comm.size 


    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="y",comm=spatial_comm).get_fine_mesh()
    
    # It doesn't matter what the actual params are, we just need a param object to pass to the Solver constructor
    angry_dolphin_params = Example2params(T, dt, mesh, bc='x', alpha=None)
    
    tsc = TimeScale(angry_dolphin_params, ensemble_size, ensemble)

    if 0:
        velocity = Function(tsc.Vu)
        
        dxs = np.zeros(len(tsc.file_index_range_for_simulation))

        for findex in tsc.file_index_range_for_simulation: 
            tsc.load_initial_conditions_from_file(pde_data_fname(tsc.file_work_space, findex, "PDESolution"), spatial_comm)
            velocity.project(tsc.gradperp(tsc.psi0))
            #print(norm(velocity, norm_type="L1")*dt)
            dxs[findex] += norm(velocity, norm_type="L1")*dt
            print(findex, ", ", dxs[findex])

        np.save(tsc.file_work_space.output_name("dxs.npy", "TestFiles"), dxs)
    else:
        plot_values(tsc.file_work_space)


    

