
import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
import numpy as np
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import * #pi, ensemble,COMM_WORLD, PETSc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tqg.example_tqg_theory_paper import TQGTheoryExample as Example1params
#from tqg.example2_tqg_theory_paper import TQGTheoryExampleTwo as Example2params
from test_alpha_convergence_spectrum_control_variables import *

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


def evaluate_mean(func):
    return assemble(func * dx)



if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = 2.5 #opts['time']
    nx = 256#opts['resolution']
    res = nx
    dt = 0.0005 #opts['time_step']
    dump_freq = 40 #opts['dump_freq']
    # alphas = [8.0, 32.0, 180.0, 256.0]
    # alphas = [4.0, 16.0, 64.0, 128.0, 220.0]
    # colours =['b', 'g',  'r',  'm',   'brown']
    # alpha = opts['alpha']
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']

    workspace_none = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, 'none'))

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="both", quad=True,comm=comm).get_fine_mesh()
    params_none =  Example1params(T,dt,mesh, bc='n', alpha=None)
    solver_none  = TQGSolver(Example1params(T,dt,mesh, bc='n', alpha=None))
    all_h5_files_none = workspace_none.list_all_h5_files(sub_dir='data')

    # flag = 'stream'
    # field_name = {'stream': 'stream', 'buoyancy' : 'buoyancy'}
    # field_name = field_name[flag]
    # field_name = 'buoyancy'
    # field_name = 'stream'
    print(field_name)

    mean_series = {}     # key is alpha value
    legend_series = {}   # legend is alpha value 

    print(len(all_h5_files_none))

    for alpha in alphas:
        print("alpha ", alpha)
        # errors = np.zeros(len(all_h5_files))
        legend_series[alpha] = "alpha = {:0.1e}".format(1./alpha**2)
        mean_series[alpha] = []
        mean_series[0] = []

        for fid in range(76): # range(len(all_h5_files_none)):
            workspace = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, alpha))
            solver_alpha = RTQGSolver(Example1params(T,dt,mesh, bc='n', alpha=alpha))
            all_h5_files = workspace.list_all_h5_files(sub_dir='data')

            print('len of all_h5_files ', len(all_h5_files), len(all_h5_files_none))

            assert (len(all_h5_files) == len(all_h5_files_none))

            print("alpha {}, fid {}".format(alpha, fid))

            fname_none = workspace_none.output_name("pde_data_{}.h5".format(fid), "data")
            fname_alpha= workspace.output_name("pde_data_{}.h5".format(fid), "data")
            assert fname_none in all_h5_files_none
            assert fname_alpha in all_h5_files

            field_alpha = None
            field_none = None
            if field_name == 'stream':
                solver_alpha.solve_for_streamfunction_data_from_file(fname_alpha[:-3])
                solver_none.solve_for_streamfunction_data_from_file(fname_alpha[:-3])
                field_alpha = solver_alpha.psi0 if flag_use_regularised_psi else solver_alpha.rpsi0 
                field_none = solver_none.psi0
            elif field_name =='buoyancy' or field_name=='pv':
                solver_alpha.load_initial_conditions_from_file(fname_alpha[:-3])
                solver_none.load_initial_conditions_from_file(fname_none[:-3])
                
                if field_name =='buoyancy':
                    field_alpha = solver_alpha.initial_b
                    field_none = solver_none.initial_b
                if field_name =='pv':
                    field_alpha = solver_alpha.initial_cond
                    field_none = solver_none.initial_cond
            else:
                raise

            mean_series[alpha].append(evaluate_mean(field_alpha))
            mean_series[0].append(evaluate_mean(field_none))


    plt.rcParams.update({"text.usetex": True})
    #plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.grid(True)
    for alpha, val in mean_series.items():
        _temp_ = "{:0.1e}".format(1./alpha**2) if alpha > 0 else 'none'
        legend_series[alpha] = r'$\alpha=$' + _temp_ 
        print(np.asarray(val).shape)
        ax.plot(val, label=legend_series[alpha])

    ax.legend(loc='lower right')
    # plt.title('wavenumber {} power spectrum time series'.format(wavenumber))
    #plt.savefig(workspace_none.output_name('test_spectrum_time_series.png'.format(wavenumber), "visuals"))
    plt.savefig(workspace_none.output_name(function_mean_value_outputname(), "visuals"))
    #plt.show()
    plt.close()


