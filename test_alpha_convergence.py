import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
import numpy as np
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import * #pi, ensemble,COMM_WORLD, PETSc

from tqg.example_tqg_theory_paper import TQGTheoryExample as Example1params
#from tqg.example2_tqg_theory_paper import TQGTheoryExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = 2.5 #opts['time']
    nx = 256#opts['resolution']
    dt = 0.0005 #opts['time_step']
    dump_freq = 40 #opts['dump_freq']
    alphas = [16.0, 32.0, 64.0, 128.0, 180.0, 220.0, 256.0]
    # alpha = opts['alpha']
    do_save_data = opts['do_save_data']
    do_save_visual = opts['do_save_visual']
    do_save_spectrum = opts['do_save_spectrum']

    workspace_none = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, 'none'))

    mesh = TorusMeshHierarchy(nx, nx, 1., 1., 0, period="both", quad=True,comm=comm).get_fine_mesh()
    params_none =  Example1params(T,dt,mesh, bc='n', alpha=None)
    solver_none  = TQGSolver(Example1params(T,dt,mesh, bc='n', alpha=None))
    all_h5_files_none = workspace_none.list_all_h5_files(sub_dir='data')
    
    all_errors_b = {}
    all_errors_q = {}

    # norm_of_truth = np.zeros(len(all_h5_files_none))

    for alpha in alphas:
        print(alpha)
        workspace = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, alpha))
        solver_alpha = RTQGSolver(Example1params(T,dt,mesh, bc='n', alpha=alpha))
        
        all_h5_files = workspace.list_all_h5_files(sub_dir='data')
        assert (len(all_h5_files) == len(all_h5_files_none))

        errors_b = np.zeros(len(all_h5_files))
        errors_q = np.zeros(len(all_h5_files))

        for fid in range(len(all_h5_files)):
            fname_none = workspace_none.output_name("pde_data_{}.h5".format(fid), "data")
            fname_alpha= workspace.output_name("pde_data_{}.h5".format(fid), "data")
            assert fname_none in all_h5_files_none
            assert fname_alpha in all_h5_files

            solver_alpha.load_initial_conditions_from_file(fname_alpha[:-3])
            solver_none.load_initial_conditions_from_file(fname_none[:-3])

            # errors[fid] += errornorm(solver_alpha.initial_b, solver_none.initial_b, norm_type="H1")**2 + errornorm(solver_alpha.initial_cond, solver_none.initial_cond, norm_type="L2")**2
            reference_b = norm(solver_none.initial_b, norm_type="H1") 
            #reference_b = norm(solver_none.initial_b, norm_type="L2") 
            reference_q = norm(solver_none.initial_cond, norm_type="L2")

            errors_b[fid] = errornorm(solver_alpha.initial_b, solver_none.initial_b, norm_type="H1")/reference_b 
            #errors_b[fid] = errornorm(solver_alpha.initial_b, solver_none.initial_b, norm_type="L2")/reference_b 
            errors_q[fid] = errornorm(solver_alpha.initial_cond, solver_none.initial_cond, norm_type="L2")/reference_q
            # errors[fid] /= reference
            # norm_of_truth[fid] += norm(solver_none.initial_b, norm_type="H1") + norm(solver_none.initial_cond, norm_type="L2")**2

        all_errors_b[alpha] = errors_b
        all_errors_q[alpha] = errors_q

    assert len(alphas) == len(all_errors_b.items())
    assert len(alphas) == len(all_errors_b.items())


    # plotting ###########################################################################

    import matplotlib.pyplot as plt
    linestyle_tuple = {
         'loosely dotted':        (0, (1, 10)),
         #'dotted':                (0, (1, 1)),
         'densely dotted':        (0, (1, 1)),
         'loosely dashed':        (0, (5, 10)),
         #'dashed':                (0, (5, 5)),
         'densely dashed':        (0, (5, 1)),
         'loosely dashdotted':    (0, (3, 10, 1, 10)),
         'dashdotted':            (0, (3, 5, 1, 5)),
         'densely dashdotted':    (0, (3, 1, 1, 1)),
         'dashdotdotted'      :   (0, (3, 5, 1, 5, 1, 5)),
         'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
         'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)),
         'solid': 'solid',
         'dotted': 'dotted',
         'dashed': 'dashed',
         'dashdot': 'dashdot'}

    def plot_relative_errors(data, ylabel, title, outputname, flag):
        line_styles = {16.0:'dashed', 32.0:'dashdot', 64.0:'dashdotdotted', 128.0:'solid', 180.0:'densely dashdotted', 220.0:'dotted'}
        plt.rcParams.update({"text.usetex": True, "font.family":"serif"}) #,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
        plt.rc('font', size=14)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=18)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('legend', fontsize=18)
        f, ax1 = plt.subplots(1,1, figsize=(8,6), dpi=300)

        #x_values = (np.cumsum(np.ones(len(all_h5_files_none)))-1) * dt * dump_freq 
        x_values = (np.cumsum(np.ones(len(all_h5_files_none)))-1) * dump_freq
        #for k,v in all_errors_b.items():
        #for k,v in data.items():
        for k in [16.0, 32.0, 64.0, 128.0, 220.0]:
            #k is alpha
            v = data[k]
            ax1.plot(x_values[:71], v[:71], label=r'$\alpha=1/{}^2$'.format(int(k)), lw=1.3, linestyle=linestyle_tuple[line_styles[k]], color='black')

        #ax1.set_xlabel('time (t)',labelpad=10)
        ax1.set_xlabel('Number of time steps',labelpad=10)
        #ax1.set_ylabel(r'$\|b - b^\alpha \|_{H^1}/\|b\|_{H^1}$')
        ax1.set_ylabel(ylabel, labelpad=20)
        if flag == 'pv':
            ax1.set_ylim([0,1.8])
        elif flag == 'buoyancy':
            ax1.set_ylim([0, 1.45])
        #ax1.set_xlim([0,1.4])
        ax1.set_xlim([0,70*dump_freq])
        ax1.set_title(title)
        ax1.grid(True)
        ax1.legend(loc='upper left')

        plt.savefig(workspace_none.output_name(outputname, "visuals"))
        plt.close()

    plot_relative_errors(all_errors_b, r'$\|b - b^\alpha \|_{H^1}/\|b\|_{H^1}$', 'Relative error - buoyancy', 'relative_error_buoyancy.png', 'buoyancy')
    plot_relative_errors(all_errors_q, r'$\|\omega - \omega^\alpha \|_{L^2}/\|\omega\|_{L^2}$', 'Relative error - potential vorticity', 'relative_error_pv.png', 'pv')

    def plot_alpha_convergence(data, ylabel, title, outputname, flag):
        plt.rcParams.update({"text.usetex": True, "font.family":"serif"}) #,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
        plt.rc('font', size=14)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=18)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        plt.rc('legend', fontsize=15)
        f, ax1 = plt.subplots(1,1,figsize=(8, 6), dpi=300)
        x_values = 1./ np.asarray(alphas)**2 
        x_ = np.zeros(len(alphas))
        # assert np.linalg.norm((x_ - x_values), ord=1) < 0.000000001
        _times = [0.3, 0.4, 0.5, 0.6]

        line_styles = {0.3:'dashed', 0.4:'dashdot', 0.5:'dashdotdotted', 0.6:'dotted'}
        marker_styles = {0.3:'x', 0.4:'+', 0.5:'2', 0.6:'^'}
        #_legend = [r't=0.3', r't=0.4', r't=0.5', r't=0.6']  # 
        _legend = ['No. of time steps = 600', 'No. of time steps = 800', 'No. of time steps = 1000', 'No. of time steps = 1200']  # 

        j = 0  # index for time index
        for t in _times:
            _index_values = [int( t / dt / dump_freq) for t in _times]
            y_ = np.zeros(x_.shape)
            i = 0  # index for alpha values
            for k, v in data.items():
                x_[i] = 1./k/k  # alpha
                y_[i] = v[_index_values[j]]
                i += 1
            ax1.loglog(x_, y_, linestyle=linestyle_tuple[line_styles[t]], marker=marker_styles[t], color='black', lw=1.3, label=_legend[j])
            j += 1

        # plot reference order 1 alpha
        i = 0
        x_ = np.zeros(len(alphas))
        #y_ = np.zeros(len(alphas))
        for k, v in data.items():
            x_[i] = 1./k/k  # k = alpha # in the equation we have alpha = alpha^2
            i += 1
        ax1.loglog(x_[3:], x_[3:] * (1000 if flag =='buoyancy' else 300), linestyle=linestyle_tuple['densely dashdotted'], color='black', lw=1, label=r'$O(\alpha)$')

        ax1.set_xlabel(r'$\alpha$', labelpad=5)
        ax1.set_ylabel(ylabel, labelpad=10)
        ax1.grid(True)
        ax1.legend(loc='lower right')
        ax1.title.set_text(title)

        plt.savefig(workspace_none.output_name(outputname + ".png", "visuals"))
        plt.close()
    
    plot_alpha_convergence(all_errors_b, r'$\|b - b^\alpha \|_{H^1}/\|b\|_{H^1}$', 'Convergence - buoyancy relative error, log-log plot', 'relative_error_convergency_buoyancy', 'buoyancy')
    plot_alpha_convergence(all_errors_q, r'$\|\omega - \omega^\alpha \|_{L^2}/\|\omega\|_{L^2}$', 'Convergence - PV relative error, log-log plot', 'relative_error_convergency_pv', 'pv')

