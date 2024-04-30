import sys
from firedrake_utility import TorusMeshHierarchy
from utility import Workspace, commandline_parser
import numpy as np
from tqg.solver import TQGSolver
from tqg.solver_regularised import RTQGSolver
from firedrake import * #pi, ensemble,COMM_WORLD, PETSc
import matplotlib.pyplot as plt

from tqg.example_tqg_theory_paper import TQGTheoryExample as Example1params
#from tqg.example2_tqg_theory_paper import TQGTheoryExampleTwo as Example2params

ensemble = ensemble.Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = ensemble.comm
ensemble_comm = ensemble.ensemble_comm


if __name__ == "__main__":

    opts = commandline_parser(sys.argv[1:])

    T = 2.5 #opts['time']
    nx = 256#opts['resolution']
    res = nx
    dt = 0.0005 #opts['time_step']
    dump_freq = 40 #opts['dump_freq']
    alphas = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 180.0, 220.0, 256.0]
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
    field_name = 'stream'
    print(field_name)

    for fid in [0]: #[32, 75]: # 75 corresponds to t=1.5   # range(len(all_h5_files)):
        # errors = np.zeros(len(all_h5_files))
        ps_series = []
        x_series  = []
        legend_series = []
        for alpha in alphas:

            workspace = Workspace("/home/wpan1/Data2/PythonProjects/TQG_Theory_Example/res_{}_alpha_{}".format(nx, alpha))
            solver_alpha = RTQGSolver(Example1params(T,dt,mesh, bc='n', alpha=alpha))
            all_h5_files = workspace.list_all_h5_files(sub_dir='data')

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
                field_alpha = solver_alpha.psi0
                field_none = solver_none.psi0
            elif field_name =='buoyancy':
                solver_alpha.load_initial_conditions_from_file(fname_alpha[:-3])
                solver_none.load_initial_conditions_from_file(fname_none[:-3])
                field_alpha = solver_alpha.initial_b
                field_none = solver_none.initial_b

            # reference_b = norm(solver_none.initial_b, norm_type="H1") 
            # reference_q = norm(solver_none.initial_cond, norm_type="L2")

            # error_field = assemble((solver_none.initial_b - solver_alpha.initial_b)**2)
            error_field = assemble(field_alpha - field_none)

            # setup fourier transform
            
            xx = np.linspace(0, 1, res+1)
            yy = np.linspace(0, 1, res+1)

            mygrid = []
            for i in range(len(xx)):
                for j in range(len(yy)):
                    mygrid.append([xx[i], yy[j]])
            mygrid = np.asarray(mygrid) # now i have a list of grid values

            grid_values = np.asarray(error_field.at(mygrid, tolerance=1e-10)) 
            grid_values = grid_values.reshape(res+1, res+1).T

            # meshgrid for contour plot

            fig = plt.figure(1)
            fig, axes = plt.subplots(1, 1)
            plt.subplots_adjust(hspace=1)
            [X,Y] = np.meshgrid(xx, yy)
            axes.contour(X,Y,grid_values)
            axes.set_title('error_{}_meshgrid_values, alpha{:0.1e} fid{}'.format(field_name, 1./alpha**2,fid))

            # plt.savefig("./test_fdrake_grid_values.png")
            plt.savefig(workspace_none.output_name("{}_error_grid_values_alpha_{}_fid_{}.png".format(field_name, alpha, fid), "visuals"))
            plt.close()

            fig = plt.figure(2)
            # Fourier analysis working with grid_values
            sampled_values = grid_values[:-1,:-1] # need to get rid off the last point for periodicity
            fourier_transform = 2*np.fft.fft2(sampled_values)
            ps = np.abs(fourier_transform[:int(res/2),:int(res/2)])**2
            # FS = scipy.fft.fftn(S)
            #plt.imshow(np.log(np.abs(scipy.fft.fftshift(FS))**2))
            plt.title("error {} power spectrum, alpha{:0.1e} fid{}".format(field_name, 1./alpha**2,fid))
            plt.imshow(np.log(ps))
            # plt.savefig("./test_fdrake_spectrum.png")
            plt.savefig(workspace_none.output_name("{}_error_spectrum_alpha_{}_fid_{}.png".format(field_name,alpha, fid), "visuals"))
            plt.close()


            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            k_val_dic = {}
            for i in range(ps.shape[0]):
                for j in range(ps.shape[1]):
                    # ksquared = i**2 + j**2
                    k = np.floor(np.sqrt(i**2 + j**2))
                    # if (ksquared in k_val_dic):
                    if (k in k_val_dic):
                        # k_val_dic[ksquared].append(ps[i,j])
                        k_val_dic[k].append(ps[i,j])
                    else:
                        # k_val_dic[ksquared] = [ps[i,j]]
                        k_val_dic[k] = [ps[i,j]]
            k_val_dic_sorted = sorted(k_val_dic.items(), key = lambda item: item[0])
            x, y = [], []
            for key, val in k_val_dic_sorted:
                #x.append(np.sqrt(key))
                x.append(key)
                y.append(np.mean(val))

            ps_series.append(y)
            x_series.append(x)
            legend_series.append("alpha = {:0.1e}".format(1./alpha**2))

            ax.loglog(x, y)
            # ax.plot(x,y)
            # ax.set_ylim(5e-8,1)
            ax.set_xlabel("log(k)")
            ax.set_ylabel("log(E(k))")
            plt.title("error {} power spectrum, alpha{:0.1e} fid{}".format(field_name,1./alpha**2,fid))
            # plt.savefig("./test_euler_energy_spectrum_radial_{}.png".format(file_index))
            plt.savefig(workspace_none.output_name("{}_error_spectrum_radial_alpha_{}_fid_{}.png".format(field_name,alpha, fid), "visuals"))
            plt.close()
        
        # plt.rcParams.update({"text.usetex": True,"font.family": "sans-serif","font.sans-serif": ["Helvetica"]})
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        for i in range(len(legend_series)):
            # ax.loglog(x_series[i], ps_series[i], label=legend_series[i])
            ax.plot(x_series[i], ps_series[i], label=legend_series[i])
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend(loc='upper right')
        #ax.legend(loc='lower left')
        
        plt.title("error {} power spectrums, fig{}".format(field_name,fid))
        plt.savefig(workspace_none.output_name("{}_error_spectrum_radial_fid_{}_.png".format(field_name,fid), "visuals"))
        plt.close()

        print(np.asarray(ps_series).shape)










